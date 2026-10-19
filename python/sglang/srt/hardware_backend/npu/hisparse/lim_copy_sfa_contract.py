"""Shape and storage contract for the GLM5Next dual-operator backend."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LimCopySfaSpec:
    max_source_tokens: int
    max_query_routes: int
    resident_tokens: int
    index_topk: int = 2048
    pool_size: int = 4
    page_size: int = 128
    latent_dim: int = 512
    rope_dim: int = 0

    def __post_init__(self):
        if self.index_topk != 2048 or self.pool_size != 4:
            raise ValueError("A5 GLM5Next LIM requires TopK2048 and KPool4")
        if self.page_size != 128 or self.latent_dim != 512 or self.rope_dim != 0:
            raise ValueError("A5 CopySFA requires page128 and NoPE512")
        if not 1 <= self.max_query_routes <= 6:
            raise ValueError("Initial GLM5Next LIM supports one to six query routes")
        if not 0 < self.max_source_tokens <= 1 << 21:
            raise ValueError("LIM source IDs must fit the 21-bit source range")
        required = min(self.max_source_tokens, self.max_query_routes * self.token_topk)
        required = self.align(required)
        if (
            self.resident_tokens < required
            or self.resident_tokens > 32640
            or self.resident_tokens % self.page_size
        ):
            raise ValueError(
                f"Resident tokens must be page aligned in [{required}, 32640]"
            )

    def align(self, value):
        return (value + self.page_size - 1) // self.page_size * self.page_size

    @property
    def token_topk(self):
        return self.index_topk + self.pool_size - 1

    @property
    def pooled_topk(self):
        return self.index_topk // self.pool_size

    @property
    def source_capacity(self):
        return self.align(self.max_source_tokens)

    def storage_bytes(self, request_capacity, layers, graph_rows=0):
        """Conservative tensor storage; excludes backend Sort/attention workspace."""
        if request_capacity < 1 or layers < 1 or graph_rows < 0:
            raise ValueError("Invalid LIM storage geometry")
        rows = max(request_capacity * self.max_query_routes, graph_rows)
        mapping = request_capacity * self.source_capacity * 4
        resident_state = request_capacity * (self.resident_tokens * 12 + 16)
        payload = request_capacity * self.resident_tokens * self.latent_dim * 2
        outputs = rows * (
            self.token_topk * 8
            + 16400 * 2
            + ((self.token_topk + 8) // 8 * 8) * 4
            + self.resident_tokens // self.page_size * 4
            + 32 * 4
        )
        union_width = 1 << (self.max_query_routes * self.token_topk - 1).bit_length()
        union = request_capacity * (union_width * 32 + self.resident_tokens * 20 + 4)
        scores = rows * (self.source_capacity // self.pool_size) * 4
        return dict(
            mapping=mapping * layers,
            resident_state=resident_state * layers,
            payload=payload * layers,
            outputs=outputs * layers,
            union=union * layers,
            score_workspace=scores,
        )


def resolve_dsa_offload_backend(value):
    if value not in ("external_plan", "lim_copy_sfa"):
        raise ValueError(
            "SGLANG_NPU_DSA_OFFLOAD_BACKEND must be external_plan or lim_copy_sfa"
        )
    return value
