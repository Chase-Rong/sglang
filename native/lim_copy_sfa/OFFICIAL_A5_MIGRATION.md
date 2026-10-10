# Official A5 algorithm migration to GLM-5.3-Flash

Source snapshots supplied by the user:

| Operator | PR | SHA256 of downloaded diff |
| --- | --- | --- |
| FusedQuantLightningIndexerManage | https://gitcode.com/cann/ops-transformer/pull/12358/ | `57dbf433526a3e18836b98d15c66cc4344f08333aa8a188204bca23b27d37126` |
| FusedScatterCopyQuantSparseFlashAttention | https://gitcode.com/cann/ops-transformer/merge_requests/12202 | `ccba9ea804f7bed0bdaf710790bbd06c9d36f44c05f466668f6a369ff4a1522d` |

The migration adapts algorithms, not the C8 ABI. The official LIM computes
FP8 per-token Top2048 and supports at most four query routes. Its CopySFA
expects a 576-element query and a 656-byte packed C8 KV row. GLM-5.3-Flash
continues to compute BF16 pooled Top512, expand four independent tokens per
pool, retain up to three causal tail tokens, and attend over BF16 NoPE512.
Changing those contracts would change model semantics.

## Migrated mechanisms

- PR12358 classifies selected candidates and compacts request misses. The
  pooled adapter classifies all 2051 expanded tokens with a 256-thread SIMT
  function while retaining independent source, slot and physical-address
  fields. It does not truncate the source ID to the official 18-bit payload.
- Request union contains payload misses only. Selected slots from every query
  are protected before allocation. Free slots are consumed first; remaining
  victims use the PR's length-hashed scan and exact protected-slot exclusion.
  The adapter scans resident slots instead of the full source domain, using
  its existing reverse map. It does not sort all resident scores. This changes
  cache replacement, not TopK membership or the attention equation.
- The official threshold-based victim shortcut depends on C8 score thresholds;
  the adapter uses exact protected-slot exclusion directly. No speculative
  score threshold or contiguous four-slot assumption is introduced.
- Union outputs publish only their valid prefix. `state[owner,3]` bounds every
  consumer; bytes after this count are unspecified, as in the official ABI.
- PR12202's request-level union threshold is 512. If any request exceeds it,
  or requires generation/rollback first fill, the kernel copies the batch's
  miss unions before attention. Otherwise each query gathers its misses as
  it computes. The decision is based on runtime union counts, not batch shape.
- Four 512-token groups use the official request-based permutation. The three
  KPool tail positions remain outside that permutation.
- During hit groups, each AIV submits up to four owned miss rows, overlaps
  those reads with attention computation, and completes their HBM persistence
  before reusing the prefetch UB or consuming the prefetched slots.
- Earlier-query miss ownership suppresses duplicate MTP writebacks. A query
  that does not own persistence still reads Host data; it never assumes another
  AIV has completed the HBM write. Positive local plans are published only for
  this AIV's completed prefetch, while global validity waits for all consumers.
- Adjacent UB writeback rows are paired only when their destination byte gap
  is representable, without reordering payloads. BF16512 uses a 1024-byte
  payload and a 1152-byte gather row, replacing the official 656/672 sizes.

## Preserved integration and boundaries

HiCache/MemFabric allocation, Host page tables, Prefill offload, PD transport,
current-token Host publication, graph lifetime and shared-indexer routing are
unchanged. Each physical attention layer still owns its payload validity.
Generation changes, rollback, mutable MTP tails, remaps and inactive Graph rows
remain part of the resident contract. One target's successful copy cannot make
another layer's payload valid.

The BF16 score/TopK math and multi-core sequence partition remain based on the
pinned CANN BF16 Indexer. The C8 PR's score-tile packed source/slot payload is
not directly applicable to KPool4. Candidate classification occurs after the
BF16 TopK join; this migration must not be described as complete overlap of
all residency work with Indexer matrix computation.

The native extension schema is unchanged, but union suffix semantics changed.
Use the matching Python adapter and rebuild the native extension in a fresh
build directory. Mixing old count-ignoring commit helpers with the new binary
is unsupported. This branch defaults to `lim_copy_sfa` when fused offload is
enabled. Set `SGLANG_NPU_DSA_OFFLOAD_BACKEND=external_plan` to select the original
LRU planner and FSA. This default change does not constitute E2E qualification.

## Validation

Host tests: `test/registered/unit/npu/test_lim_official_schedule.py` exercises
protected replacement across one/four/six routes, high source IDs, remaps,
generation transitions and reshuffles. Its C++ test executes the actual
schedule header at the 512/513 boundary and checks the gather permutation.

NPU gates: native planner, fused Indexer, native pair/framework and CopySFA
transfer tests under `test/manual/npu/`. Transfer cases include 512/513 misses,
all-hit/all-miss, cold fills and B10Q6. The scattered 80%-hit case retains 1638
random resident sources, adds 410 misses and independently shuffles each query's
2048 entries; its attention oracle gathers the actual selected source IDs.
Run correctness before collecting warm Graph samples. Shared-card component
latency is not E2E TPOT acceptance.

### Qualification on 2026-10-10

- Local CPU/contract tests: 59 passed, one skipped because the Windows host has
  no C++ compiler. The skipped schedule-header test passed remotely.
- Remote final gate: 56 passed (eight Host tests and 48 NPU tests). This includes
  the standalone BF16 Indexer, fused selection, supplied-TopK management,
  native pair, short prefix in a larger reservation, 1/32/64 attention heads,
  inactive Graph rows, remapping, rollback and poisoned union suffixes.
- Host: `141.61.54.135`, container: `jzx_sglang_5.3_flash2`, CANN 9.1,
  Python 3.12.13. Tests cap NPU allocation at 2 GiB per process. Other users'
  services were neither stopped nor modified. Follow-up timing shares busy
  cards and is exploratory, not isolated-card performance acceptance.
- New extension: `/home/j00628475/lim_official_pr_20261010/build-v6/libsgl_lim_copy_sfa.so`,
  SHA256 `b826a20b4b63ec4c6034a15dba86d3ca9762756863f7d4263b2283308fc87149`.
- Previous extension: `/home/j00628475/lim_copy_sfa_ut_20261008/native-build-copysfa-v6/libsgl_lim_copy_sfa.so`,
  SHA256 `e8c447dc30ba356e3af5630e95dd2efd58d2dd1fbae88e03d36f7022cba1523d`.
- Matching source/adapters and final tests reside under
  `/home/j00628475/lim_official_pr_20261010/source`. The compiled v6 and final
  source differ only by Python formatting; that formatting does not change
  the generated launch ABI or kernel algorithms.
- Logs, samples and the comparison harness are retained in the workspace's
  `results/official_pr_20261010/`, excluded from production changes.
- Final ABBA passed 18 cases per round. Compared within the new rounds,
  LIM+CopySFA reduced full component-chain latency versus external_plan by
  23.47% (B1Q1), 7.01% (B1Q6), 23.53% (B10Q1) and 27.49% (B10Q6).
  These fixed-input warm Graph measurements use 16K context and are not a
  full MTP step or serving TPOT. B10Q6 cold CopySFA regressed 11.9% versus
  the previous CopySFA. Full results and limitations are in
  `results/official_pr_20261010/FINAL_COMPARISON.md`.

The SIMT classifier needs dynamic UB partitioning. The default SDK direct
launcher produced a vector UB bounds fault. `prepare_configured_launch.py`
preserves the generated argument layout and embeds the matching mixed-core ELF;
`configured_launch.h` uses `aclrtLaunchKernelWithHostArgs` with 216 KiB dynamic
UB and batch scheduling. Device-only argument launch is not interchangeable
with this Host argument buffer. The six-route manager reuses dead sorting UB
for victims rather than exceeding the 216 KiB partition.
