"""Write Host page mappings directly into stable graph buffers."""

import triton
import triton.language as tl


@triton.jit
def _host_page_table_kernel(
    device_table,
    mapping,
    output,
    rows: tl.constexpr,
    columns: tl.constexpr,
    output_columns: tl.constexpr,
    mapping_size: tl.constexpr,
    input_row_stride: tl.constexpr,
    input_column_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    output_column_stride: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * block + tl.arange(0, block)
    valid_input = (row < rows) & (column < columns)
    page = tl.load(
        device_table + row * input_row_stride + column * input_column_stride,
        valid_input,
        other=0,
    ).to(tl.int64)
    valid_page = valid_input & (page > 0) & (page < mapping_size)
    safe_page = tl.where(valid_page, page, 0)
    host_page = tl.load(mapping + safe_page)
    tl.store(
        output + row * output_row_stride + column * output_column_stride,
        tl.where(valid_page, host_page, 0),
        column < output_columns,
    )


def fill_host_page_table(device_table, mapping, output):
    if output.numel():
        _host_page_table_kernel[(output.shape[0], triton.cdiv(output.shape[1], 256))](
            device_table,
            mapping,
            output,
            device_table.shape[0],
            device_table.shape[1],
            output.shape[1],
            mapping.numel(),
            *device_table.stride(),
            *output.stride(),
            block=256,
        )
    return output
