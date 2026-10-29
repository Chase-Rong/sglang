add_ops_compile_options(FusedSparseAttentionOverlap OPTIONS
 --cce-auto-sync=off -Wno-deprecated-declarations
 -mllvm -cce-vf-remove-membar=false -mllvm -cce-aicore-hoist-movemask=false)
add_kernels_compile()
