---
title: 'Module: jax/experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.transposed_ragged_dot_mgpu`/
symbols:
  transposed_ragged_dot: transposed_ragged_dot().
  main: main().
  transposed_ragged_dot.body.mn_loop.acc_scope.block_matmul: transposed_ragged_dot().body().mn_loop().acc_scope().block_matmul().
  transposed_ragged_dot.body.mn_loop: transposed_ragged_dot().body().mn_loop().
  transposed_ragged_dot.body.mn_loop.acc_scope: transposed_ragged_dot().body().mn_loop().acc_scope().
  transposed_ragged_dot.body.mn_loop.store_scope: transposed_ragged_dot().body().mn_loop().store_scope().
  ref_transposed_ragged_dot: ref_transposed_ragged_dot().
  transposed_ragged_dot.body: transposed_ragged_dot().body().
---
# Module: [`jax/experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.py)

## Functions
- `acc_scope(acc_ref)` — [`L101`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.py#L101)
- `block_matmul(block_idx, lhs_smem, rhs_smem)` — [`L102`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.py#L102)
- `body(group_sizes_gmem, group_starts_gmem, group_ends_gmem, group_num_blocks_gmem, group_block_starts_gmem, lhs_gmem, rhs_gmem, o_gmem)` — [`L72`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.py#L72)
- `main(unused_argv)` — [`L225`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.py#L225)
- `mn_loop(loop_info: plgpu.NDLoopInfo)` — [`L87`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.py#L87)
- `ref_transposed_ragged_dot(lhs, rhs, group_sizes)` — [`L214`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.py#L214)
- `store_scope(o_smem)` — [`L179`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.py#L179)
- `transposed_ragged_dot(lhs, rhs, *, group_sizes, block_m: int, block_n: int, block_k: int, max_concurrent_steps: int, grid_block_n: int)` — [`L30`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.py#L30)

