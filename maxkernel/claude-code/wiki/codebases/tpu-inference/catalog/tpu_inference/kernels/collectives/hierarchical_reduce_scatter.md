---
title: 'Module: tpu_inference/kernels/collectives/hierarchical_reduce_scatter.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/collectives/hierarchical_reduce_scatter.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.collectives.hierarchical_reduce_scatter`/
symbols:
  hierarchical_reduce_scatter_local: hierarchical_reduce_scatter_local().
  _next_multiple_of: _next_multiple_of().
  _accumulate: _accumulate().
  _get_hypercube_chunk_idx: _get_hypercube_chunk_idx().
  hier_rs_kernel: hier_rs_kernel().
  hierarchical_reduce_scatter: hierarchical_reduce_scatter().
---
# Module: [`tpu_inference/kernels/collectives/hierarchical_reduce_scatter.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/hierarchical_reduce_scatter.py)

## Functions
- `_accumulate(recv_hbm, run_hbm, out_hbm, vmem_recv, vmem_run, sync_sems, vmem_idx=0)` — [`L29`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/hierarchical_reduce_scatter.py#L29)
- `_get_hypercube_chunk_idx(loop_idx, future_dims, prev_dims, my_chip_id, target_dim, dim_val)` — [`L71`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/hierarchical_reduce_scatter.py#L71)
- `_next_multiple_of(val, multiple)` — [`L24`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/hierarchical_reduce_scatter.py#L24)
- `hier_rs_kernel(input_ref, output_ref, running_sum_ref, recv_buf_ref, vmem_recv_ref, vmem_run_ref, final_copy_sem, phase1_sync_sems, load_recv_sem, load_run_sem, store_run_sem, *phase2_sync_sems_args, num_chips: int, num_hypercube_dims: int, num_micro_batches: int, hidden_size_dim: int, final_chunk_size: int, full_chunk_size: int, mb_size: int, axis_name: str='x')` — [`L85`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/hierarchical_reduce_scatter.py#L85)
- `hierarchical_reduce_scatter(x: jax.Array, *, mesh: jax.sharding.Mesh, in_specs: jax.sharding.PartitionSpec=jax.sharding.PartitionSpec('x', None), num_micro_batches: int=2)` — [`L428`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/hierarchical_reduce_scatter.py#L428)
- `hierarchical_reduce_scatter_local(local_x: jax.Array, num_devices: int, num_micro_batches: int=2, axis_name: str | tuple[str, ...]='x')` — [`L344`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/hierarchical_reduce_scatter.py#L344)

