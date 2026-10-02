---
title: 'Module: tpu_inference/kernels/collectives/all_gather_matmul.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/collectives/all_gather_matmul.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.collectives.all_gather_matmul`/
symbols:
  P: P.
  _cdiv: _cdiv().
  _all_gather_kernel: _all_gather_kernel().
  get_vmem_estimate_bytes: get_vmem_estimate_bytes().
  validate_inputs: validate_inputs().
  all_gather_matmul: all_gather_matmul().
  get_kernel_name: get_kernel_name().
---
# Module: [`tpu_inference/kernels/collectives/all_gather_matmul.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/all_gather_matmul.py)

## Functions
- `_all_gather_kernel(x_hbm_ref, y_hbm_ref, o_hbm_ref, x_hbm_scratch_ref, x_local_copy_sem, y_local_copy_sem, o_local_copy_sem, send_sems, recv_sems, x_vmem_scratch_ref, y_vmem_scratch_ref, o_vmem_scratch_ref, acc_vmem_scratch_ref, axis_name: str, bn: int, bk: int, debug_mode=False, rhs_transpose: bool=False)` — [`L32`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/all_gather_matmul.py#L32) — Pallas kernel for all-gather.
- `_cdiv(x, y)` — [`L19`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/all_gather_matmul.py#L19)
- `all_gather_matmul(x: jax.Array, y: jax.Array, mesh: jax.sharding.AbstractMesh, axis_name: str, collective_id: int | None=0, bn: int | None=None, bk: int | None=None, rhs_transpose: bool=False)` — [`L593`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/all_gather_matmul.py#L593) — Performs all-gather on the input tensor and then a matmul.
- `get_kernel_name(bn: int, bk: int, rhs_transpose: bool)` — [`L733`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/all_gather_matmul.py#L733)
- `get_vmem_estimate_bytes(m, n, k, bn, acc_bytes, tp_size, x_dtype, y_dtype, out_dtype)` — [`L529`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/all_gather_matmul.py#L529) — Returns the total vmem bytes used by the kernel.
- `validate_inputs(x, y, tp_size, rhs_transpose=False)` — [`L555`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/all_gather_matmul.py#L555) — Validates the inputs to the all_gather_matmul kernel.

## Module values
- `P` — [`L16`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/all_gather_matmul.py#L16)

