---
title: 'Module: ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.all_gather_matmul._pallas_impl`/
symbols:
  all_gather_matmul: all_gather_matmul().
  _all_gather_kernel._start_or_wait_copy: _all_gather_kernel()._start_or_wait_copy().
  _all_gather_kernel._start_first_remote_copy: _all_gather_kernel()._start_first_remote_copy().
  _all_gather_kernel._start_subsequent_remote_copy: _all_gather_kernel()._start_subsequent_remote_copy().
  _all_gather_kernel._start_subsequent_x_local_copy: _all_gather_kernel()._start_subsequent_x_local_copy().
  _all_gather_kernel._start_o_local_copy: _all_gather_kernel()._start_o_local_copy().
  _all_gather_kernel._wait_o_local_copy: _all_gather_kernel()._wait_o_local_copy().
  _all_gather_kernel._wait_subsequent_x_local_copy: _all_gather_kernel()._wait_subsequent_x_local_copy().
  _all_gather_kernel._wait_first_remote_copy: _all_gather_kernel()._wait_first_remote_copy().
  _all_gather_kernel._wait_subsequent_remote_copy: _all_gather_kernel()._wait_subsequent_remote_copy().
  _all_gather_kernel._do_first_x_local_copy: _all_gather_kernel()._do_first_x_local_copy().
  _all_gather_kernel._do_subsequent_x_left_local_copy: _all_gather_kernel()._do_subsequent_x_left_local_copy().
  _all_gather_kernel._do_subsequent_x_right_local_copy: _all_gather_kernel()._do_subsequent_x_right_local_copy().
  _all_gather_kernel._do_y_local_copy: _all_gather_kernel()._do_y_local_copy().
  _all_gather_kernel._do_first_left_remote_copy: _all_gather_kernel()._do_first_left_remote_copy().
  _all_gather_kernel._do_first_right_remote_copy: _all_gather_kernel()._do_first_right_remote_copy().
  _all_gather_kernel._do_subsequent_left_remote_copy: _all_gather_kernel()._do_subsequent_left_remote_copy().
  _all_gather_kernel._do_subsequent_right_remote_copy: _all_gather_kernel()._do_subsequent_right_remote_copy().
  _all_gather_kernel._do_o_local_copy: _all_gather_kernel()._do_o_local_copy().
  _resolve_tp_size: _resolve_tp_size().
  all_gather_matmul._all_gather_matmul_call: all_gather_matmul()._all_gather_matmul_call().
  _all_gather_kernel._start_first_x_local_copy: _all_gather_kernel()._start_first_x_local_copy().
  _all_gather_kernel._start_y_local_copy: _all_gather_kernel()._start_y_local_copy().
  _all_gather_kernel._mxu: _all_gather_kernel()._mxu().
  _all_gather_kernel._wait_y_local_copy: _all_gather_kernel()._wait_y_local_copy().
  _all_gather_kernel._wait_first_x_local_copy: _all_gather_kernel()._wait_first_x_local_copy().
  _infer_axis_size: _infer_axis_size().
  _local_barrier: _local_barrier().
  _all_gather_kernel: _all_gather_kernel().
  _all_gather_kernel._do_mxu: _all_gather_kernel()._do_mxu().
  _all_gather_kernel._get_start_o_local_copy_cond: _all_gather_kernel()._get_start_o_local_copy_cond().
  _all_gather_kernel._get_wait_o_local_copy_cond: _all_gather_kernel()._get_wait_o_local_copy_cond().
  get_vmem_estimate_bytes: get_vmem_estimate_bytes().
  validate_inputs: validate_inputs().
  _all_gather_kernel._do_mxu._update: _all_gather_kernel()._do_mxu()._update().
---
# Module: [`ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py)

## Functions
- `_all_gather_kernel(x_hbm_ref, y_hbm_ref, o_hbm_ref, x_hbm_scratch_ref, x_local_copy_sem, y_local_copy_sem, o_local_copy_sem, send_sems, recv_sems, x_vmem_scratch_ref, y_vmem_scratch_ref, o_vmem_scratch_ref, acc_vmem_scratch_ref, axis_name: str, bn: int, bk: int, rhs_transpose: bool = False)` — [`L107`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L107) — Pallas kernel for all-gather.
- `_all_gather_matmul_call(x, y, bn, bk, rhs_transpose)` — [`L706`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L706)
- `_do_first_left_remote_copy(wait: bool = False)` — [`L224`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L224)
- `_do_first_right_remote_copy(wait: bool = False)` — [`L235`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L235)
- `_do_first_x_local_copy(wait: bool = False)` — [`L173`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L173)
- `_do_mxu()` — [`L268`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L268)
- `_do_o_local_copy(wait: bool = False)` — [`L314`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L314)
- `_do_subsequent_left_remote_copy(wait: bool = False)` — [`L246`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L246)
- `_do_subsequent_right_remote_copy(wait: bool = False)` — [`L257`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L257)
- `_do_subsequent_x_left_local_copy(wait: bool = False)` — [`L182`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L182)
- `_do_subsequent_x_right_local_copy(wait: bool = False)` — [`L191`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L191)
- `_do_y_local_copy(wait: bool = False)` — [`L207`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L207)
- `_get_start_o_local_copy_cond()` — [`L387`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L387)
- `_get_wait_o_local_copy_cond()` — [`L409`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L409)
- `_infer_axis_size(axis_name: str)` — [`L62`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L62) — Infer collective axis size from the active mapped context when available.
- `_local_barrier(left_neighbor, right_neighbor, double_barrier: bool = True)` — [`L82`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L82) — Barrier with neighbors using TPU semaphores.
- `_mxu()` — [`L406`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L406)
- `_resolve_tp_size(tp_size: int | None, axis_name: str)` — [`L74`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L74) — Resolve tensor-parallel world size using explicit value, axis context, then global device count.
- `_start_first_remote_copy()` — [`L351`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L351)
- `_start_first_x_local_copy()` — [`L371`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L371)
- `_start_o_local_copy()` — [`L401`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L401)
- `_start_or_wait_copy(op: jax._src.pallas.mosaic.primitives.AsyncCopyDescriptor, wait: bool = False)` — [`L164`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L164)
- `_start_subsequent_remote_copy()` — [`L365`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L365)
- `_start_subsequent_x_local_copy()` — [`L378`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L378)
- `_start_y_local_copy()` — [`L384`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L384)
- `_update()` — [`L309`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L309)
- `_wait_first_remote_copy()` — [`L444`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L444)
- `_wait_first_x_local_copy()` — [`L433`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L433)
- `_wait_o_local_copy()` — [`L423`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L423)
- `_wait_subsequent_remote_copy()` — [`L455`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L455)
- `_wait_subsequent_x_local_copy()` — [`L438`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L438)
- `_wait_y_local_copy()` — [`L428`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L428)
- `all_gather_matmul(x: jax.Array, y: jax.Array, axis_name: str, tp_size: int | None = None, collective_id: int | None = 0, bn: int | None = None, bk: int | None = None, rhs_transpose: bool = False)` — [`L571`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L571) — Low-level Pallas kernel launcher: all-gather ``x`` then compute ``x_full @ y``.
- `get_vmem_estimate_bytes(m, n, k, bn, acc_bytes, tp_size, x_dtype, y_dtype, out_dtype)` — [`L460`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L460) — Estimate total VMEM bytes consumed by the all-gather matmul kernel.
- `validate_inputs(x, y, tp_size, rhs_transpose=False)` — [`L519`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_pallas_impl.py#L519) — Validate inputs to the all-gather matmul kernel and raise on constraint violations.

