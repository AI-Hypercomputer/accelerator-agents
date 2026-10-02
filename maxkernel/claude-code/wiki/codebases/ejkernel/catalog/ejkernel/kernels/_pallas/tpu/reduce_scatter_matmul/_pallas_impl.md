---
title: 'Module: ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.reduce_scatter_matmul._pallas_impl`/
symbols:
  Ref: Ref.
  _kernel: _kernel().
  reduce_scatter_matmul: reduce_scatter_matmul().
  _kernel._prologue: _kernel()._prologue().
  _kernel._main_loop: _kernel()._main_loop().
  _kernel.accumulate_computation_to_slot: _kernel().accumulate_computation_to_slot().
  tiled_matmul_hbm: tiled_matmul_hbm().
  _kernel.compute_matmul_top_half: _kernel().compute_matmul_top_half().
  _kernel.compute_matmul_bot_half: _kernel().compute_matmul_bot_half().
  mod: mod().
  _resolve_tp_size: _resolve_tp_size().
  tiled_add_hbm: tiled_add_hbm().
  _kernel.get_left_target_block: _kernel().get_left_target_block().
  _kernel.get_right_target_block: _kernel().get_right_target_block().
  reduce_scatter_matmul.kernel_fn: reduce_scatter_matmul().kernel_fn().
  ALGORITHM_DIAGRAM: ALGORITHM_DIAGRAM.
  _kernel._main_loop._epilogue: _kernel()._main_loop()._epilogue().
  _kernel._main_loop._accumulate: _kernel()._main_loop()._accumulate().
  KernelConfig: KernelConfig#
  KernelConfig.num_devices: KernelConfig#num_devices.
  KernelConfig.m_block: KernelConfig#m_block.
  KernelConfig.m_half_block: KernelConfig#m_half_block.
  KernelConfig.bm: KernelConfig#bm.
  KernelConfig.bn: KernelConfig#bn.
  KernelConfig.bk: KernelConfig#bk.
  _kernel.copy_computation_to_slot: _kernel().copy_computation_to_slot().
  _infer_axis_size: _infer_axis_size().
  _kernel.local_barrier: _kernel().local_barrier().
  _kernel.signal_left_neighbor: _kernel().signal_left_neighbor().
  _kernel.signal_right_neighbor: _kernel().signal_right_neighbor().
---
# Module: [`ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py)

## Classes
### `KernelConfig`  ·  implements/extends NamedTuple
- def: [`ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py:122`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L122)
- doc: Configuration for the kernel.
- signature: `class KernelConfig(NamedTuple):`
- members:
  - `bk` — [`L130`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L130)
  - `bm` — [`L128`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L128)
  - `bn` — [`L129`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L129)
  - `m_block` — [`L126`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L126)
  - `m_half_block` — [`L127`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L127)
  - `num_devices` — [`L125`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L125)
- used by: [`_kernel`](_pallas_impl.md#_kernel), [`reduce_scatter_matmul`](_pallas_impl.md#reduce_scatter_matmul)

## Functions
- `_accumulate()` — [`L527`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L527)
- `_epilogue()` — [`L506`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L506)
- `_infer_axis_size(axis_name: str)` — [`L97`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L97) — Infer collective axis size from the active mapped context when available.
- `_kernel(x_ref: Ref, y_ref: Ref, out_ref: Ref, scratch_ref: Ref, computation_scratch_ref: Ref, x_vmem_ref: Ref, y_vmem_ref: Ref, acc_vmem_ref: Ref, out_vmem_ref: Ref, add_vmem_ref: Ref, send_left_sem: Ref, recv_left_sem: Ref, send_right_sem: Ref, recv_right_sem: Ref, copy_sem: Ref, left_capacity_sem: Ref, right_capacity_sem: Ref, *, config: KernelConfig, axis_name: str)` — [`L272`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L272) — Bidirectional Reduce-Scatter Matmul Kernel with M-split algorithm.
- `_main_loop()` — [`L472`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L472)
- `_prologue()` — [`L462`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L462)
- `_resolve_tp_size(tp_size: int | None, axis_name: str)` — [`L109`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L109) — Resolve tensor-parallel world size using explicit value, axis context, then global device count.
- `accumulate_computation_to_slot(compute_slot, dst_slot)` — [`L384`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L384) — Add computation_scratch_ref[compute_slot] to scratch_ref[dst_slot].
- `compute_matmul_bot_half(block_idx, out_slot)` — [`L361`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L361) — Compute matmul for BOT half of specified block.
- `compute_matmul_top_half(block_idx, out_slot)` — [`L338`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L338) — Compute matmul for TOP half of specified block.
- `copy_computation_to_slot(compute_slot, dst_slot)` — [`L396`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L396) — Copy computation_scratch_ref[compute_slot] to scratch_ref[dst_slot].
- `get_left_target_block(step)` — [`L330`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L330) — LEFT direction: compute TOP half of block (my_id + step + 1) % N.
- `get_right_target_block(step)` — [`L334`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L334) — RIGHT direction: compute BOT half of block (my_id - step - 1) % N.
- `kernel_fn(x_ref, y_ref, out_ref, scratch_ref, computation_scratch_ref, x_vmem_ref, y_vmem_ref, acc_vmem_ref, out_vmem_ref, add_vmem_ref, send_left_sem, recv_left_sem, send_right_sem, recv_right_sem, copy_sem, left_capacity_sem, right_capacity_sem)` — [`L627`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L627)
- `local_barrier()` — [`L406`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L406) — Barrier with both neighbors using double-barrier pattern.
- `mod(x: jax.Array, n: int)` — [`L117`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L117) — Modulo operation that works with JAX arrays.
- `reduce_scatter_matmul(x: jax.Array, y: jax.Array, *, axis_name: str = "x", tp_size: int | None = None, collective_id: int | None = 0, bm: int = 128, bn: int = 128, bk: int = 128)` — [`L532`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L532) — Bidirectional reduce-scatter matmul with M-split algorithm.
- `signal_left_neighbor()` — [`L440`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L440) — Signal left neighbor that we are ready to receive from them.
- `signal_right_neighbor()` — [`L449`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L449) — Signal right neighbor that we are ready to receive from them.
- `tiled_add_hbm(src_hbm_ref: Ref, dst_hbm_ref: Ref, src_vmem_ref: Ref, dst_vmem_ref: Ref, copy_sem: Ref, *, bm: int, bn: int)` — [`L220`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L220) — Tiled addition: dst += src, with HBM inputs.
- `tiled_matmul_hbm(x_hbm_ref: Ref, y_hbm_ref: Ref, out_hbm_ref: Ref, x_vmem_ref: Ref, y_vmem_ref: Ref, acc_vmem_ref: Ref, out_vmem_ref: Ref, copy_sem: Ref, *, m_block_idx: int | jax.Array, m_size: int, bm: int, bn: int, bk: int)` — [`L133`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L133) — Tiled matmul: out = x[m_block_idx*bm:m_block_idx*bm+m_size, :] @ y.T.

## Module values
- `ALGORITHM_DIAGRAM` — [`L704`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L704)
- `Ref` — [`L94`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_pallas_impl.py#L94)

