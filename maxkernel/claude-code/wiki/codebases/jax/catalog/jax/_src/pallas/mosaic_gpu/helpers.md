---
title: 'Module: jax/_src/pallas/mosaic_gpu/helpers.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic_gpu/helpers.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic_gpu.helpers`/
symbols:
  dynamic_scheduling_loop.decorator._scoped.try_cancel_body: dynamic_scheduling_loop().decorator()._scoped().try_cancel_body().
  dynamic_scheduling_loop.decorator: dynamic_scheduling_loop().decorator().
  nd_loop.decorator.wrapper: nd_loop().decorator().wrapper().
  planar_snake: planar_snake().
  nd_loop: nd_loop().
  warp_map: warp_map().
  nd_loop.decorator: nd_loop().decorator().
  dynamic_scheduling_loop.decorator._scoped: dynamic_scheduling_loop().decorator()._scoped().
  NDLoopInfo: NDLoopInfo#
  NDLoopInfo.index: NDLoopInfo#index.
  planar_snake.tile_coordinates: planar_snake().tile_coordinates().
  dynamic_scheduling_loop: dynamic_scheduling_loop().
  find_swizzle: find_swizzle().
  dynamic_scheduling_loop.decorator._scoped.try_cancel_body.wait_until_slot_available: dynamic_scheduling_loop().decorator()._scoped().try_cancel_body().wait_until_slot_available().
  dynamic_scheduling_loop.decorator._scoped.reset_cancel_barrier: dynamic_scheduling_loop().decorator()._scoped().reset_cancel_barrier().
  NDLoopInfo.local_index: NDLoopInfo#local_index.
  format_tcgen05_sparse_metadata: format_tcgen05_sparse_metadata().
  NDLoopInfo.num_local_steps: NDLoopInfo#num_local_steps.
  inline_ptx: inline_ptx().
  inline_ptx.ptx: inline_ptx().ptx().
  dynamic_scheduling_loop.decorator._scoped.try_cancel_cond: dynamic_scheduling_loop().decorator()._scoped().try_cancel_cond().
  _FENCE_PROXY_ASYNC_GENERIC_ACQUIRE_SHARED_CLUSTER: _FENCE_PROXY_ASYNC_GENERIC_ACQUIRE_SHARED_CLUSTER.
  _FENCE_PROXY_ASYNC_GENERIC_RELEASE_SHARED_CTA: _FENCE_PROXY_ASYNC_GENERIC_RELEASE_SHARED_CTA.
---
# Module: [`jax/_src/pallas/mosaic_gpu/helpers.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py)

## Classes
### `NDLoopInfo`
- def: [`jax/_src/pallas/mosaic_gpu/helpers.py:36`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L36)
- doc: Container dataclass for loop iteration information.
- signature: `class NDLoopInfo:`
- members:
  - `index` — [`L45`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L45)
  - `local_index` — [`L46`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L46)
  - `num_local_steps` — [`L47`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L47)
- uses (calls/refs, reference-scoped): [`Array`](../../basearray.md#Array)
- used by: [`mn_loop`](../../../experimental/pallas/ops/gpu/blackwell_matmul_mgpu.md#matmul_kernel.kernel.mn_loop), [`_mn_loop`](../../../experimental/pallas/ops/gpu/hopper_matmul_mgpu.md#kernel._pipeline_scope._mn_loop), [`_mn_loop`](../../../experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.md#mixed_matmul_kernel.kernel._pipeline_scope._mn_loop), [`mn_loop`](../../../experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.md#ragged_dot_kernel.kernel._scoped.mn_loop), [`try_cancel_body`](helpers.md#dynamic_scheduling_loop.decorator._scoped.try_cancel_body), [`wrapper`](helpers.md#nd_loop.decorator.wrapper), [`_transfer_loop`](../../../experimental/pallas/ops/gpu/all_gather_mgpu.md#all_gather.kernel._transfer_loop), [`_transfer_loop`](../../../experimental/pallas/ops/gpu/reduce_scatter_mgpu.md#reduce_scatter.kernel._transfer_loop), [`mn_loop`](../../../experimental/pallas/ops/gpu/ragged_dot_mgpu.md#ragged_dot.body.mn_loop), [`mn_loop`](../../../experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.md#transposed_ragged_dot.body.mn_loop), [`nd_loop`](helpers.md#nd_loop), [`dynamic_scheduling_loop`](helpers.md#dynamic_scheduling_loop)

## Functions
- `_scoped(try_cancel_buffer, try_cancel_barrier, cancel_used_barrier)` — [`L370`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L370)
- `decorator(body)` — [`L144`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L144)
- `decorator(body)` — [`L369`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L369)
- `dynamic_scheduling_loop(grid_names: Sequence[Hashable], *, thread_axis: Hashable | None = None, cluster_axes: tuple[str | tuple[str, ...], ...] = (), init_carry: None = None)` — [`L314`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L314)
- `find_swizzle(minor_dim_bits: int, what: str = "")` — [`L226`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L226) — Returns the largest swizzle that can be applied to a memory region.
- `format_tcgen05_sparse_metadata(meta, operand_dtype)` — [`L186`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L186) — Formats the sparse metadata for tcgen05.mma into the expected format.
- `inline_ptx(asm: str)` — [`L449`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L449) — Inserts inline PTX assembly.
- `nd_loop(grid: Sequence[int | jax.Array], *, collective_axes: Sequence[Hashable] | Hashable, tiling: Sequence[int] | None = None, init_carry: None = None)` — [`L51`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L51)
- `planar_snake(lin_idx: jax.Array, shape: tuple[int | jax.Array, int | jax.Array], minor_dim: int, tile_width: int)` — [`L255`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L255) — Converts a linear index into an index into shape, trying to optimize locality.
- `ptx(_)` — [`L453`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L453)
- `reset_cancel_barrier(slot)` — [`L411`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L411)
- `tile_coordinates(lin_idx, width)` — [`L290`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L290)
- `try_cancel_body(carry)` — [`L375`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L375)
- `try_cancel_cond(carry)` — [`L371`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L371)
- `wait_until_slot_available()` — [`L383`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L383)
- `warp_map(f: Callable[[jax.Array], T],)` — [`L460`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L460) — Runs a function with single warp semantics, passing it the warp ID.
- `wrapper(wave_step, carry)` — [`L145`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L145)

## Module values
- `_FENCE_PROXY_ASYNC_GENERIC_ACQUIRE_SHARED_CLUSTER` — [`L441`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L441)
- `_FENCE_PROXY_ASYNC_GENERIC_RELEASE_SHARED_CTA` — [`L444`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/helpers.py#L444)

