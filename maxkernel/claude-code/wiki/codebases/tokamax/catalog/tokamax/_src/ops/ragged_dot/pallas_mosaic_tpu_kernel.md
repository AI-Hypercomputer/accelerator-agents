---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_tpu_kernel`/
symbols:
  gmm: gmm().
  tgmm: tgmm().
  QArray: QArray.
  tgmm.kernel._do: tgmm().kernel()._do().
  gmm.kernel.accum: gmm().kernel().accum().
  tgmm.kernel._case4: tgmm().kernel()._case4().
  _quantize_as: _quantize_as().
  _get_store_mask: _get_store_mask().
  tgmm.kernel._case2: tgmm().kernel()._case2().
  tgmm.kernel._case3: tgmm().kernel()._case3().
  _validate_args: _validate_args().
  make_group_metadata: make_group_metadata().
  gmm._maybe_get_subslice: gmm()._maybe_get_subslice().
  gmm.kernel: gmm().kernel().
  tgmm.kernel._zero_acc: tgmm().kernel()._zero_acc().
  tgmm.kernel._store_accum: tgmm().kernel()._store_accum().
  tgmm.kernel._case1: tgmm().kernel()._case1().
  tgmm.kernel._stage1: tgmm().kernel()._stage1().
  tgmm.kernel._stage2: tgmm().kernel()._stage2().
  tgmm.kernel._stage3: tgmm().kernel()._stage3().
  CanonicalPrecision: CanonicalPrecision.
  GroupMetadata: GroupMetadata.
  _scale_out_by_scale: _scale_out_by_scale().
  _TilingFn: _TilingFn.
  gmm.rhs_index_map: gmm().rhs_index_map().
  gmm.dot: gmm().dot().
  gmm._maybe_get_subslice.getter: gmm()._maybe_get_subslice().getter().
  gmm.lhs_index_map: gmm().lhs_index_map().
  gmm.out_index_map: gmm().out_index_map().
  tgmm.dot: tgmm().dot().
  tgmm.kernel: tgmm().kernel().
  tgmm.lhs_index_map: tgmm().lhs_index_map().
  tgmm.rhs_index_map: tgmm().rhs_index_map().
  tgmm.out_index_map: tgmm().out_index_map().
  _KERNEL_NAME_GMM._KERNEL_NAME_GMM: _KERNEL_NAME_GMM._KERNEL_NAME_GMM.
  _KERNEL_NAME_TGMM._KERNEL_NAME_TGMM: _KERNEL_NAME_TGMM._KERNEL_NAME_TGMM.
  gmm.kernel._zero_acc: gmm().kernel()._zero_acc().
  tgmm.kernel._case5: tgmm().kernel()._case5().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py)

## Functions
- `_case1()` — [`L819`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L819)
- `_case2()` — [`L824`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L824)
- `_case3()` — [`L830`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L830)
- `_case4()` — [`L836`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L836)
- `_case5()` — [`L843`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L843)
- `_do()` — [`L781`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L781)
- `_get_store_mask(*, grid_id: jax.Array, group_metadata: GroupMetadata, tm: int, tn: int)` — [`L255`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L255) — Mask for rows that belong to the current group in the current tile.
- `_maybe_get_subslice(x, idx: int, subslice_count: int | None, axis: int)` — [`L434`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L434) — Returns a subslice of the given array along the given axis.
- `_quantize_as(x, qdtype: jnp.dtype, axis: int, scale: float | None)` — [`L272`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L272)
- `_scale_out_by_scale(out: jax.Array, scales: jax.Array, axis: int | None = None)` — [`L285`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L285)
- `_stage1()` — [`L853`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L853)
- `_stage2()` — [`L858`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L858)
- `_stage3()` — [`L863`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L863)
- `_store_accum()` — [`L775`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L775)
- `_validate_args(lhs: jax.Array | QArray, rhs: jax.Array | QArray, group_sizes: jax.Array, *, expected_rhs_dims: int = 3)` — [`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L47) — Validates the arguments for the gmm function.
- `_zero_acc()` — [`L460`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L460)
- `_zero_acc()` — [`L772`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L772)
- `accum(is_last_k_tile)` — [`L463`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L463)
- `dot(x, y, preferred_element_type, *, precision=precision)` — [`L417`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L417)
- `dot(x, y, preferred_element_type, *, precision=precision)` — [`L732`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L732)
- `getter(x)` — [`L440`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L440)
- `gmm(lhs: jax.Array | QArray, rhs: jax.Array | QArray, group_sizes: jax.Array, precision: CanonicalPrecision, out_dtype: jnp.dtype, tiling: tuple[int, int, int] | _TilingFn | None = (128, 128, 128), input_buffer_count: int = 2, group_offset: jax.Array | None = None, transpose_rhs: bool = False, interpret: bool = False, manual_axis_type: ManualAxisType | None = None, lhs_qdtype: jnp.dtype | None = None, lhs_static_scale: float | None = None, rhs_qdtype: jnp.dtype | None = None, rhs_static_scale: float | None = None, activation: base.ActivationFunction | None = None)` — [`L335`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L335) — Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.
- `kernel(group_metadata, _, lhs_ref, rhs_ref, out_ref, acc_scratch, *, subchannel_iters: int)` — [`L447`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L447)
- `kernel(group_metadata, _, lhs_ref, rhs_ref, out_ref, acc_scratch, *, subchannel_iters: int)` — [`L747`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L747)
- `lhs_index_map(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L526`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L526)
- `lhs_index_map(n_i, k_i, grid_id, group_metadata, group_offset)` — [`L867`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L867)
- `make_group_metadata(*, group_sizes: jax.Array, m: int, tm: int, start_group: jax.Array, num_nonzero_groups: int, visit_empty_groups: bool)` — [`L74`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L74) — Create the metadata needed for grouped matmul computation.
- `out_index_map(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L544`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L544)
- `out_index_map(n_i, k_i, grid_id, group_metadata, group_offset)` — [`L879`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L879)
- `rhs_index_map(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L532`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L532)
- `rhs_index_map(n_i, k_i, grid_id, group_metadata, group_offset)` — [`L873`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L873)
- `tgmm(lhs: jax.Array | QArray, rhs: jax.Array | QArray, group_sizes: jax.Array, precision: CanonicalPrecision, out_dtype: jnp.dtype, tiling: tuple[int, int, int] | _TilingFn | None = (128, 128, 128), input_buffer_count: int = 2, group_offset: jax.Array | None = None, num_actual_groups: int | None = None, interpret: bool = False, manual_axis_type: ManualAxisType | None = None, lhs_qdtype: jnp.dtype | None = None, lhs_static_scale: float | None = None, rhs_qdtype: jnp.dtype | None = None, rhs_static_scale: float | None = None, activation: base.ActivationFunction | None = None, combine_scopes: bool = False)` — [`L644`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L644) — Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :].

## Module values
- `CanonicalPrecision` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L40)
- `GroupMetadata` — [`L71`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L71)
- `QArray` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L41)
- `_KERNEL_NAME_GMM` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L43)
- `_KERNEL_NAME_TGMM` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L44)
- `_TilingFn` — [`L315`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py#L315)

