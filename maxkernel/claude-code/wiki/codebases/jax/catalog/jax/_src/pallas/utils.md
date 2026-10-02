---
title: 'Module: jax/_src/pallas/utils.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/utils.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.utils`/
symbols:
  cdiv: cdiv().
  pattern_match_scan_to_fori_loop: pattern_match_scan_to_fori_loop().
  nextafter_lowering_helper: nextafter_lowering_helper().
  pattern_match_while_to_fori_loop: pattern_match_while_to_fori_loop().
  _erf_inv_64_lowering_helper: _erf_inv_64_lowering_helper().
  next_power_of_2: next_power_of_2().
  erf_inv_lowering_helper: erf_inv_lowering_helper().
  _erf_inv_32_lowering_helper: _erf_inv_32_lowering_helper().
  MeshInfo: MeshInfo#
  MeshInfo.axis_names: MeshInfo#axis_names.
  sign_lowering_helper: sign_lowering_helper().
  MeshInfo.from_mesh: MeshInfo#from_mesh().
  _erf_inv_64_lowering_helper.get_coefficient: _erf_inv_64_lowering_helper().get_coefficient().
  align_to: align_to().
  strides_from_shape: strides_from_shape().
  MeshInfo.mesh_shape: MeshInfo#mesh_shape.
  MeshInfo.mesh_strides: MeshInfo#mesh_strides.
---
# Module: [`jax/_src/pallas/utils.py`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py)

## Classes
### `MeshInfo`
- def: [`jax/_src/pallas/utils.py:455`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L455)
- signature: `class MeshInfo:`
- members:
  - `from_mesh(mesh: Any)` — [`L461`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L461)
  - `axis_names` — [`L457`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L457)
  - `mesh_shape` — [`L456`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L456)
  - `mesh_strides` — [`L458`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L458)
- uses (calls/refs, reference-scoped): [`strides_from_shape`](utils.md#strides_from_shape)
- used by: [`body`](mosaic_gpu/lowering.md#lower_jaxpr_to_module.body), [`_semaphore_signal_multicast_lowering`](mosaic_gpu/primitives.md#_semaphore_signal_multicast_lowering), [`_axis_index_rule`](mosaic_gpu/lowering.md#_axis_index_rule), [`_multimem_store_lowering_rule`](mosaic_gpu/primitives.md#_multimem_store_lowering_rule), [`_multimem_load_reduce_lowering_rule`](mosaic_gpu/primitives.md#_multimem_load_reduce_lowering_rule), [`_multimem_load_reduce_lowering_rule_wg`](mosaic_gpu/primitives.md#_multimem_load_reduce_lowering_rule_wg), [`_axis_index_rule`](mosaic/lowering.md#_axis_index_rule), [`from_mesh`](mosaic/lowering.md#UnpipelinedLoweringContext.from_mesh), [`device_id_to_logical`](primitives.md#device_id_to_logical), [`_prepare_mesh_info`](mosaic/lowering.md#MosaicGridMapping._prepare_mesh_info), [`_device_id_dict_to_mesh`](primitives.md#_device_id_dict_to_mesh), [`mesh_info`](mosaic_gpu/lowering.md#ModuleContext.mesh_info), [`mesh_info`](mosaic/lowering.md#MosaicGridMapping.mesh_info), [`jax_mesh_context`](mosaic/lowering.md#LoweringContext.jax_mesh_context)

## Functions
- `_erf_inv_32_lowering_helper(x)` — [`L249`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L249)
- `_erf_inv_64_lowering_helper(x)` — [`L276`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L276)
- `align_to(a: int, alignment: int)` — [`L62`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L62)
- `cdiv(a: int, b: int)` — [`L32`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L32)
- `erf_inv_lowering_helper(x)` — [`L344`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L344)
- `get_coefficient(i)` — [`L321`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L321)
- `next_power_of_2(x: int)` — [`L104`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L104) — Returns the next power of two greater than or equal to `x`.
- `nextafter_lowering_helper(x, y)` — [`L367`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L367)
- `pattern_match_scan_to_fori_loop(jaxpr: jax_core.Jaxpr, num_consts: int, num_carry: int)` — [`L111`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L111)
- `pattern_match_while_to_fori_loop(cond_jaxpr: jax_core.ClosedJaxpr, cond_nconsts: int, body_jaxpr: jax_core.ClosedJaxpr, body_nconsts: int)` — [`L168`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L168)
- `sign_lowering_helper(x)` — [`L352`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L352)
- `strides_from_shape(shape: tuple[int, ...])` — [`L95`](../../../../../../../raw/code/jax/jax/_src/pallas/utils.py#L95)

