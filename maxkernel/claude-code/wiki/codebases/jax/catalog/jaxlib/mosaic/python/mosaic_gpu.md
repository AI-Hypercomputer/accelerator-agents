---
title: 'Module: jaxlib/mosaic/python/mosaic_gpu.py'
type: catalog
provenance: extracted
module: jaxlib/mosaic/python/mosaic_gpu.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jaxlib.mosaic.python.mosaic_gpu`/
symbols:
  WarpMapOp: WarpMapOp#
  WarpMapOp.body: WarpMapOp#body().
  warp_map: warp_map().
  WarpMapOp.__init__: WarpMapOp#__init__().
---
# Module: [`jaxlib/mosaic/python/mosaic_gpu.py`](../../../../../../../raw/code/jax/jaxlib/mosaic/python/mosaic_gpu.py)

## Classes
### `WarpMapOp`
- def: [`jaxlib/mosaic/python/mosaic_gpu.py:39`](../../../../../../../raw/code/jax/jaxlib/mosaic/python/mosaic_gpu.py#L39)
- doc: An extension to the automatically generated WarpMapOp bindings.
- signature: `class WarpMapOp(_mosaic_gpu_gen_ops.WarpMapOp):`
- members:
  - `body(self)` — [`L48`](../../../../../../../raw/code/jax/jaxlib/mosaic/python/mosaic_gpu.py#L48)
- protocol/private: `__init__`[`L42`](../../../../../../../raw/code/jax/jaxlib/mosaic/python/mosaic_gpu.py#L42)
- used by: [`_core_map_lowering_rule`](../../../jax/_src/pallas/mosaic_gpu/lowering.md#_core_map_lowering_rule), [`_mpmd_map_lowering_rule`](../../../jax/_src/pallas/mosaic_gpu/lowering.md#_mpmd_map_lowering_rule), [`_warp_map_constraint_system`](../../../jax/experimental/mosaic/gpu/layout_inference.md#_warp_map_constraint_system), [`is_known_divisible`](../../../jax/experimental/mosaic/gpu/utils.md#is_known_divisible), [`_mgpu_warp_map_op_lowering_rule`](../../../jax/experimental/mosaic/gpu/dialect_lowering.md#_mgpu_warp_map_op_lowering_rule), [`warp_map`](mosaic_gpu.md#warp_map)

## Functions
- `warp_map(operands, *, loc=None, ip=None)` — [`L52`](../../../../../../../raw/code/jax/jaxlib/mosaic/python/mosaic_gpu.py#L52)

