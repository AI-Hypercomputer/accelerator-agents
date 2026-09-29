---
title: 'Module: aqt/jax/v2/tiled_dot_general.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/tiled_dot_general.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.tiled_dot_general`/
symbols:
  AxisTiling: AxisTiling#
  TilingState: TilingState#
  TilingState.apply: TilingState#apply().
  TilingState.to_tiled_axes_transposed: TilingState#to_tiled_axes_transposed().
  AxisIdx: AxisIdx.
  AxisSize: AxisSize.
  AqtTileMap: AqtTileMap.
  EinsumEqnLetter: EinsumEqnLetter.
  EinsumTileSizes: EinsumTileSizes.
  BROADCAST_PREFIX: BROADCAST_PREFIX.
  AxisTiling.axis: AxisTiling#axis.
  AxisTiling.tile_count: AxisTiling#tile_count.
  AxisTiling.tile_size: AxisTiling#tile_size.
  AxisTiling.__post_init__: AxisTiling#__post_init__().
  AxisTiling.complete_missing: AxisTiling#complete_missing().
  TensorTiling: TensorTiling#
  TensorTiling.contraction_axes: TensorTiling#contraction_axes.
  TensorTiling.remaining_axes: TensorTiling#remaining_axes.
  Cfg: Cfg#
  Cfg.lhs: Cfg#lhs.
  Cfg.rhs: Cfg#rhs.
  Cfg.from_einsum: Cfg#from_einsum().
  Cfg.complete_missing: Cfg#complete_missing().
  interleave: interleave().
  get_ra: get_ra().
  maybe_add_one: maybe_add_one().
  TilingState.untiled_shape: TilingState#untiled_shape.
  TilingState.tile_map: TilingState#tile_map.
  TilingState.tiled_shape: TilingState#tiled_shape.
  TilingState.__post_init__: TilingState#__post_init__().
  TilingState.tile_one_axis: TilingState#tile_one_axis().
  TilingState.tile_axes: TilingState#tile_axes().
  TilingState._is_broadcasted_ax: TilingState#_is_broadcasted_ax().
  TilingState.get_broadcasted_tile_map_indexes: TilingState#get_broadcasted_tile_map_indexes().
  TilingState.unapply: TilingState#unapply().
  TilingState.broadcast_to_other: TilingState#broadcast_to_other().
  TilingState.axes_shape: TilingState#axes_shape().
  print_dimension_numbers: print_dimension_numbers().
  generate_tiling_state: generate_tiling_state().
  generate_tiling_states_for_dot_general: generate_tiling_states_for_dot_general().
  tiled_dot_general_with_tiling_states: tiled_dot_general_with_tiling_states().
  tiled_dot_general: tiled_dot_general().
---
# Module: [`aqt/jax/v2/tiled_dot_general.py`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py)

## Classes
### `AxisTiling`
- def: [`aqt/jax/v2/tiled_dot_general.py:47`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L47)
- doc: Axis tiling configuration for subchannel quantization.
- signature: `class AxisTiling:`
- members:
  - `complete_missing(self, shape)` — [`L58`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L58) — Completes missing tile_count or tile_size.
  - `axis` — [`L49`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L49)
  - `tile_count` — [`L51`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L51)
  - `tile_size` — [`L52`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L52)
- protocol/private: `__post_init__`[`L54`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L54)
- used by: [`AxisTiling`](aqt_quantizer.md#AxisTiling)

### `Cfg`
- def: [`aqt/jax/v2/tiled_dot_general.py:85`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L85)
- doc: Sequence of (lhs, rhs) configurations.
- signature: `class Cfg:`
- members:
  - `complete_missing(self, lhs_shape: tuple[AxisSize, ...], rhs_shape: tuple[AxisSize, ...])` — [`L135`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L135) — Makes lhs and rhs to cover all the axes.
  - `from_einsum(cls, eqn: str, einsum_tile_sizes: EinsumTileSizes)` — [`L91`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L91) — Creates Cfg based on einsum equation and tile sizes.
  - `lhs` — [`L87`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L87)
  - `rhs` — [`L88`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L88)

### `TensorTiling`
- def: [`aqt/jax/v2/tiled_dot_general.py:78`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L78)
- signature: `class TensorTiling:`
- members:
  - `contraction_axes` — [`L79`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L79)
  - `remaining_axes` — [`L80`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L80)

### `TilingState`
- def: [`aqt/jax/v2/tiled_dot_general.py:182`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L182) — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
- doc: Structure for bookkeeping of AxisIdx while tiling.
- signature: `class TilingState:`
- members:
  - `apply(self, x: jnp.ndarray)` — [`L238`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L238) — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
  - `axes_shape(self, axes: list[AxisIdx])` — [`L265`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L265)
  - `broadcast_to_other(self, bcast_shape: tuple[AxisSize, ...])` — [`L252`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L252) — Adds new axes (bcast_shape) on AxisIdx=0.
  - `get_broadcasted_tile_map_indexes(self)` — [`L234`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L234) — Returns the list of keys in `self.tile_map` associated with broadcasting axes.
  - `tile_axes(self, ats: Iterable[AxisTiling])` — [`L225`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L225)
  - `tile_one_axis(self, at: AxisTiling)` — [`L204`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L204) — Tiles (splits) one axis while maintaining all AxisIdx.
  - `to_tiled_axes_transposed(self, axes: Iterable[AxisIdx | str])` — [`L268`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L268) — The given 'axes' parameter defines axes in untiled represented Array. — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
  - `unapply(self, tiled_x: jnp.ndarray)` — [`L244`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L244)
  - `tile_map` — [`L194`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L194)
  - `tiled_shape` — [`L196`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L196)
  - `untiled_shape` — [`L185`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L185)
- protocol/private: `__post_init__`[`L198`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L198), `_is_broadcasted_ax`[`L229`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L229)
- used by: [`calibrate`](aqt_quantizer.md#Quantizer.calibrate), [`TilingState`](aqt_quantizer.md#TilingState)

## Functions
- `generate_tiling_state(tensor: jnp.ndarray, tiled_axes: Iterable[AxisTiling])` — [`L321`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L321) — Generates tiling states for the given tensor.
- `generate_tiling_states_for_dot_general(cfg: Cfg, lhs, rhs, dimension_numbers: jax.lax.DotDimensionNumbers)` — [`L334`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L334) — Do tiling for `lhs` and `rhs` and returns the intermediate tiling states.
- `get_ra(rank, ca, ba)` — [`L173`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L173)
- `interleave(tile_count, tile_size, tile_map, remaining_axes, product=False)` — [`L150`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L150) — Interleave the tile_count and tile_size of remaining axes.
- `maybe_add_one(i, min_i)` — [`L177`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L177)
- `print_dimension_numbers(dimension_numbers, lhs, rhs, label)` — [`L297`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L297) — Prints dimension numbers before and/or after tiling.
- `tiled_dot_general(cfg: Cfg, lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None, dot_general=jax.lax.dot_general)` — [`L546`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L546) — local dot_general.
- `tiled_dot_general_with_tiling_states(lhs: jnp.ndarray, xlhs: TilingState, rhs: jnp.ndarray, xrhs: TilingState, untiled_dimension_numbers: jax.lax.DotDimensionNumbers, precision=None, preferred_element_type=None, dot_general=jax.lax.dot_general)` — [`L401`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L401) — local dot_general with tiling states.

## Module values
- `AqtTileMap` — [`L39`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L39)
- `AxisIdx` — [`L37`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L37)
- `AxisSize` — [`L38`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L38)
- `BROADCAST_PREFIX` — [`L43`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L43)
- `EinsumEqnLetter` — [`L40`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L40)
- `EinsumTileSizes` — [`L41`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general.py#L41)

