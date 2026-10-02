---
title: 'Module: jax/_src/pallas/einshape.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/einshape.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.einshape`/
symbols:
  _tile_preserving_einshape_kernel: _tile_preserving_einshape_kernel().
  _einshape: _einshape().
  _einshape_kernel: _einshape_kernel().
  _is_tile_preserving: _is_tile_preserving().
  get_einshape_transforms: get_einshape_transforms().
  _einshape_lo_abstract_eval: _einshape_lo_abstract_eval().
  Einshape.__init__: Einshape#__init__().
  Factor: Factor#
  Einshape.expand: Einshape#expand().
  _consolidate: _consolidate().
  _einshape_lo_lowering_rule: _einshape_lo_lowering_rule().
  _apply_split: _apply_split().
  Transform: Transform.
  _tile_preserving_einshape_kernel.get_outer_shape: _tile_preserving_einshape_kernel().get_outer_shape().
  einshape_lo: einshape_lo().
  Einshape.assert_is_tile_preserving: Einshape#assert_is_tile_preserving.
  einshape: einshape().
  Factor.size: Factor#size.
  einshape_lo_p: einshape_lo_p.
  _default_einshape_kernel: _default_einshape_kernel().
  _init_dims: _init_dims().
  SplitDims.transform_shape: SplitDims#transform_shape().
  MergeDims.transform_shape: MergeDims#transform_shape().
  _array_to_2d_tile_array: _array_to_2d_tile_array().
  _2d_tile_array_to_array: _2d_tile_array_to_array().
  SplitDims: SplitDims#
  MergeDims: MergeDims#
  Transpose: Transpose#
  MergeDims.index: MergeDims#index.
  Factor.kind: Factor#kind.
  _parse_equation: _parse_equation().
  _einshape_lo_lowering: _einshape_lo_lowering().
  _einshape_lo_lowering.f: _einshape_lo_lowering().f().
  Einshape: Einshape#
  Transpose.transform_shape: Transpose#transform_shape().
  SplitDims.index: SplitDims#index.
  MergeDims.count: MergeDims#count.
  _parse_side: _parse_side().
  SplitDims.sizes: SplitDims#sizes.
  Transpose.permutation: Transpose#permutation.
  Einshape.equation: Einshape#equation.
  Einshape.sizes: Einshape#sizes.
---
# Module: [`jax/_src/pallas/einshape.py`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py)

## Classes
### `Einshape`  ·  implements/extends VJPHiPrimitive
- def: [`jax/_src/pallas/einshape.py:357`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L357)
- doc: Einshape primitive.
- signature: `class Einshape(hijax.VJPHiPrimitive):`
- members:
  - `expand(self, x: jax_typing.Array)` — [`L384`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L384)
  - `assert_is_tile_preserving` — [`L375`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L375)
  - `equation` — [`L373`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L373)
  - `sizes` — [`L374`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L374)
- protocol/private: `__init__`[`L360`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L360)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array), [`ShapedArray`](../core.md#ShapedArray), [`FrozenDict`](../frozen_dict.md#FrozenDict), [`_einshape`](einshape.md#_einshape), [`VJPHiPrimitive`](../hijax.md#VJPHiPrimitive), [`in_avals`](../hijax.md#VJPHiPrimitive.in_avals), [`__init__`](../hijax.md#VJPHiPrimitive.__init__), [`eval_shape`](../api.md#eval_shape), [`out_aval`](../hijax.md#VJPHiPrimitive.out_aval), [`params`](../hijax.md#VJPHiPrimitive.params), [`einshape_lo`](einshape.md#einshape_lo), [`ShapedArray`](../hijax.md#ShapedArray)
- used by: [`expand`](../hijax.md#VJPHiPrimitive.expand), [`VJPHiPrimitive`](../hijax.md#VJPHiPrimitive), [`einshape`](einshape.md#einshape)

### `Factor`  ·  implements/extends NamedTuple
- def: [`jax/_src/pallas/einshape.py:442`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L442)
- signature: `class Factor(NamedTuple):`
- members:
  - `kind` — [`L444`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L444)
  - `size` — [`L443`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L443)
- used by: [`_is_tile_preserving`](einshape.md#_is_tile_preserving), [`_consolidate`](einshape.md#_consolidate), [`_apply_split`](einshape.md#_apply_split), [`get_outer_shape`](einshape.md#_tile_preserving_einshape_kernel.get_outer_shape), [`_init_dims`](einshape.md#_init_dims)

### `MergeDims`
- def: [`jax/_src/pallas/einshape.py:94`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L94)
- signature: `class MergeDims:`
- members:
  - `transform_shape(self, shape: tuple[int, ...])` — [`L98`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L98)
  - `count` — [`L96`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L96)
  - `index` — [`L95`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L95)
- used by: [`_tile_preserving_einshape_kernel`](einshape.md#_tile_preserving_einshape_kernel), [`_einshape`](einshape.md#_einshape), [`_is_tile_preserving`](einshape.md#_is_tile_preserving), [`get_einshape_transforms`](einshape.md#get_einshape_transforms), [`Transform`](einshape.md#Transform)

### `SplitDims`
- def: [`jax/_src/pallas/einshape.py:85`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L85)
- signature: `class SplitDims:`
- members:
  - `transform_shape(self, shape: tuple[int, ...])` — [`L89`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L89)
  - `index` — [`L86`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L86)
  - `sizes` — [`L87`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L87)
- used by: [`_tile_preserving_einshape_kernel`](einshape.md#_tile_preserving_einshape_kernel), [`_einshape`](einshape.md#_einshape), [`_is_tile_preserving`](einshape.md#_is_tile_preserving), [`get_einshape_transforms`](einshape.md#get_einshape_transforms), [`Transform`](einshape.md#Transform)

### `Transpose`
- def: [`jax/_src/pallas/einshape.py:107`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L107)
- signature: `class Transpose:`
- members:
  - `transform_shape(self, shape: tuple[int, ...])` — [`L110`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L110)
  - `permutation` — [`L108`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L108)
- used by: [`_tile_preserving_einshape_kernel`](einshape.md#_tile_preserving_einshape_kernel), [`_einshape`](einshape.md#_einshape), [`_is_tile_preserving`](einshape.md#_is_tile_preserving), [`get_einshape_transforms`](einshape.md#get_einshape_transforms), [`Transform`](einshape.md#Transform)

## Functions
- `_2d_tile_array_to_array(tiles: np.ndarray)` — [`L465`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L465)
- `_apply_split(factors: list[Factor], targets: tuple[int, ...])` — [`L503`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L503)
- `_array_to_2d_tile_array(x: jax_typing.Array, tiling: tuple[int, ...])` — [`L447`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L447)
- `_consolidate(factors: list[Factor])` — [`L472`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L472) — Merges contiguous 'outer' factors to allow valid arbitrary outer-dimension reshapes.
- `_default_einshape_kernel(equation: str, x: jax_typing.Array, **sizes: int)` — [`L438`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L438)
- `_einshape(equation: str, value: jax_typing.Array, **sizes: int)` — [`L277`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L277) — Reshapes and transposes an array according to an einshape equation.
- `_einshape_kernel(equation: str, x: jax_typing.Array, assert_is_tile_preserving: bool, **size_vars: int)` — [`L627`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L627)
- `_einshape_lo_abstract_eval(x_aval: jax_core.ShapedArray, *, equation: str, sizes: tuple[tuple[str, int], ...], assert_is_tile_preserving: bool)` — [`L323`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L323)
- `_einshape_lo_lowering(ctx: mlir.LoweringRuleContext, x, *, equation: str, sizes: tuple[tuple[str, int], ...], assert_is_tile_preserving: bool)` — [`L337`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L337)
- `_einshape_lo_lowering_rule(ctx: tpu_lowering.LoweringRuleContext, x, *, equation: str, sizes: tuple[tuple[str, int], ...], assert_is_tile_preserving: bool)` — [`L648`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L648)
- `_init_dims(shape: tuple[int, ...], t1: int, t2: int)` — [`L483`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L483)
- `_is_tile_preserving(shape: tuple[int, ...], transforms: Sequence[Transform], tiling: tuple[int, int] | None = None)` — [`L586`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L586)
- `_parse_equation(equation: str)` — [`L156`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L156) — Parses an einshape equation.
- `_parse_side(s: str)` — [`L118`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L118) — Parses one side of an einshape equation into groups of named dimensions.
- `_tile_preserving_einshape_kernel(equation: str, x: jax_typing.Array, **size_vars: int)` — [`L548`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L548)
- `einshape(equation: str, x: jax_typing.Array, assert_is_tile_preserving: bool = False, **sizes: int)` — [`L393`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L393) — Reshapes and transposes an array according to an einshape equation.
- `einshape_lo(equation: str, x: jax_typing.Array, assert_is_tile_preserving: bool, **sizes: int)` — [`L311`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L311)
- `f(x)` — [`L347`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L347)
- `get_einshape_transforms(equation: str, input_shape: tuple[int, ...], **sizes: int)` — [`L164`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L164) — Parses an einshape equation into a sequence of transforms.
- `get_outer_shape(dims_list: list[list[Factor]])` — [`L561`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L561)

## Module values
- `Transform` — [`L115`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L115)
- `einshape_lo_p` — [`L308`](../../../../../../../raw/code/jax/jax/_src/pallas/einshape.py#L308)

