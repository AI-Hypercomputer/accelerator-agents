---
title: 'Module: aqt/jax/v2/tiled_dot_general_test.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/tiled_dot_general_test.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.tiled_dot_general_test`/
symbols:
  AxisIdx: AxisIdx.
  TensorTiling: TensorTiling.
  Cfg: Cfg.
  AxisTiling: AxisTiling.
  get_shape_from_axes: get_shape_from_axes().
  assign_input_shape: assign_input_shape().
  get_axis_tiles: get_axis_tiles().
  generate_inputs: generate_inputs().
  generate_tiling_cfgs: generate_tiling_cfgs().
  TiledDotGeneralTest: TiledDotGeneralTest#
  TiledDotGeneralTest.test_tiled_dot_general_shape: TiledDotGeneralTest#test_tiled_dot_general_shape().
  TiledDotGeneralTest.test_tiled_dot_general: TiledDotGeneralTest#test_tiled_dot_general().
  TiledDotGeneralTest.test_single: TiledDotGeneralTest#test_single().
  TiledDotGeneralTest.test_tiling_state_for_single_tensor: TiledDotGeneralTest#test_tiling_state_for_single_tensor().
  TiledDotGeneralTest.test_negative_axis_index: TiledDotGeneralTest#test_negative_axis_index().
---
# Module: [`aqt/jax/v2/tiled_dot_general_test.py`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py)

## Classes
### `TiledDotGeneralTest`
- def: [`aqt/jax/v2/tiled_dot_general_test.py:144`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py#L144)
- signature: `class TiledDotGeneralTest(parameterized.TestCase):`
- members:
  - `test_negative_axis_index(self)` — [`L277`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py#L277)
  - `test_single(self)` — [`L226`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py#L226)
  - `test_tiled_dot_general(self)` — [`L191`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py#L191)
  - `test_tiled_dot_general_shape(self)` — [`L146`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py#L146)
  - `test_tiling_state_for_single_tensor(self)` — [`L256`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py#L256)

## Functions
- `assign_input_shape(rng_key, ca_shape: list[int], ba_shape: list[int], ra_shape: list[int])` — [`L39`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py#L39) — Randomly assign an axis to a shape.
- `generate_inputs(rng_key, num_ca: int, num_ba: int, num_lhs_ra: int, num_rhs_ra: int, max_shape_val: int)` — [`L76`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py#L76)
- `generate_tiling_cfgs(lhs_shape, lhs_ca, lhs_ra, rhs_shape, rhs_ca, rhs_ra)` — [`L117`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py#L117)
- `get_axis_tiles(axes: list[AxisIdx], input_shape: list[int], subsample_ts: int=2)` — [`L56`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py#L56)
- `get_shape_from_axes(axes: list[AxisIdx], shape)` — [`L35`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py#L35)

## Module values
- `AxisIdx` — [`L29`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py#L29)
- `AxisTiling` — [`L32`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py#L32)
- `Cfg` — [`L31`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py#L31)
- `TensorTiling` — [`L30`](../../../../../../../raw/code/aqt/aqt/jax/v2/tiled_dot_general_test.py#L30)

