---
title: 'Module: aqt/jax/v2/utils.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/utils.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.utils`/
symbols:
  static_field: static_field().
  AxisIdx: AxisIdx.
  flax_slots_kw_only_dataclass: flax_slots_kw_only_dataclass.
  Context: Context#
  get_remaining_axes: get_remaining_axes().
  DotGeneralT: DotGeneralT.
  AxisSize: AxisSize.
  ShapeTemplate: ShapeTemplate.
  AutoScaleSearchConfig: AutoScaleSearchConfig.
  assert_shape: assert_shape().
  assert_eq: assert_eq().
  flax_slots_dataclass: flax_slots_dataclass.
  dataclass_field: dataclass_field().
  QuantMode: QuantMode#
  QuantMode.TRAIN: QuantMode#TRAIN.
  QuantMode.CALIBRATE: QuantMode#CALIBRATE.
  QuantMode.CONVERT: QuantMode#CONVERT.
  QuantMode.SERVE: QuantMode#SERVE.
  dynamic_field: dynamic_field().
  print_diff: print_diff().
  test_pprint_eq: test_pprint_eq().
  infer_dtype_from_bits: infer_dtype_from_bits().
  Context.key: Context#key.
  Context.train_step: Context#train_step.
  Context.quant_mode: Context#quant_mode.
---
# Module: [`aqt/jax/v2/utils.py`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py)

## Classes
### `Context`
- def: [`aqt/jax/v2/utils.py:152`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L152) — documented in [aqt-jax-v2-aqt_quantizer](../../../../concepts/aqt-jax-v2-aqt_quantizer.md)
- signature: `class Context:`
- members:
  - `key` — [`L153`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L153)
  - `quant_mode` — [`L155`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L155)
  - `train_step` — [`L154`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L154)
- used by: [`quantizer_make`](aqt_quantizer.md#quantizer_make), `test_e1m2_vs_e0m3`, [`set_context`](../aqt_dot_general.md#DefaultDotGeneralQuantizer.set_context), [`context`](aqt_quantizer.md#Quantizer.context), [`set_context`](../aqt_dot_general.md#DotGeneralQuantizer.set_context)

### `QuantMode`
- def: [`aqt/jax/v2/utils.py:85`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L85)
- signature: `class QuantMode(enum.Enum):`
- members:
  - `CALIBRATE` — [`L87`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L87)
  - `CONVERT` — [`L88`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L88)
  - `SERVE` — [`L89`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L89)
  - `TRAIN` — [`L86`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L86)

## Functions
- `assert_eq(value: Any, expected: Any, value_name: str)` — [`L58`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L58)
- `assert_shape(shape: Sequence[int], shape_template: ShapeTemplate, msg: str)` — [`L48`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L48)
- `dataclass_field(default='no default value')` — [`L76`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L76)
- `dynamic_field(**kwargs)` — [`L96`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L96)
- `get_remaining_axes(rank: int, contraction_axes: Sequence[AxisIdx], batch_axes: Sequence[AxisIdx])` — [`L138`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L138) — Returns the remaining axes. — documented in [aqt-jax-aqt_dot_general](../../../../concepts/aqt-jax-aqt_dot_general.md)
- `infer_dtype_from_bits(bits: int)` — [`L120`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L120) — Get the dtype for the number of bits provided.
- `print_diff(str_a: str, str_b: str, do_print_diff=False)` — [`L100`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L100)
- `static_field(**kwargs)` — [`L92`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L92) — documented in [aqt-jax-v2-utils](../../../../concepts/aqt-jax-v2-utils.md)
- `test_pprint_eq(input_a: Any, input_b: Any, remove_memory_addresses: bool=False)` — [`L109`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L109)

## Module values
- `AutoScaleSearchConfig` — [`L45`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L45)
- `AxisIdx` — [`L34`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L34) — documented in [aqt-jax-v2-utils](../../../../concepts/aqt-jax-v2-utils.md)
- `AxisSize` — [`L35`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L35)
- `DotGeneralT` — [`L41`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L41)
- `ShapeTemplate` — [`L38`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L38)
- `flax_slots_dataclass` — [`L63`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L63)
- `flax_slots_kw_only_dataclass` — [`L71`](../../../../../../../raw/code/aqt/aqt/jax/v2/utils.py#L71) — documented in [aqt-jax-v2-utils](../../../../concepts/aqt-jax-v2-utils.md)

