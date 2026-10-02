---
title: 'Module: qwix/contrib/smooth_quant.py'
type: catalog
provenance: extracted
module: qwix/contrib/smooth_quant.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix.contrib.smooth_quant`/
symbols:
  SqCalibrationProvider.compute_stats: SqCalibrationProvider#compute_stats().
  quantize_params: quantize_params().
  SqInferenceProvider.dot_general: SqInferenceProvider#dot_general().
  WithSqScale: WithSqScale#
  SqRule: SqRule#
  SqCalibrationProvider: SqCalibrationProvider#
  SqInferenceProvider: SqInferenceProvider#
  SqCalibrationProvider.get_rule_type: SqCalibrationProvider#get_rule_type().
  SqInferenceProvider.get_intercept_map: SqInferenceProvider#get_intercept_map().
  SqInferenceProvider._apply_sq_scale: SqInferenceProvider#_apply_sq_scale().
  WithSqScale.contracting_axis: WithSqScale#contracting_axis.
  compute_scales_ratio: compute_scales_ratio().
  SqRule.alpha: SqRule#alpha.
  WithSqScale.inv_sq_scale: WithSqScale#inv_sq_scale.
  SqCalibrationProvider.get_stats_suffix: SqCalibrationProvider#get_stats_suffix().
---
# Module: [`qwix/contrib/smooth_quant.py`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py)

## Classes
### `SqCalibrationProvider`  ·  implements/extends SinglePassCalibrationProvider
- def: [`qwix/contrib/smooth_quant.py:83`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py#L83)
- doc: Calibration provider for SQ.
- signature: `class SqCalibrationProvider(calibration.SinglePassCalibrationProvider):`
- members:
  - `compute_stats(self, lhs: jax.Array)` — [`L95`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py#L95) — documented in [qwix-_src-core-qarray](../../../concepts/qwix-_src-core-qarray.md)
  - `get_rule_type(self)` — [`L92`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py#L92)
  - `get_stats_suffix(self)` — [`L158`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py#L158)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../_src/qconfig.md#QuantizationRule.weight_qtype), [`QuantizationRule`](../_src/qconfig.md#QuantizationRule), [`HowToQuantize`](../_src/core/qarray.md#HowToQuantize), [`act_qtype`](../_src/qconfig.md#QuantizationRule.act_qtype), [`qtype`](../_src/core/qarray.md#HowToQuantize.qtype), [`channelwise_axes`](../_src/core/qarray.md#HowToQuantize.channelwise_axes), [`calibration_method`](../_src/core/qarray.md#HowToQuantize.calibration_method), [`_get_current_rule_and_op_id`](../_src/qconfig.md#QuantizationProvider._get_current_rule_and_op_id), [`calibrate`](../_src/core/qarray.md#calibrate), [`compute_scale_zero_point`](../_src/core/qarray.md#compute_scale_zero_point), [`weight_calibration_method`](../_src/qconfig.md#QuantizationRule.weight_calibration_method), [`act_calibration_method`](../_src/qconfig.md#QuantizationRule.act_calibration_method), [`get_current_module`](../_src/utils/flax_util.md#get_current_module), [`SinglePassCalibrationProvider`](calibration.md#SinglePassCalibrationProvider), `scope`, [`SqRule`](smooth_quant.md#SqRule), [`alpha`](smooth_quant.md#SqRule.alpha), [`compute_scales_ratio`](smooth_quant.md#compute_scales_ratio)
- used by: [`get_rule_type`](calibration.md#CalibrationProvider.get_rule_type), [`SinglePassCalibrationProvider`](calibration.md#SinglePassCalibrationProvider), [`get_stats_suffix`](calibration.md#CalibrationProvider.get_stats_suffix), [`compute_stats`](calibration.md#SinglePassCalibrationProvider.compute_stats)  (6 test-only)

### `SqInferenceProvider`  ·  implements/extends PtqProvider
- def: [`qwix/contrib/smooth_quant.py:251`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py#L251)
- doc: Inference provider for SQ.
- signature: `class SqInferenceProvider(ptq.PtqProvider):`
- members:
  - `_apply_sq_scale(self, lhs: jax.Array, inv_sq_scale: jax.Array)` — [`L254`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py#L254) — Applies per-channel SQ scale compensation.
  - `dot_general(self, lhs: jax.Array, rhs: jax.Array | WithSqScale | ptq.WithAux[qarray.QArray], dimension_numbers: jax.lax.DotDimensionNumbers, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, *, out_sharding: jax.sharding.NamedSharding | None = None)` — [`L264`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py#L264) — documented in [qwix-_src-core-dot_general](../../../concepts/qwix-_src-core-dot_general.md)
  - `get_intercept_map(self)` — [`L288`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py#L288) — Used for interception.
- uses (calls/refs, reference-scoped): [`QArray`](../_src/core/qarray.md#QArray), [`PtqProvider`](../_src/providers/ptq.md#PtqProvider), [`WithAux`](../_src/providers/ptq.md#WithAux), [`dot_general`](../_src/core/dot_general.md#dot_general), [`get_intercept_map`](../_src/providers/ptq.md#PtqProvider.get_intercept_map), [`array`](../_src/providers/ptq.md#WithAux.array), [`WithSqScale`](smooth_quant.md#WithSqScale), [`inv_sq_scale`](smooth_quant.md#WithSqScale.inv_sq_scale)
- used by: [`PtqProvider`](../_src/providers/ptq.md#PtqProvider), [`dot_general`](../_src/providers/ptq.md#PtqProvider.dot_general), [`get_intercept_map`](../_src/providers/ptq.md#PtqProvider.get_intercept_map)  (4 test-only)

### `SqRule`  ·  implements/extends QuantizationRule
- def: [`qwix/contrib/smooth_quant.py:49`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py#L49) — documented in [qwix-contrib-smooth_quant](../../../concepts/qwix-contrib-smooth_quant.md)
- doc: Use this rule to enable SQ.
- signature: `class SqRule(qconfig.QuantizationRule):`
- members:
  - `alpha` — [`L56`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py#L56)
- uses (calls/refs, reference-scoped): [`QuantizationRule`](../_src/qconfig.md#QuantizationRule)
- used by: [`QuantizationRule`](../_src/qconfig.md#QuantizationRule), [`compute_stats`](smooth_quant.md#SqCalibrationProvider.compute_stats), [`get_rule_type`](smooth_quant.md#SqCalibrationProvider.get_rule_type)  (6 test-only)

### `WithSqScale`  ·  implements/extends WithAux
- def: [`qwix/contrib/smooth_quant.py:60`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py#L60) — documented in [qwix-contrib-smooth_quant](../../../concepts/qwix-contrib-smooth_quant.md)
- doc: A quantized array with SQ per-channel scales.
- signature: `class WithSqScale(ptq.WithAux[qarray.QArray]):`
- members:
  - `contracting_axis` — [`L76`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py#L76)
  - `inv_sq_scale` — [`L75`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py#L75)
- uses (calls/refs, reference-scoped): [`QArray`](../_src/core/qarray.md#QArray), [`WithAux`](../_src/providers/ptq.md#WithAux)
- used by: [`WithAux`](../_src/providers/ptq.md#WithAux), [`quantize_params`](smooth_quant.md#quantize_params), [`dot_general`](smooth_quant.md#SqInferenceProvider.dot_general)  (2 test-only)

## Functions
- `compute_scales_ratio(act_stats: jax.Array, weight_stats: jax.Array, alpha: float)` — [`L39`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py#L39) — Compute the scales ratio for Smooth Quantization (SQ).
- `quantize_params(params: Any, abstract_quantized_params: Any, sq_quant_stats: Any, *, allow_extra_params: bool = False)` — [`L162`](../../../../../../raw/code/qwix/qwix/contrib/smooth_quant.py#L162) — Quantizes parameters with Smooth Quantization (SQ). — documented in [qwix-_src-providers-ptq](../../../concepts/qwix-_src-providers-ptq.md)

