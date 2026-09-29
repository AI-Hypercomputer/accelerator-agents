---
title: 'Module: qwix/contrib/awq.py'
type: catalog
provenance: extracted
module: qwix/contrib/awq.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix.contrib.awq`/
symbols:
  quantize_params._quantize: quantize_params()._quantize().
  AwqInferenceProvider._apply_awq_scale: AwqInferenceProvider#_apply_awq_scale().
  quantize_params: quantize_params().
  AwqInferenceProvider.dot_general: AwqInferenceProvider#dot_general().
  WithAwqScale: WithAwqScale#
  AwqRule: AwqRule#
  AwqInferenceProvider.einsum: AwqInferenceProvider#einsum().
  AwqCalibrationProvider: AwqCalibrationProvider#
  AwqInferenceProvider: AwqInferenceProvider#
  AwqInferenceProvider.get_intercept_map: AwqInferenceProvider#get_intercept_map().
  AwqInferenceProvider._preprocess_operand: AwqInferenceProvider#_preprocess_operand().
  AwqCalibrationProvider.get_rule_type: AwqCalibrationProvider#get_rule_type().
  WithAwqScale.contracting_axis: WithAwqScale#contracting_axis.
  AwqCalibrationProvider.compute_stats: AwqCalibrationProvider#compute_stats().
  AwqCalibrationProvider.get_stats_suffix: AwqCalibrationProvider#get_stats_suffix().
  _STATS_SUFFIX: _STATS_SUFFIX.
  WithAwqScale.awq_scale: WithAwqScale#awq_scale.
  AwqRule.n_grid: AwqRule#n_grid.
---
# Module: [`qwix/contrib/awq.py`](../../../../../../raw/code/qwix/qwix/contrib/awq.py)

## Classes
### `AwqCalibrationProvider`  ·  implements/extends SinglePassCalibrationProvider
- def: [`qwix/contrib/awq.py:79`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L79)
- doc: Calibration provider for AWQ.
- signature: `class AwqCalibrationProvider(calibration.SinglePassCalibrationProvider):`
- members:
  - `compute_stats(self, lhs: jax.Array)` — [`L92`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L92)
  - `get_rule_type(self)` — [`L89`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L89)
  - `get_stats_suffix(self)` — [`L97`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L97)
- uses (calls/refs, reference-scoped): [`QuantizationRule`](../_src/qconfig.md#QuantizationRule), [`SinglePassCalibrationProvider`](calibration.md#SinglePassCalibrationProvider), [`AwqRule`](awq.md#AwqRule), [`_STATS_SUFFIX`](awq.md#_STATS_SUFFIX), [`compute_act_scale`](awq_core.md#compute_act_scale)
- used by: [`get_rule_type`](calibration.md#CalibrationProvider.get_rule_type), [`SinglePassCalibrationProvider`](calibration.md#SinglePassCalibrationProvider), [`get_stats_suffix`](calibration.md#CalibrationProvider.get_stats_suffix), [`compute_stats`](calibration.md#SinglePassCalibrationProvider.compute_stats)  (8 test-only)

### `AwqInferenceProvider`  ·  implements/extends PtqProvider
- def: [`qwix/contrib/awq.py:159`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L159)
- doc: Inference provider for AWQ.
- signature: `class AwqInferenceProvider(ptq.PtqProvider):`
- members:
  - `_apply_awq_scale(self, rhs: WithAwqScale)` — [`L170`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L170) — Dequantizes and applies per-channel AWQ scale compensation. — documented in [qwix-contrib-awq](../../../concepts/qwix-contrib-awq.md)
  - `dot_general(self, lhs: jax.Array, rhs: jax.Array | WithAwqScale | ptq.WithAux[qarray.QArray], dimension_numbers: jax.lax.DotDimensionNumbers, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, *, out_sharding: jax.sharding.NamedSharding | None = None)` — [`L182`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L182) — documented in [qwix-contrib-awq](../../../concepts/qwix-contrib-awq.md)
  - `einsum(self, einsum_str: str, *operands: jax.Array, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, _dot_general: Any = jax.lax.dot_general, out_sharding=None)` — [`L206`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L206)
  - `get_intercept_map(self)` — [`L235`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L235) — Used for interception.
- protocol/private: `_preprocess_operand`[`L215`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L215)
- uses (calls/refs, reference-scoped): [`QArray`](../_src/core/qarray.md#QArray), [`PtqProvider`](../_src/providers/ptq.md#PtqProvider), [`dequantize`](../_src/core/qarray.md#dequantize), [`WithAux`](../_src/providers/ptq.md#WithAux), [`dot_general`](../_src/providers/ptq.md#PtqProvider.dot_general), [`einsum`](../_src/providers/ptq.md#PtqProvider.einsum), [`get_intercept_map`](../_src/providers/ptq.md#PtqProvider.get_intercept_map), [`array`](../_src/providers/ptq.md#WithAux.array), [`WithAwqScale`](awq.md#WithAwqScale), [`contracting_axis`](awq.md#WithAwqScale.contracting_axis), [`awq_scale`](awq.md#WithAwqScale.awq_scale)
- used by: [`PtqProvider`](../_src/providers/ptq.md#PtqProvider), [`dot_general`](../_src/providers/ptq.md#PtqProvider.dot_general), [`einsum`](../_src/providers/ptq.md#PtqProvider.einsum), [`get_intercept_map`](../_src/providers/ptq.md#PtqProvider.get_intercept_map)  (5 test-only)

### `AwqRule`  ·  implements/extends QuantizationRule
- def: [`qwix/contrib/awq.py:48`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L48)
- doc: Use this rule to enable AWQ.
- signature: `class AwqRule(qconfig.QuantizationRule):`
- members:
  - `n_grid` — [`L55`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L55)
- uses (calls/refs, reference-scoped): [`QuantizationRule`](../_src/qconfig.md#QuantizationRule)
- used by: [`QuantizationRule`](../_src/qconfig.md#QuantizationRule), [`get_rule_type`](awq.md#AwqCalibrationProvider.get_rule_type)  (8 test-only)

### `WithAwqScale`  ·  implements/extends WithAux
- def: [`qwix/contrib/awq.py:59`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L59) — documented in [qwix-contrib-awq](../../../concepts/qwix-contrib-awq.md)
- doc: A quantized array with AWQ per-channel scales.
- signature: `class WithAwqScale(ptq.WithAux[qarray.QArray]):`
- members:
  - `awq_scale` — [`L72`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L72)
  - `contracting_axis` — [`L73`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L73)
- uses (calls/refs, reference-scoped): [`QArray`](../_src/core/qarray.md#QArray), [`WithAux`](../_src/providers/ptq.md#WithAux)
- used by: [`WithAux`](../_src/providers/ptq.md#WithAux), [`_quantize`](awq.md#quantize_params._quantize), [`_apply_awq_scale`](awq.md#AwqInferenceProvider._apply_awq_scale), [`dot_general`](awq.md#AwqInferenceProvider.dot_general), [`einsum`](awq.md#AwqInferenceProvider.einsum), [`_preprocess_operand`](awq.md#AwqInferenceProvider._preprocess_operand)  (1 test-only)

## Functions
- `_quantize(ctx: calibration.CalibratedQuantContext)` — [`L127`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L127)
- `quantize_params(params: Any, abstract_quantized_params: Any, awq_quant_stats: Any, *, allow_extra_params: bool = False, n_grid: int = 20)` — [`L101`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L101) — Quantizes the params with AWQ. — documented in [qwix-contrib-awq](../../../concepts/qwix-contrib-awq.md)

## Module values
- `_STATS_SUFFIX` — [`L44`](../../../../../../raw/code/qwix/qwix/contrib/awq.py#L44)

