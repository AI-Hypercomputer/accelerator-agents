---
title: 'Module: qwix/contrib/calibration.py'
type: catalog
provenance: extracted
module: qwix/contrib/calibration.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix.contrib.calibration`/
symbols:
  extract_calibrated_quant_context: extract_calibrated_quant_context().
  quantize_params_with_calibration: quantize_params_with_calibration().
  SinglePassCalibrationProvider._collect_stats: SinglePassCalibrationProvider#_collect_stats().
  CalibrationProvider.dot_general: CalibrationProvider#dot_general().
  CalibrationProvider.get_rule_type: CalibrationProvider#get_rule_type().
  SinglePassCalibrationProvider: SinglePassCalibrationProvider#
  CalibrationProvider.get_stats_suffix: CalibrationProvider#get_stats_suffix().
  CalibrationProvider: CalibrationProvider#
  CalibratedQuantContext.weight: CalibratedQuantContext#weight.
  CalibrationProvider.einsum: CalibrationProvider#einsum().
  SinglePassCalibrationProvider.compute_stats: SinglePassCalibrationProvider#compute_stats().
  CalibrationProvider.get_intercept_map: CalibrationProvider#get_intercept_map().
  normalize_weight: normalize_weight().
  CalibratedQuantContext: CalibratedQuantContext#
  CalibratedQuantContext.abs_w: CalibratedQuantContext#abs_w.
  CalibrationProvider._collect_stats: CalibrationProvider#_collect_stats().
  CalibratedQuantContext.calibration_stats: CalibratedQuantContext#calibration_stats.
  CalibratedQuantContext.how: CalibratedQuantContext#how.
  CalibratedQuantContext.restore_shape: CalibratedQuantContext#restore_shape.
  CalibrationProvider.stats_dot_general: CalibrationProvider#stats_dot_general().
  CalibratedQuantContext.contracting_axis: CalibratedQuantContext#contracting_axis.
  CalibratedQuantContext.path: CalibratedQuantContext#path.
  normalize_weight.restore_shape: normalize_weight().restore_shape().
---
# Module: [`qwix/contrib/calibration.py`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py)

## Classes
### `CalibratedQuantContext`
- def: [`qwix/contrib/calibration.py:234`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L234) — documented in [qwix-contrib-gptq](../../../concepts/qwix-contrib-gptq.md)
- doc: A weight prepared for algorithm-specific quantization.
- signature: `class CalibratedQuantContext:`
- members:
  - `abs_w` — [`L250`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L250) — documented in [qwix-contrib-gptq](../../../concepts/qwix-contrib-gptq.md)
  - `calibration_stats` — [`L249`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L249)
  - `contracting_axis` — [`L251`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L251)
  - `how` — [`L248`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L248)
  - `path` — [`L253`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L253)
  - `restore_shape` — [`L252`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L252)
  - `weight` — [`L247`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L247)
- uses (calls/refs, reference-scoped): [`WithAux`](../_src/providers/ptq.md#WithAux)
- used by: [`extract_calibrated_quant_context`](calibration.md#extract_calibrated_quant_context), [`_quantize_weight`](qep.md#_quantize_weight), [`_quantize`](awq.md#quantize_params._quantize), [`quantize_params_with_calibration`](calibration.md#quantize_params_with_calibration), [`_quantize`](gptq.md#quantize_params._quantize), [`_quantize`](qep.md#quantize_params._quantize)

### `CalibrationProvider`  ·  implements/extends QuantizationProvider
- def: [`qwix/contrib/calibration.py:30`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L30) — documented in [qwix-contrib-gptq](../../../concepts/qwix-contrib-gptq.md)
- doc: Base class for calibration providers that intercept dot_general/einsum.
- signature: `class CalibrationProvider(qconfig.QuantizationProvider, metaclass=abc.ABCMeta):`
- members:
  - `_collect_stats(self, lhs: jax.Array, weight_name: str, *, module_path: tuple[str, ...], op_name: str, op_id: str | None, lhs_id: int)` — [`L48`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L48) — Collects statistics from the reshaped input activations.
  - `dot_general(self, lhs: jax.Array, rhs: jax.Array, dimension_numbers: jax.lax.DotDimensionNumbers, *args, rule: qconfig.QuantizationRule | None = None, op_id: str | None = None, **kwargs)` — [`L72`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L72) — Intercepts supported weight-bearing ``dot_general`` ops for calibration. — documented in [qwix-_src-utils-flax_util](../../../concepts/qwix-_src-utils-flax_util.md)
  - `einsum(self, einsum_str, *operands, **kwargs)` — [`L140`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L140) — Intercepts supported binary ``einsum`` ops via their lowered dot call.
  - `get_intercept_map(self)` — [`L163`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L163)
  - `get_rule_type(self)` — [`L40`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L40) — Returns the rule type that this provider handles.
  - `get_stats_suffix(self)` — [`L44`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L44) — Returns the suffix for the stats variable name (e.g., '_gptq').
  - `stats_dot_general(lhs, rhs, dimension_numbers, *args, **kwargs)` — [`L150`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L150)
- uses (calls/refs, reference-scoped): [`QuantizationRule`](../_src/qconfig.md#QuantizationRule), [`_get_current_rule_and_op_id`](../_src/qconfig.md#QuantizationProvider._get_current_rule_and_op_id), [`QuantizationProvider`](../_src/qconfig.md#QuantizationProvider), [`find_param`](../_src/utils/flax_util.md#find_param), [`_collect_stats`](qep.md#_CaptureProvider._collect_stats), [`get_intercept_map`](../_src/qconfig.md#QuantizationProvider.get_intercept_map), [`_collect_stats`](calibration.md#SinglePassCalibrationProvider._collect_stats), [`SinglePassCalibrationProvider`](calibration.md#SinglePassCalibrationProvider), [`get_current_module_path`](../_src/utils/flax_util.md#get_current_module_path), [`get_rule_type`](awq.md#AwqCalibrationProvider.get_rule_type), [`get_rule_type`](gptq.md#GptqCalibrationProvider.get_rule_type), [`get_rule_type`](qep.md#_CaptureProvider.get_rule_type), [`get_rule_type`](smooth_quant.md#SqCalibrationProvider.get_rule_type), [`_CaptureProvider`](qep.md#_CaptureProvider), [`get_stats_suffix`](awq.md#AwqCalibrationProvider.get_stats_suffix), [`get_stats_suffix`](gptq.md#GptqCalibrationProvider.get_stats_suffix), [`get_stats_suffix`](qep.md#_CaptureProvider.get_stats_suffix), [`get_stats_suffix`](smooth_quant.md#SqCalibrationProvider.get_stats_suffix)
- used by: [`QuantizationProvider`](../_src/qconfig.md#QuantizationProvider), [`get_intercept_map`](../_src/qconfig.md#QuantizationProvider.get_intercept_map), [`_collect_stats`](calibration.md#SinglePassCalibrationProvider._collect_stats), [`SinglePassCalibrationProvider`](calibration.md#SinglePassCalibrationProvider), [`_CaptureProvider`](qep.md#_CaptureProvider)

### `SinglePassCalibrationProvider`  ·  implements/extends CalibrationProvider
- def: [`qwix/contrib/calibration.py:170`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L170)
- doc: Calibration provider that collects single-pass statistics.
- signature: `class SinglePassCalibrationProvider(CalibrationProvider, metaclass=abc.ABCMeta):`
- members:
  - `_collect_stats(self, lhs: jax.Array, weight_name: str, *, module_path: tuple[str, ...], op_name: str, op_id: str | None, lhs_id: int)` — [`L182`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L182) — Accumulates one batch of single-pass calibration statistics. — documented in [qwix-_src-averaging](../../../concepts/qwix-_src-averaging.md)
  - `compute_stats(self, lhs: jax.Array)` — [`L179`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L179) — Computes statistics from the input array. — documented in [qwix-contrib-calibration](../../../concepts/qwix-contrib-calibration.md)
- uses (calls/refs, reference-scoped): [`compute_stats`](smooth_quant.md#SqCalibrationProvider.compute_stats), [`init`](../_src/averaging.md#SimpleMovingAverage.init), [`update`](../_src/averaging.md#SimpleMovingAverage.update), [`SimpleMovingAverage`](../_src/averaging.md#SimpleMovingAverage), [`get_or_create_variable`](../_src/utils/flax_util.md#get_or_create_variable), [`get_stats_suffix`](calibration.md#CalibrationProvider.get_stats_suffix), [`AwqCalibrationProvider`](awq.md#AwqCalibrationProvider), [`GptqCalibrationProvider`](gptq.md#GptqCalibrationProvider), [`CalibrationProvider`](calibration.md#CalibrationProvider), [`SqCalibrationProvider`](smooth_quant.md#SqCalibrationProvider), [`should_update_quant_stats`](../_src/utils/flax_util.md#should_update_quant_stats), [`compute_stats`](awq.md#AwqCalibrationProvider.compute_stats), [`compute_stats`](gptq.md#GptqCalibrationProvider.compute_stats)
- used by: [`AwqCalibrationProvider`](awq.md#AwqCalibrationProvider), [`GptqCalibrationProvider`](gptq.md#GptqCalibrationProvider), [`CalibrationProvider`](calibration.md#CalibrationProvider), [`SqCalibrationProvider`](smooth_quant.md#SqCalibrationProvider), [`_collect_stats`](calibration.md#CalibrationProvider._collect_stats)

## Functions
- `extract_calibrated_quant_context(path: tuple[str, ...], weight: jax.Array, abs_w: ptq.WithAux, stats: Any)` — [`L256`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L256) — Extracts the calibration context for a single weight. — documented in [qwix-_src-averaging](../../../concepts/qwix-_src-averaging.md)
- `normalize_weight(x: jax.Array, contraction_axis: int)` — [`L204`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L204) — Normalizes a weight tensor into (rows, columns) format. — documented in [qwix-contrib-calibration](../../../concepts/qwix-contrib-calibration.md)
- `quantize_params_with_calibration(params: Any, abstract_quantized_params: Any, quant_stats: Any, stats_suffix: str, quantize_fn: Callable[[CalibratedQuantContext], Any], *, allow_extra_params: bool = False)` — [`L304`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L304) — Shared framework for calibration-based weight quantization. — documented in [qwix-contrib-calibration](../../../concepts/qwix-contrib-calibration.md)
- `restore_shape(x)` — [`L226`](../../../../../../raw/code/qwix/qwix/contrib/calibration.py#L226)

