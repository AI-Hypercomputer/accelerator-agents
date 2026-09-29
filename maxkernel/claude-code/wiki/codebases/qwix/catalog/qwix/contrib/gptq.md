---
title: 'Module: qwix/contrib/gptq.py'
type: catalog
provenance: extracted
module: qwix/contrib/gptq.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix.contrib.gptq`/
symbols:
  quantize_params._quantize: quantize_params()._quantize().
  GptqRule: GptqRule#
  quantize_params: quantize_params().
  GptqCalibrationProvider: GptqCalibrationProvider#
  GptqCalibrationProvider.get_rule_type: GptqCalibrationProvider#get_rule_type().
  GptqCalibrationProvider.compute_stats: GptqCalibrationProvider#compute_stats().
  GptqCalibrationProvider.get_stats_suffix: GptqCalibrationProvider#get_stats_suffix().
  _STATS_SUFFIX: _STATS_SUFFIX.
---
# Module: [`qwix/contrib/gptq.py`](../../../../../../raw/code/qwix/qwix/contrib/gptq.py)

## Classes
### `GptqCalibrationProvider`  ·  implements/extends SinglePassCalibrationProvider
- def: [`qwix/contrib/gptq.py:41`](../../../../../../raw/code/qwix/qwix/contrib/gptq.py#L41) — documented in [qwix-contrib-gptq](../../../concepts/qwix-contrib-gptq.md)
- doc: Calibration provider for GPTQ.
- signature: `class GptqCalibrationProvider(calibration.SinglePassCalibrationProvider):`
- members:
  - `compute_stats(self, lhs: jax.Array)` — [`L53`](../../../../../../raw/code/qwix/qwix/contrib/gptq.py#L53)
  - `get_rule_type(self)` — [`L50`](../../../../../../raw/code/qwix/qwix/contrib/gptq.py#L50)
  - `get_stats_suffix(self)` — [`L57`](../../../../../../raw/code/qwix/qwix/contrib/gptq.py#L57)
- uses (calls/refs, reference-scoped): [`QuantizationRule`](../_src/qconfig.md#QuantizationRule), [`GptqRule`](gptq.md#GptqRule), [`SinglePassCalibrationProvider`](calibration.md#SinglePassCalibrationProvider), [`compute_hessian`](gptq_core.md#compute_hessian), [`_STATS_SUFFIX`](gptq.md#_STATS_SUFFIX)
- used by: [`get_rule_type`](calibration.md#CalibrationProvider.get_rule_type), [`SinglePassCalibrationProvider`](calibration.md#SinglePassCalibrationProvider), [`get_stats_suffix`](calibration.md#CalibrationProvider.get_stats_suffix), [`compute_stats`](calibration.md#SinglePassCalibrationProvider.compute_stats)  (8 test-only)

### `GptqRule`  ·  implements/extends QuantizationRule
- def: [`qwix/contrib/gptq.py:37`](../../../../../../raw/code/qwix/qwix/contrib/gptq.py#L37) — documented in [qwix-contrib-gptq](../../../concepts/qwix-contrib-gptq.md)
- doc: Use this rule to enable GPTQ.
- signature: `class GptqRule(qconfig.QuantizationRule):`
- uses (calls/refs, reference-scoped): [`QuantizationRule`](../_src/qconfig.md#QuantizationRule), [`QepRule`](qep.md#QepRule)
- used by: [`QuantizationRule`](../_src/qconfig.md#QuantizationRule), [`QepRule`](qep.md#QepRule), [`get_rule_type`](gptq.md#GptqCalibrationProvider.get_rule_type)  (11 test-only)

## Functions
- `_quantize(ctx: calibration.CalibratedQuantContext)` — [`L85`](../../../../../../raw/code/qwix/qwix/contrib/gptq.py#L85) — documented in [qwix-contrib-gptq](../../../concepts/qwix-contrib-gptq.md)
- `quantize_params(params: Any, abstract_quantized_params: Any, gptq_quant_stats: Any, *, allow_extra_params: bool = False, gptq_block_size: int = 128, gptq_damping_factor: float = 0.01)` — [`L61`](../../../../../../raw/code/qwix/qwix/contrib/gptq.py#L61) — Quantizes the params with GPTQ. — documented in [qwix-contrib-gptq](../../../concepts/qwix-contrib-gptq.md)

## Module values
- `_STATS_SUFFIX` — [`L33`](../../../../../../raw/code/qwix/qwix/contrib/gptq.py#L33)

