---
title: 'Module: tests/contrib/calibration_test.py'
type: catalog
provenance: extracted
module: tests/contrib/calibration_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests.contrib.calibration_test`/
symbols:
  QuantizeParamsWithCalibrationTest.test_nnx_returns_pure_dict: QuantizeParamsWithCalibrationTest#test_nnx_returns_pure_dict().
  QuantizeParamsWithCalibrationTest.test_matches_gptq_quantize_params: QuantizeParamsWithCalibrationTest#test_matches_gptq_quantize_params().
  QuantizeParamsWithCalibrationTest.test_delegates_to_quantize_fn: QuantizeParamsWithCalibrationTest#test_delegates_to_quantize_fn().
  QuantizeParamsWithCalibrationTest.test_ptq_fallback_for_unmatched_params: QuantizeParamsWithCalibrationTest#test_ptq_fallback_for_unmatched_params().
  QuantizeParamsWithCalibrationTest._setup_model_and_stats: QuantizeParamsWithCalibrationTest#_setup_model_and_stats().
  QuantizeParamsWithCalibrationTest.mock_quantize: QuantizeParamsWithCalibrationTest#mock_quantize().
  QuantizeParamsWithCalibrationTest.gptq_quantize: QuantizeParamsWithCalibrationTest#gptq_quantize().
  NormalizeWeightTest.test_basic_shape: NormalizeWeightTest#test_basic_shape().
  NormalizeWeightTest.test_contraction_axis_0: NormalizeWeightTest#test_contraction_axis_0().
  NormalizeWeightTest.test_contraction_axis_last: NormalizeWeightTest#test_contraction_axis_last().
  QuantizeParamsWithCalibrationTest._setup_model_and_stats.DenseModel: QuantizeParamsWithCalibrationTest#_setup_model_and_stats().DenseModel#
  NormalizeWeightTest: NormalizeWeightTest#
  QuantizeParamsWithCalibrationTest: QuantizeParamsWithCalibrationTest#
  QuantizeParamsWithCalibrationTest._setup_model_and_stats.DenseModel.__call__: QuantizeParamsWithCalibrationTest#_setup_model_and_stats().DenseModel#__call__().
---
# Module: [`tests/contrib/calibration_test.py`](../../../../../../raw/code/qwix/tests/contrib/calibration_test.py)

## Classes
### `DenseModel`  ·  implements/extends Module
- def: [`tests/contrib/calibration_test.py:65`](../../../../../../raw/code/qwix/tests/contrib/calibration_test.py#L65)
- signature: `class DenseModel(nn.Module):`
- protocol/private: `__call__`[`L68`](../../../../../../raw/code/qwix/tests/contrib/calibration_test.py#L68)
- used by: (1 test-only callers)

### `NormalizeWeightTest`  ·  implements/extends TestCase
- def: [`tests/contrib/calibration_test.py:31`](../../../../../../raw/code/qwix/tests/contrib/calibration_test.py#L31)
- signature: `class NormalizeWeightTest(parameterized.TestCase):`
- members:
  - `test_basic_shape(self)` — [`L33`](../../../../../../raw/code/qwix/tests/contrib/calibration_test.py#L33)
  - `test_contraction_axis_0(self)` — [`L41`](../../../../../../raw/code/qwix/tests/contrib/calibration_test.py#L41)
  - `test_contraction_axis_last(self)` — [`L50`](../../../../../../raw/code/qwix/tests/contrib/calibration_test.py#L50)
- uses (calls/refs, reference-scoped): [`normalize_weight`](../../qwix/contrib/calibration.md#normalize_weight)

### `QuantizeParamsWithCalibrationTest`  ·  implements/extends TestCase
- def: [`tests/contrib/calibration_test.py:60`](../../../../../../raw/code/qwix/tests/contrib/calibration_test.py#L60)
- signature: `class QuantizeParamsWithCalibrationTest(parameterized.TestCase):`
- members:
  - `_setup_model_and_stats(self, rules)` — [`L62`](../../../../../../raw/code/qwix/tests/contrib/calibration_test.py#L62) — Helper to create a model, calibrate with GPTQ, and return all pieces.
  - `gptq_quantize(ctx)` — [`L166`](../../../../../../raw/code/qwix/tests/contrib/calibration_test.py#L166)
  - `mock_quantize(ctx)` — [`L101`](../../../../../../raw/code/qwix/tests/contrib/calibration_test.py#L101)
  - `test_delegates_to_quantize_fn(self)` — [`L91`](../../../../../../raw/code/qwix/tests/contrib/calibration_test.py#L91) — Tests that quantize_fn is called with a properly constructed context.
  - `test_matches_gptq_quantize_params(self)` — [`L158`](../../../../../../raw/code/qwix/tests/contrib/calibration_test.py#L158) — Tests that the shared utility produces identical results to gptq.
  - `test_nnx_returns_pure_dict(self)` — [`L196`](../../../../../../raw/code/qwix/tests/contrib/calibration_test.py#L196)
  - `test_ptq_fallback_for_unmatched_params(self)` — [`L127`](../../../../../../raw/code/qwix/tests/contrib/calibration_test.py#L127) — Tests that params without calibration stats get PTQ quantization.
- uses (calls/refs, reference-scoped): [`weight_qtype`](../../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../../qwix/_src/model.md#quantize_model), [`module_path`](../../qwix/_src/qconfig.md#QuantizationRule.module_path), [`quantize`](../../qwix/_src/core/qarray.md#quantize), [`PtqProvider`](../../qwix/_src/providers/ptq.md#PtqProvider), [`quantize_weight`](../../qwix/contrib/gptq_core.md#quantize_weight), [`quantize_params_with_calibration`](../../qwix/contrib/calibration.md#quantize_params_with_calibration), [`GptqRule`](../../qwix/contrib/gptq.md#GptqRule), [`init`](../../qwix/_src/averaging.md#SimpleMovingAverage.init), [`update`](../../qwix/_src/averaging.md#SimpleMovingAverage.update), [`SimpleMovingAverage`](../../qwix/_src/averaging.md#SimpleMovingAverage), [`quantize_params`](../../qwix/contrib/gptq.md#quantize_params), [`GptqCalibrationProvider`](../../qwix/contrib/gptq.md#GptqCalibrationProvider)  (1 test-only)

