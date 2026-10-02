---
title: 'Module: tests/contrib/smooth_quant_test.py'
type: catalog
provenance: extracted
module: tests/contrib/smooth_quant_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests.contrib.smooth_quant_test`/
symbols:
  SqTest.test_dense_model_linen: SqTest#test_dense_model_linen().
  SqTest.test_dense_model_nnx: SqTest#test_dense_model_nnx().
  SqTest.test_partial_quantization: SqTest#test_partial_quantization().
  SqTest.test_multi_layer_model: SqTest#test_multi_layer_model().
  SqTest.test_different_qtypes: SqTest#test_different_qtypes().
  SqTest.test_multiple_calibration_batches: SqTest#test_multiple_calibration_batches().
  SqTest.test_dense_model_nnx.Model.__call__: SqTest#test_dense_model_nnx().Model#__call__().
  SqTest.extract_fn_quant_stats: SqTest#extract_fn_quant_stats().
  SqTest.test_dense_model_linen.Model: SqTest#test_dense_model_linen().Model#
  SqTest.test_dense_model_nnx.Model: SqTest#test_dense_model_nnx().Model#
  Model.dense1: Model#dense1.
  Model.dense2: Model#dense2.
  SqTest.test_multiple_calibration_batches.Model: SqTest#test_multiple_calibration_batches().Model#
  SqTest.test_different_qtypes.Model: SqTest#test_different_qtypes().Model#
  SqTest.test_multi_layer_model.Model: SqTest#test_multi_layer_model().Model#
  SqTest.test_partial_quantization.Model: SqTest#test_partial_quantization().Model#
  SqTest: SqTest#
  SqTest.test_dense_model_linen.Model.__call__: SqTest#test_dense_model_linen().Model#__call__().
  SqTest.test_dense_model_nnx.Model.__init__: SqTest#test_dense_model_nnx().Model#__init__().
  SqTest.test_multiple_calibration_batches.Model.__call__: SqTest#test_multiple_calibration_batches().Model#__call__().
  SqTest.test_different_qtypes.Model.__call__: SqTest#test_different_qtypes().Model#__call__().
  SqTest.test_multi_layer_model.Model.__call__: SqTest#test_multi_layer_model().Model#__call__().
  SqTest.test_partial_quantization.Model.__call__: SqTest#test_partial_quantization().Model#__call__().
---
# Module: [`tests/contrib/smooth_quant_test.py`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py)

## Classes
### `Model`  ·  implements/extends Module
- def: [`tests/contrib/smooth_quant_test.py:302`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L302)
- signature: `class Model(nn.Module):`
- members:
  - `dense1` — [`L117`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L117)
  - `dense2` — [`L118`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L118)
- protocol/private: `__call__`[`L42`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L42), `__call__`[`L120`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L120), `__call__`[`L196`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L196), `__call__`[`L225`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L225), `__call__`[`L255`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L255), `__call__`[`L305`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L305), `__init__`[`L116`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L116)
- used by: (1 test-only callers)

### `SqTest`  ·  implements/extends TestCase
- def: [`tests/contrib/smooth_quant_test.py:27`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L27)
- signature: `class SqTest(parameterized.TestCase):`
- members:
  - `extract_fn_quant_stats(x)` — [`L156`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L156)
  - `test_dense_model_linen(self)` — [`L29`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L29) — Test SQ calibration and quantization on a simple dense model.
  - `test_dense_model_nnx(self)` — [`L111`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L111) — Test SQ calibration and quantization on a simple dense model with nnx.
  - `test_different_qtypes(self, qtype)` — [`L219`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L219) — Test SQ with different quantization types.
  - `test_multi_layer_model(self)` — [`L249`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L249) — Test SQ on a model with multiple quantized layers.
  - `test_multiple_calibration_batches(self)` — [`L190`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L190) — Test that SQ properly averages across calibration batches.
  - `test_partial_quantization(self)` — [`L299`](../../../../../../raw/code/qwix/tests/contrib/smooth_quant_test.py#L299) — Test SQ with only some layers quantized.
- uses (calls/refs, reference-scoped): [`weight_qtype`](../../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../../qwix/_src/model.md#quantize_model), [`act_qtype`](../../qwix/_src/qconfig.md#QuantizationRule.act_qtype), [`module_path`](../../qwix/_src/qconfig.md#QuantizationRule.module_path), [`PtqProvider`](../../qwix/_src/providers/ptq.md#PtqProvider), [`quantize_params`](../../qwix/_src/providers/ptq.md#quantize_params), [`tile_size`](../../qwix/_src/qconfig.md#QuantizationRule.tile_size), [`quantize_params`](../../qwix/contrib/smooth_quant.md#quantize_params), [`QuantStat`](../../qwix/_src/utils/flax_util.md#QuantStat), [`WithSqScale`](../../qwix/contrib/smooth_quant.md#WithSqScale), [`SqRule`](../../qwix/contrib/smooth_quant.md#SqRule), [`SqCalibrationProvider`](../../qwix/contrib/smooth_quant.md#SqCalibrationProvider), [`SqInferenceProvider`](../../qwix/contrib/smooth_quant.md#SqInferenceProvider)  (6 test-only)

