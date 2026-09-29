---
title: 'Module: tests/contrib/awq_test.py'
type: catalog
provenance: extracted
module: tests/contrib/awq_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests.contrib.awq_test`/AwqTest#
symbols:
  AwqTest.test_dense_model_linen: test_dense_model_linen().
  AwqTest.test_awq_outperforms_ptq: test_awq_outperforms_ptq().
  AwqTest.test_einsum_model_linen: test_einsum_model_linen().
  AwqTest.test_partial_quantization: test_partial_quantization().
  AwqTest.test_multi_layer_model: test_multi_layer_model().
  AwqTest.test_different_qtypes: test_different_qtypes().
  AwqTest.test_mixed_model_safety: test_mixed_model_safety().
  AwqTest.test_multiple_calibration_batches: test_multiple_calibration_batches().
  AwqTest.calibrate_step: calibrate_step().
  AwqTest.test_dense_model_linen.Model: test_dense_model_linen().Model#
  AwqTest.test_multiple_calibration_batches.Model: test_multiple_calibration_batches().Model#
  AwqTest.test_different_qtypes.Model: test_different_qtypes().Model#
  AwqTest.test_multi_layer_model.Model: test_multi_layer_model().Model#
  AwqTest.test_partial_quantization.Model: test_partial_quantization().Model#
  AwqTest.test_awq_outperforms_ptq.Model: test_awq_outperforms_ptq().Model#
  AwqTest.test_einsum_model_linen.EinsumModel: test_einsum_model_linen().EinsumModel#
  AwqTest.test_mixed_model_safety.MixedModel: test_mixed_model_safety().MixedModel#
  AwqTest: ''
  AwqTest.test_dense_model_linen.Model.__call__: test_dense_model_linen().Model#__call__().
  AwqTest.test_multiple_calibration_batches.Model.__call__: test_multiple_calibration_batches().Model#__call__().
  AwqTest.test_different_qtypes.Model.__call__: test_different_qtypes().Model#__call__().
  AwqTest.test_multi_layer_model.Model.__call__: test_multi_layer_model().Model#__call__().
  AwqTest.test_partial_quantization.Model.__call__: test_partial_quantization().Model#__call__().
  AwqTest.test_awq_outperforms_ptq.Model.__call__: test_awq_outperforms_ptq().Model#__call__().
  AwqTest.test_einsum_model_linen.EinsumModel.__call__: test_einsum_model_linen().EinsumModel#__call__().
  AwqTest.test_mixed_model_safety.MixedModel.__call__: test_mixed_model_safety().MixedModel#__call__().
---
# Module: [`tests/contrib/awq_test.py`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py)

## Classes
### `AwqTest`  ·  implements/extends TestCase
- def: [`tests/contrib/awq_test.py:25`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L25)
- signature: `class AwqTest(parameterized.TestCase):`
- members:
  - `calibrate_step(v, x)` — [`L361`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L361)
  - `test_awq_outperforms_ptq(self)` — [`L256`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L256) — Test that AWQ provides better accuracy than plain PTQ.
  - `test_dense_model_linen(self)` — [`L27`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L27) — Test AWQ calibration and quantization on a simple dense model.
  - `test_different_qtypes(self, qtype)` — [`L133`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L133) — Test AWQ with different quantization types.
  - `test_einsum_model_linen(self)` — [`L333`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L333) — Tests that an einsum model is quantized correctly with AWQ.
  - `test_mixed_model_safety(self)` — [`L404`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L404) — Tests that unsupported einsums (like Attention) are safely ignored.
  - `test_multi_layer_model(self)` — [`L163`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L163) — Test AWQ on a model with multiple quantized layers.
  - `test_multiple_calibration_batches(self)` — [`L104`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L104) — Test that AWQ properly averages across calibration batches.
  - `test_partial_quantization(self)` — [`L213`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L213) — Test AWQ with only some layers quantized.
- uses (calls/refs, reference-scoped): [`weight_qtype`](../../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../../qwix/_src/model.md#quantize_model), [`module_path`](../../qwix/_src/qconfig.md#QuantizationRule.module_path), [`PtqProvider`](../../qwix/_src/providers/ptq.md#PtqProvider), [`quantize_params`](../../qwix/_src/providers/ptq.md#quantize_params), [`tile_size`](../../qwix/_src/qconfig.md#QuantizationRule.tile_size), [`quantize_params`](../../qwix/contrib/awq.md#quantize_params), [`WithAwqScale`](../../qwix/contrib/awq.md#WithAwqScale), [`AwqRule`](../../qwix/contrib/awq.md#AwqRule), [`AwqCalibrationProvider`](../../qwix/contrib/awq.md#AwqCalibrationProvider), [`AwqInferenceProvider`](../../qwix/contrib/awq.md#AwqInferenceProvider)  (8 test-only)

### `EinsumModel`  ·  implements/extends Module
- def: [`tests/contrib/awq_test.py:336`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L336)
- signature: `class EinsumModel(nn.Module):`
- protocol/private: `__call__`[`L339`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L339)
- used by: (1 test-only callers)

### `MixedModel`  ·  implements/extends Module
- def: [`tests/contrib/awq_test.py:407`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L407)
- signature: `class MixedModel(nn.Module):`
- protocol/private: `__call__`[`L410`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L410)
- used by: (1 test-only callers)

### `Model`  ·  implements/extends Module
- def: [`tests/contrib/awq_test.py:267`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L267)
- signature: `class Model(nn.Module):`
- protocol/private: `__call__`[`L40`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L40), `__call__`[`L110`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L110), `__call__`[`L139`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L139), `__call__`[`L169`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L169), `__call__`[`L219`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L219), `__call__`[`L270`](../../../../../../raw/code/qwix/tests/contrib/awq_test.py#L270)
- used by: (1 test-only callers)

