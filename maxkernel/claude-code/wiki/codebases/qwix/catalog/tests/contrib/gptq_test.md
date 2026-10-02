---
title: 'Module: tests/contrib/gptq_test.py'
type: catalog
provenance: extracted
module: tests/contrib/gptq_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests.contrib.gptq_test`/
symbols:
  GptqTest.test_einsum_model_linen: GptqTest#test_einsum_model_linen().
  GptqTest.test_dense_model_linen: GptqTest#test_dense_model_linen().
  GptqTest.test_dense_model_nnx: GptqTest#test_dense_model_nnx().
  GptqTest.test_einsum_model_nnx: GptqTest#test_einsum_model_nnx().
  GptqTest.test_mixed_model_nnx: GptqTest#test_mixed_model_nnx().
  GptqTest.test_mixed_model_linen: GptqTest#test_mixed_model_linen().
  GptqTest.calibrate_step: GptqTest#calibrate_step().
  GptqTest.test_dense_model_nnx.Model.__call__: GptqTest#test_dense_model_nnx().Model#__call__().
  GptqTest.test_einsum_model_nnx.EinsumModel.__call__: GptqTest#test_einsum_model_nnx().EinsumModel#__call__().
  GptqTest.test_mixed_model_nnx.MixedModel.__call__: GptqTest#test_mixed_model_nnx().MixedModel#__call__().
  GptqTest.test_dense_model_linen.DenseModel: GptqTest#test_dense_model_linen().DenseModel#
  GptqTest.test_dense_model_nnx.Model: GptqTest#test_dense_model_nnx().Model#
  Model.dense1: Model#dense1.
  Model.dense2: Model#dense2.
  GptqTest.test_einsum_model_linen.EinsumModel: GptqTest#test_einsum_model_linen().EinsumModel#
  GptqTest.test_einsum_model_nnx.EinsumModel: GptqTest#test_einsum_model_nnx().EinsumModel#
  EinsumModel.einsum1: EinsumModel#einsum1.
  EinsumModel.einsum2: EinsumModel#einsum2.
  GptqTest.test_mixed_model_linen.MixedModel: GptqTest#test_mixed_model_linen().MixedModel#
  GptqTest.test_mixed_model_nnx.MixedModel: GptqTest#test_mixed_model_nnx().MixedModel#
  MixedModel.key: MixedModel#key.
  GptqTest: GptqTest#
  GptqTest.test_dense_model_linen.DenseModel.__call__: GptqTest#test_dense_model_linen().DenseModel#__call__().
  GptqTest.test_dense_model_nnx.Model.__init__: GptqTest#test_dense_model_nnx().Model#__init__().
  GptqTest.test_einsum_model_linen.EinsumModel.__call__: GptqTest#test_einsum_model_linen().EinsumModel#__call__().
  GptqTest.test_einsum_model_nnx.EinsumModel.__init__: GptqTest#test_einsum_model_nnx().EinsumModel#__init__().
  GptqTest.test_mixed_model_linen.MixedModel.__call__: GptqTest#test_mixed_model_linen().MixedModel#__call__().
  GptqTest.test_mixed_model_nnx.MixedModel.__init__: GptqTest#test_mixed_model_nnx().MixedModel#__init__().
---
# Module: [`tests/contrib/gptq_test.py`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py)

## Classes
### `DenseModel`  ·  implements/extends Module
- def: [`tests/contrib/gptq_test.py:32`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L32)
- signature: `class DenseModel(nn.Module):`
- protocol/private: `__call__`[`L34`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L34)
- used by: (1 test-only callers)

### `EinsumModel`  ·  implements/extends Module
- def: [`tests/contrib/gptq_test.py:197`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L197)
- signature: `class EinsumModel(nnx.Module):`
- members:
  - `einsum1` — [`L200`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L200)
  - `einsum2` — [`L201`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L201)
- protocol/private: `__call__`[`L140`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L140), `__call__`[`L203`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L203), `__init__`[`L199`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L199)
- used by: (1 test-only callers)

### `GptqTest`  ·  implements/extends TestCase
- def: [`tests/contrib/gptq_test.py:27`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L27)
- signature: `class GptqTest(parameterized.TestCase):`
- members:
  - `calibrate_step(v, x)` — [`L156`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L156)
  - `test_dense_model_linen(self)` — [`L29`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L29) — Tests that a dense model is quantized correctly.
  - `test_dense_model_nnx(self)` — [`L82`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L82) — Tests that a dense model is quantized correctly with NNX.
  - `test_einsum_model_linen(self)` — [`L134`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L134) — Tests that an einsum model is quantized correctly.
  - `test_einsum_model_nnx(self)` — [`L194`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L194) — Tests that an einsum model is quantized correctly with NNX.
  - `test_mixed_model_linen(self)` — [`L246`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L246) — Tests that unsupported einsums (like Attention) are safely ignored.
  - `test_mixed_model_nnx(self)` — [`L287`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L287) — Tests that unsupported einsums are safely ignored in NNX.
- uses (calls/refs, reference-scoped): [`weight_qtype`](../../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../../qwix/_src/model.md#quantize_model), [`module_path`](../../qwix/_src/qconfig.md#QuantizationRule.module_path), [`PtqProvider`](../../qwix/_src/providers/ptq.md#PtqProvider), [`quantize_params`](../../qwix/_src/providers/ptq.md#quantize_params), [`GptqRule`](../../qwix/contrib/gptq.md#GptqRule), [`QuantStat`](../../qwix/_src/utils/flax_util.md#QuantStat), [`quantize_params`](../../qwix/contrib/gptq.md#quantize_params), [`GptqCalibrationProvider`](../../qwix/contrib/gptq.md#GptqCalibrationProvider)  (6 test-only)

### `MixedModel`  ·  implements/extends Module
- def: [`tests/contrib/gptq_test.py:290`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L290)
- signature: `class MixedModel(nnx.Module):`
- members:
  - `key` — [`L293`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L293)
- protocol/private: `__call__`[`L252`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L252), `__call__`[`L297`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L297), `__init__`[`L292`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L292)
- used by: (1 test-only callers)

### `Model`  ·  implements/extends Module
- def: [`tests/contrib/gptq_test.py:85`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L85)
- signature: `class Model(nnx.Module):`
- members:
  - `dense1` — [`L88`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L88)
  - `dense2` — [`L89`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L89)
- protocol/private: `__call__`[`L91`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L91), `__init__`[`L87`](../../../../../../raw/code/qwix/tests/contrib/gptq_test.py#L87)
- used by: (1 test-only callers)

