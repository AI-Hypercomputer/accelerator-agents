---
title: 'Module: integration_tests/coverage_test.py'
type: catalog
provenance: extracted
module: integration_tests/coverage_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `integration_tests.coverage_test`/
symbols:
  CoverageTest.test_coverage: CoverageTest#test_coverage().
  Transformer.__call__: Transformer#__call__().
  Transformer: Transformer#
  Transformer.vocab_size: Transformer#vocab_size.
  Transformer.embedding_dim: Transformer#embedding_dim.
  CNN: CNN#
  Transformer.num_layers: Transformer#num_layers.
  Transformer.num_heads: Transformer#num_heads.
  Transformer.qkv_features: Transformer#qkv_features.
  Transformer.hidden_dim: Transformer#hidden_dim.
  Transformer.create_inputs: Transformer#create_inputs().
  CNN.__call__: CNN#__call__().
  CNN.create_inputs: CNN#create_inputs().
  CoverageTest: CoverageTest#
---
# Module: [`integration_tests/coverage_test.py`](../../../../../raw/code/qwix/integration_tests/coverage_test.py)

## Classes
### `CNN`  ·  implements/extends Module
- def: [`integration_tests/coverage_test.py:28`](../../../../../raw/code/qwix/integration_tests/coverage_test.py#L28)
- doc: A simple CNN model.
- signature: `class CNN(nn.Module):`
- members:
  - `create_inputs(self)` — [`L45`](../../../../../raw/code/qwix/integration_tests/coverage_test.py#L45)
- protocol/private: `__call__`[`L32`](../../../../../raw/code/qwix/integration_tests/coverage_test.py#L32)
- used by: [`test_coverage`](coverage_test.md#CoverageTest.test_coverage)

### `CoverageTest`  ·  implements/extends TestCase
- def: [`integration_tests/coverage_test.py:81`](../../../../../raw/code/qwix/integration_tests/coverage_test.py#L81)
- signature: `class CoverageTest(parameterized.TestCase):`
- members:
  - `test_coverage(self, model: Transformer | CNN, rule: qconfig.QuantizationRule)` — [`L104`](../../../../../raw/code/qwix/integration_tests/coverage_test.py#L104) — documented in [qwix-_src-qconfig](../../concepts/qwix-_src-qconfig.md)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../qwix/_src/model.md#quantize_model), [`QuantizationRule`](../qwix/_src/qconfig.md#QuantizationRule), [`module_path`](../qwix/_src/qconfig.md#QuantizationRule.module_path), [`PtqProvider`](../qwix/_src/providers/ptq.md#PtqProvider), [`quantize_params`](../qwix/_src/providers/ptq.md#quantize_params), [`QtProvider`](../qwix/_src/providers/qt.md#QtProvider), [`Transformer`](coverage_test.md#Transformer), [`CNN`](coverage_test.md#CNN), [`create_inputs`](coverage_test.md#Transformer.create_inputs)

### `Transformer`  ·  implements/extends Module
- def: [`integration_tests/coverage_test.py:49`](../../../../../raw/code/qwix/integration_tests/coverage_test.py#L49)
- doc: A simple Transformer model.
- signature: `class Transformer(nn.Module):`
- members:
  - `create_inputs(self)` — [`L75`](../../../../../raw/code/qwix/integration_tests/coverage_test.py#L75)
  - `embedding_dim` — [`L57`](../../../../../raw/code/qwix/integration_tests/coverage_test.py#L57)
  - `hidden_dim` — [`L55`](../../../../../raw/code/qwix/integration_tests/coverage_test.py#L55)
  - `num_heads` — [`L53`](../../../../../raw/code/qwix/integration_tests/coverage_test.py#L53)
  - `num_layers` — [`L52`](../../../../../raw/code/qwix/integration_tests/coverage_test.py#L52)
  - `qkv_features` — [`L54`](../../../../../raw/code/qwix/integration_tests/coverage_test.py#L54)
  - `vocab_size` — [`L56`](../../../../../raw/code/qwix/integration_tests/coverage_test.py#L56)
- protocol/private: `__call__`[`L60`](../../../../../raw/code/qwix/integration_tests/coverage_test.py#L60)
- used by: [`test_coverage`](coverage_test.md#CoverageTest.test_coverage)

