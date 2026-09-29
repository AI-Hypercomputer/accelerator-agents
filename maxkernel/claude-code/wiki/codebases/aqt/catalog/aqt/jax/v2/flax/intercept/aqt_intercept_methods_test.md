---
title: 'Module: aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.flax.intercept.aqt_intercept_methods_test`/
symbols:
  MlpBlock: MlpBlock#
  MlpBlock.__call__: MlpBlock#__call__().
  NestedMlpBlock: NestedMlpBlock#
  NestedMlpBlock.__call__: NestedMlpBlock#__call__().
  MockDotGeneralGenerator: MockDotGeneralGenerator#
  MockDotGeneralGenerator.__init__: MockDotGeneralGenerator#__init__().
  MockDotGeneralGenerator.__call__: MockDotGeneralGenerator#__call__().
  MockDotGeneralGeneratorByModule: MockDotGeneralGeneratorByModule#
  MockDotGeneralGeneratorByModule.__init__: MockDotGeneralGeneratorByModule#__init__().
  MockDotGeneralGeneratorByModule.generate_by_module: MockDotGeneralGeneratorByModule#generate_by_module().
  AqtInterceptMethodsTest: AqtInterceptMethodsTest#
  AqtInterceptMethodsTest.test_intercept_methods_replace_dot_general_count: AqtInterceptMethodsTest#test_intercept_methods_replace_dot_general_count().
  AqtInterceptMethodsTest.test_aqt_dot_general_generator: AqtInterceptMethodsTest#test_aqt_dot_general_generator().
---
# Module: [`aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py)

## Classes
### `AqtInterceptMethodsTest`
- def: [`aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py:67`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py#L67)
- signature: `class AqtInterceptMethodsTest(parameterized.TestCase):`
- members:
  - `test_aqt_dot_general_generator(self)` — [`L104`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py#L104)
  - `test_intercept_methods_replace_dot_general_count(self, model_cls, dot_general_generator_cls, expected_count)` — [`L86`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py#L86)

### `MlpBlock`
- def: [`aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py:27`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py#L27)
- signature: `class MlpBlock(nn.Module):`
- protocol/private: `__call__`[`L29`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py#L29)

### `MockDotGeneralGenerator`
- def: [`aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py:45`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py#L45)
- signature: `class MockDotGeneralGenerator(aqt_intercept_methods.DotGeneralGenerator):`
- protocol/private: `__call__`[`L50`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py#L50), `__init__`[`L46`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py#L46)

### `MockDotGeneralGeneratorByModule`
- def: [`aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py:55`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py#L55)
- signature: `class MockDotGeneralGeneratorByModule(aqt_intercept_methods.DotGeneralGeneratorByModule):`
- members:
  - `generate_by_module(self, module: nn.Module)` — [`L61`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py#L61)
- protocol/private: `__init__`[`L57`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py#L57)

### `NestedMlpBlock`
- def: [`aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py:36`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py#L36)
- signature: `class NestedMlpBlock(nn.Module):`
- protocol/private: `__call__`[`L38`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods_test.py#L38)

