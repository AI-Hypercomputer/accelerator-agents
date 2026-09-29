---
title: 'Module: tests/_src/model_test.py'
type: catalog
provenance: extracted
module: tests/_src/model_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.model_test`/
symbols:
  ModelTest.test_quantize_linen_model: ModelTest#test_quantize_linen_model().
  CustomProvider: CustomProvider#
  NnModel.__call__: NnModel#__call__().
  CustomProvider.__init__: CustomProvider#__init__().
  CustomProvider.get_intercept_map: CustomProvider#get_intercept_map().
  NnModel: NnModel#
  NnModel.sin: NnModel#sin().
  CustomProvider._intercept_map: CustomProvider#_intercept_map.
  CustomProvider.process_model_output: CustomProvider#process_model_output().
  CustomProvider._initial_run_complete: CustomProvider#_initial_run_complete.
  ModelTest: ModelTest#
---
# Module: [`tests/_src/model_test.py`](../../../../../../raw/code/qwix/tests/_src/model_test.py)

## Classes
### `CustomProvider`  ·  implements/extends QuantizationProvider
- def: [`tests/_src/model_test.py:34`](../../../../../../raw/code/qwix/tests/_src/model_test.py#L34)
- signature: `class CustomProvider(qconfig.QuantizationProvider):`
- members:
  - `get_intercept_map(self)` — [`L45`](../../../../../../raw/code/qwix/tests/_src/model_test.py#L45)
  - `process_model_output(self, method_name: str, model_output: Any)` — [`L48`](../../../../../../raw/code/qwix/tests/_src/model_test.py#L48)
- protocol/private: `__init__`[`L36`](../../../../../../raw/code/qwix/tests/_src/model_test.py#L36), `_initial_run_complete`[`L49`](../../../../../../raw/code/qwix/tests/_src/model_test.py#L49), `_intercept_map`[`L40`](../../../../../../raw/code/qwix/tests/_src/model_test.py#L40)
- uses (calls/refs, reference-scoped): [`QuantizationProvider`](../../qwix/_src/qconfig.md#QuantizationProvider), [`__init__`](../../qwix/_src/qconfig.md#QuantizationProvider.__init__)
- used by: [`QuantizationProvider`](../../qwix/_src/qconfig.md#QuantizationProvider), [`get_intercept_map`](../../qwix/_src/qconfig.md#QuantizationProvider.get_intercept_map), [`process_model_output`](../../qwix/_src/qconfig.md#QuantizationProvider.process_model_output)  (1 test-only)

### `ModelTest`  ·  implements/extends TestCase
- def: [`tests/_src/model_test.py:53`](../../../../../../raw/code/qwix/tests/_src/model_test.py#L53)
- signature: `class ModelTest(absltest.TestCase):`
- members:
  - `test_quantize_linen_model(self)` — [`L55`](../../../../../../raw/code/qwix/tests/_src/model_test.py#L55)
- uses (calls/refs, reference-scoped): [`quantize_linen_model`](../../qwix/_src/model.md#quantize_linen_model)  (2 test-only)

### `NnModel`  ·  implements/extends Module
- def: [`tests/_src/model_test.py:25`](../../../../../../raw/code/qwix/tests/_src/model_test.py#L25)
- signature: `class NnModel(nn.Module):`
- members:
  - `sin(self, x)` — [`L30`](../../../../../../raw/code/qwix/tests/_src/model_test.py#L30)
- protocol/private: `__call__`[`L27`](../../../../../../raw/code/qwix/tests/_src/model_test.py#L27)
- used by: (1 test-only callers)

