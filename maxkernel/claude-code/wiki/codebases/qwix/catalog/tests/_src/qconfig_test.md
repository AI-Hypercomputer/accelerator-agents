---
title: 'Module: tests/_src/qconfig_test.py'
type: catalog
provenance: extracted
module: tests/_src/qconfig_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.qconfig_test`/
symbols:
  QconfigTest.test_some_rules_unused: QconfigTest#test_some_rules_unused().
  QconfigTest.test_all_rules_used: QconfigTest#test_all_rules_used().
  QconfigTest.test_get_unused_rules_before_quantize_model: QconfigTest#test_get_unused_rules_before_quantize_model().
  QconfigTest.test_invalid_rule_static_scale_without_act_qtype: QconfigTest#test_invalid_rule_static_scale_without_act_qtype().
  QconfigTest.setUp.MyModel.__call__: QconfigTest#setUp().MyModel#__call__().
  QconfigTest.model: QconfigTest#model.
  QconfigTest.x: QconfigTest#x.
  QconfigTest.setUp.MyModel: QconfigTest#setUp().MyModel#
  MyModel.lin1: MyModel#lin1.
  MyModel.lin2: MyModel#lin2.
  MyModel.layers: MyModel#layers.
  QconfigTest: QconfigTest#
  QconfigTest.setUp: QconfigTest#setUp().
  QconfigTest.setUp.MyModel.__init__: QconfigTest#setUp().MyModel#__init__().
---
# Module: [`tests/_src/qconfig_test.py`](../../../../../../raw/code/qwix/tests/_src/qconfig_test.py)

## Classes
### `MyModel`  ·  implements/extends Module
- def: [`tests/_src/qconfig_test.py:29`](../../../../../../raw/code/qwix/tests/_src/qconfig_test.py#L29)
- signature: `class MyModel(nnx.Module):`
- members:
  - `layers` — [`L34`](../../../../../../raw/code/qwix/tests/_src/qconfig_test.py#L34)
  - `lin1` — [`L32`](../../../../../../raw/code/qwix/tests/_src/qconfig_test.py#L32)
  - `lin2` — [`L33`](../../../../../../raw/code/qwix/tests/_src/qconfig_test.py#L33)
- protocol/private: `__call__`[`L38`](../../../../../../raw/code/qwix/tests/_src/qconfig_test.py#L38), `__init__`[`L31`](../../../../../../raw/code/qwix/tests/_src/qconfig_test.py#L31)
- used by: (1 test-only callers)

### `QconfigTest`  ·  implements/extends TestCase
- def: [`tests/_src/qconfig_test.py:23`](../../../../../../raw/code/qwix/tests/_src/qconfig_test.py#L23)
- signature: `class QconfigTest(absltest.TestCase):`
- members:
  - `setUp(self)` — [`L25`](../../../../../../raw/code/qwix/tests/_src/qconfig_test.py#L25)
  - `test_all_rules_used(self)` — [`L44`](../../../../../../raw/code/qwix/tests/_src/qconfig_test.py#L44)
  - `test_get_unused_rules_before_quantize_model(self)` — [`L95`](../../../../../../raw/code/qwix/tests/_src/qconfig_test.py#L95)
  - `test_invalid_rule_static_scale_without_act_qtype(self)` — [`L115`](../../../../../../raw/code/qwix/tests/_src/qconfig_test.py#L115)
  - `test_some_rules_unused(self)` — [`L64`](../../../../../../raw/code/qwix/tests/_src/qconfig_test.py#L64)
  - `model` — [`L41`](../../../../../../raw/code/qwix/tests/_src/qconfig_test.py#L41)
  - `x` — [`L42`](../../../../../../raw/code/qwix/tests/_src/qconfig_test.py#L42)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../../qwix/_src/model.md#quantize_model), [`QuantizationRule`](../../qwix/_src/qconfig.md#QuantizationRule), [`QArray`](../../qwix/_src/core/qarray.md#QArray), [`act_qtype`](../../qwix/_src/qconfig.md#QuantizationRule.act_qtype), [`module_path`](../../qwix/_src/qconfig.md#QuantizationRule.module_path), [`PtqProvider`](../../qwix/_src/providers/ptq.md#PtqProvider), [`act_static_scale`](../../qwix/_src/qconfig.md#QuantizationRule.act_static_scale), [`get_unused_rules`](../../qwix/_src/qconfig.md#QuantizationProvider.get_unused_rules)  (1 test-only)

