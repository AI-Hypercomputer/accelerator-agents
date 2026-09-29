---
title: 'Module: tests/contrib/sparsity/sparsity_module_test.py'
type: catalog
provenance: extracted
module: tests/contrib/sparsity/sparsity_module_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests.contrib.sparsity.sparsity_module_test`/SparsityQtTest#
symbols:
  SparsityQtTest.test_weight_sparsity: test_weight_sparsity().
  SparsityQtTest.test_eval_mode: test_eval_mode().
  SparsityQtTest.test_activation_sparsity: test_activation_sparsity().
  SparsityQtTest.test_no_sparsity: test_no_sparsity().
  SparsityQtTest: ''
---
# Module: [`tests/contrib/sparsity/sparsity_module_test.py`](../../../../../../../raw/code/qwix/tests/contrib/sparsity/sparsity_module_test.py)

## Classes
### `SparsityQtTest`  ·  implements/extends TestCase
- def: [`tests/contrib/sparsity/sparsity_module_test.py:23`](../../../../../../../raw/code/qwix/tests/contrib/sparsity/sparsity_module_test.py#L23)
- signature: `class SparsityQtTest(parameterized.TestCase):`
- members:
  - `test_activation_sparsity(self)` — [`L33`](../../../../../../../raw/code/qwix/tests/contrib/sparsity/sparsity_module_test.py#L33)
  - `test_eval_mode(self)` — [`L70`](../../../../../../../raw/code/qwix/tests/contrib/sparsity/sparsity_module_test.py#L70)
  - `test_no_sparsity(self)` — [`L25`](../../../../../../../raw/code/qwix/tests/contrib/sparsity/sparsity_module_test.py#L25)
  - `test_weight_sparsity(self)` — [`L48`](../../../../../../../raw/code/qwix/tests/contrib/sparsity/sparsity_module_test.py#L48)
- uses (calls/refs, reference-scoped): [`SparsityRule`](../../../qwix/_src/core/sparsity.md#SparsityRule), [`step`](../../../qwix/contrib/sparsity/sparsity_module.md#SparsityModule.step), [`weight_sparsity_m`](../../../qwix/_src/core/sparsity.md#SparsityRule.weight_sparsity_m), [`weight_sparsity_n`](../../../qwix/_src/core/sparsity.md#SparsityRule.weight_sparsity_n), [`SparsityModule`](../../../qwix/contrib/sparsity/sparsity_module.md#SparsityModule), [`activation_sparsity_m`](../../../qwix/_src/core/sparsity.md#SparsityRule.activation_sparsity_m), [`weight_sparsity_start_step`](../../../qwix/_src/core/sparsity.md#SparsityRule.weight_sparsity_start_step), [`activation_sparsity_n`](../../../qwix/_src/core/sparsity.md#SparsityRule.activation_sparsity_n), [`eval_mode`](../../../qwix/_src/core/sparsity.md#SparsityRule.eval_mode), [`weight_sparsity_update_step`](../../../qwix/_src/core/sparsity.md#SparsityRule.weight_sparsity_update_step)

