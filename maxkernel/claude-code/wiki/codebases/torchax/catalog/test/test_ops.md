---
title: 'Module: test/test_ops.py'
type: catalog
provenance: extracted
module: test/test_ops.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_ops`/
symbols:
  TestOpInfo.setUp: TestOpInfo#setUp().
  TestOpInfo.test_reference_eager: TestOpInfo#test_reference_eager().
  run_export_and_compare: run_export_and_compare().
  ops_to_test: ops_to_test.
  diff_output: diff_output().
  TestOpInfo.setUpClass: TestOpInfo#setUpClass().
  atol_dict: atol_dict.
  skiplist: skiplist.
  not_support_ops_list: not_support_ops_list.
  variant_test_name_to_skip: variant_test_name_to_skip.
  random_ops: random_ops.
  should_ignore_indexes: should_ignore_indexes.
  TestOpInfo: TestOpInfo#
  TestOpInfo.env: TestOpInfo#env.
  TestOpInfo.replace_values_below_threshold: TestOpInfo#replace_values_below_threshold().
---
# Module: [`test/test_ops.py`](../../../../../raw/code/torchax/test/test_ops.py)

## Classes
### `TestOpInfo`  ·  implements/extends TestCase
- def: [`test/test_ops.py:198`](../../../../../raw/code/torchax/test/test_ops.py#L198)
- signature: `class TestOpInfo(TestCase):`
- members:
  - `replace_values_below_threshold(self, torch_tensor, threshold)` — [`L212`](../../../../../raw/code/torchax/test/test_ops.py#L212)
  - `setUp(self)` — [`L203`](../../../../../raw/code/torchax/test/test_ops.py#L203)
  - `setUpClass(cls)` — [`L200`](../../../../../raw/code/torchax/test/test_ops.py#L200)
  - `test_reference_eager(self, device, dtype, op)` — [`L216`](../../../../../raw/code/torchax/test/test_ops.py#L216)
  - `env` — [`L204`](../../../../../raw/code/torchax/test/test_ops.py#L204)
- uses (calls/refs, reference-scoped): [`default_env`](../torchax/__init__.md#default_env), [`config`](../torchax/tensor.md#Environment.config), [`enable_accuracy_mode`](../torchax/__init__.md#enable_accuracy_mode), [`debug_print_each_op`](../torchax/config.md#Configuration.debug_print_each_op)  (4 test-only)

## Functions
- `diff_output(testcase, output1, output2, rtol, atol, equal_nan=True, check_output=True)` — [`L116`](../../../../../raw/code/torchax/test/test_ops.py#L116)
- `run_export_and_compare(testcase, func, sample_input, check_output=True, equal_nan=True, ignore_indices=False)` — [`L142`](../../../../../raw/code/torchax/test/test_ops.py#L142)

## Module values
- `atol_dict` — [`L102`](../../../../../raw/code/torchax/test/test_ops.py#L102)
- `not_support_ops_list` — [`L61`](../../../../../raw/code/torchax/test/test_ops.py#L61)
- `ops_to_test` — [`L183`](../../../../../raw/code/torchax/test/test_ops.py#L183)
- `random_ops` — [`L75`](../../../../../raw/code/torchax/test/test_ops.py#L75)
- `should_ignore_indexes` — [`L195`](../../../../../raw/code/torchax/test/test_ops.py#L195)
- `skiplist` — [`L29`](../../../../../raw/code/torchax/test/test_ops.py#L29)
- `variant_test_name_to_skip` — [`L71`](../../../../../raw/code/torchax/test/test_ops.py#L71)

