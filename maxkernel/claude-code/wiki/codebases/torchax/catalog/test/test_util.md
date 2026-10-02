---
title: 'Module: test/test_util.py'
type: catalog
provenance: extracted
module: test/test_util.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_util`/
symbols:
  TestListUtils.test_partition_empty_list: TestListUtils#test_partition_empty_list().
  TestListUtils.test_partition_even_odd: TestListUtils#test_partition_even_odd().
  TestListUtils.test_partition_all_true: TestListUtils#test_partition_all_true().
  TestListUtils.test_partition_all_false: TestListUtils#test_partition_all_false().
  TestListUtils.test_partition_mixed_types: TestListUtils#test_partition_mixed_types().
  is_even: is_even().
  TestListUtils.test_partition_with_lambda: TestListUtils#test_partition_with_lambda().
  TestListUtils.test_merge_empty_lists: TestListUtils#test_merge_empty_lists().
  TestListUtils.test_merge_basic: TestListUtils#test_merge_basic().
  TestListUtils.test_merge_no_none_in_list1: TestListUtils#test_merge_no_none_in_list1().
  TestListUtils.test_merge_all_none_in_list1: TestListUtils#test_merge_all_none_in_list1().
  TestListUtils.test_merge_mixed_types: TestListUtils#test_merge_mixed_types().
  TestListUtils.test_merge_unequal_lengths: TestListUtils#test_merge_unequal_lengths().
  is_string: is_string().
  is_positive: is_positive().
  TestListUtils: TestListUtils#
---
# Module: [`test/test_util.py`](../../../../../raw/code/torchax/test/test_util.py)

## Classes
### `TestListUtils`  ·  implements/extends TestCase
- def: [`test/test_util.py:33`](../../../../../raw/code/torchax/test/test_util.py#L33)
- signature: `class TestListUtils(unittest.TestCase):`
- members:
  - `test_merge_all_none_in_list1(self)` — [`L99`](../../../../../raw/code/torchax/test/test_util.py#L99) — Test merge when list1 contains only None.
  - `test_merge_basic(self)` — [`L85`](../../../../../raw/code/torchax/test/test_util.py#L85) — Test basic merging with None values in the first list.
  - `test_merge_empty_lists(self)` — [`L81`](../../../../../raw/code/torchax/test/test_util.py#L81) — Test merge with empty lists.
  - `test_merge_mixed_types(self)` — [`L106`](../../../../../raw/code/torchax/test/test_util.py#L106) — Test merge with mixed data types.
  - `test_merge_no_none_in_list1(self)` — [`L92`](../../../../../raw/code/torchax/test/test_util.py#L92) — Test merge when list1 has no None values.
  - `test_merge_unequal_lengths(self)` — [`L113`](../../../../../raw/code/torchax/test/test_util.py#L113) — Test that merge raises AssertionError for lists of unequal length.
  - `test_partition_all_false(self)` — [`L54`](../../../../../raw/code/torchax/test/test_util.py#L54) — Test partition when the predicate is always false.
  - `test_partition_all_true(self)` — [`L47`](../../../../../raw/code/torchax/test/test_util.py#L47) — Test partition when the predicate is always true.
  - `test_partition_empty_list(self)` — [`L36`](../../../../../raw/code/torchax/test/test_util.py#L36) — Test partition with an empty list.
  - `test_partition_even_odd(self)` — [`L40`](../../../../../raw/code/torchax/test/test_util.py#L40) — Test partitioning numbers into even and odd.
  - `test_partition_mixed_types(self)` — [`L61`](../../../../../raw/code/torchax/test/test_util.py#L61) — Test partition with a list of mixed types.
  - `test_partition_with_lambda(self)` — [`L69`](../../../../../raw/code/torchax/test/test_util.py#L69) — Test partition using a lambda function as the predicate.
- uses (calls/refs, reference-scoped): [`merge`](../torchax/util.md#merge), [`partition`](../torchax/util.md#partition)  (2 test-only)

## Functions
- `is_even(n)` — [`L21`](../../../../../raw/code/torchax/test/test_util.py#L21)
- `is_positive(n)` — [`L25`](../../../../../raw/code/torchax/test/test_util.py#L25)
- `is_string(s)` — [`L29`](../../../../../raw/code/torchax/test/test_util.py#L29)

