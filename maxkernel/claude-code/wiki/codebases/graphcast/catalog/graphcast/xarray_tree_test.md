---
title: 'Module: graphcast/xarray_tree_test.py'
type: catalog
provenance: extracted
module: graphcast/xarray_tree_test.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.xarray_tree_test`/
symbols:
  XarrayTreeTest.test_map_structure_maps_over_leaves_but_preserves_dataset_type: XarrayTreeTest#test_map_structure_maps_over_leaves_but_preserves_dataset_type().
  XarrayTreeTest.test_map_structure_on_dataset_plain_dict_when_coords_incompatible: XarrayTreeTest#test_map_structure_on_dataset_plain_dict_when_coords_incompatible().
  XarrayTreeTest.test_map_structure_on_dataset_drops_vars_with_none_return_values: XarrayTreeTest#test_map_structure_on_dataset_drops_vars_with_none_return_values().
  XarrayTreeTest.test_map_structure_on_dataset_returns_plain_dict_other_return_types: XarrayTreeTest#test_map_structure_on_dataset_returns_plain_dict_other_return_types().
  XarrayTreeTest.test_map_structure_two_args_different_variable_orders: XarrayTreeTest#test_map_structure_two_args_different_variable_orders().
  TEST_DATASET: TEST_DATASET.
  XarrayTreeTest.fn: XarrayTreeTest#fn().
  XarrayTreeTest.test_map_structure_on_data_arrays: XarrayTreeTest#test_map_structure_on_data_arrays().
  XarrayTreeTest: XarrayTreeTest#
---
# Module: [`graphcast/xarray_tree_test.py`](../../../../../raw/code/graphcast/graphcast/xarray_tree_test.py)

## Classes
### `XarrayTreeTest`  ·  implements/extends TestCase
- def: [`graphcast/xarray_tree_test.py:34`](../../../../../raw/code/graphcast/graphcast/xarray_tree_test.py#L34)
- signature: `class XarrayTreeTest(absltest.TestCase):`
- members:
  - `fn(leaf)` — [`L37`](../../../../../raw/code/graphcast/graphcast/xarray_tree_test.py#L37)
  - `test_map_structure_maps_over_leaves_but_preserves_dataset_type(self)` — [`L36`](../../../../../raw/code/graphcast/graphcast/xarray_tree_test.py#L36)
  - `test_map_structure_on_data_arrays(self)` — [`L49`](../../../../../raw/code/graphcast/graphcast/xarray_tree_test.py#L49)
  - `test_map_structure_on_dataset_drops_vars_with_none_return_values(self)` — [`L70`](../../../../../raw/code/graphcast/graphcast/xarray_tree_test.py#L70)
  - `test_map_structure_on_dataset_plain_dict_when_coords_incompatible(self)` — [`L55`](../../../../../raw/code/graphcast/graphcast/xarray_tree_test.py#L55)
  - `test_map_structure_on_dataset_returns_plain_dict_other_return_types(self)` — [`L78`](../../../../../raw/code/graphcast/graphcast/xarray_tree_test.py#L78)
  - `test_map_structure_two_args_different_variable_orders(self)` — [`L87`](../../../../../raw/code/graphcast/graphcast/xarray_tree_test.py#L87)
- uses (calls/refs, reference-scoped): [`map_structure`](xarray_tree.md#map_structure), [`TEST_DATASET`](xarray_tree_test.md#TEST_DATASET)

## Module values
- `TEST_DATASET` — [`L22`](../../../../../raw/code/graphcast/graphcast/xarray_tree_test.py#L22)

