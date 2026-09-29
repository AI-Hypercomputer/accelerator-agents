---
title: 'Module: graphcast/data_utils_test.py'
type: catalog
provenance: extracted
module: graphcast/data_utils_test.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.data_utils_test`/DataUtilsTest#
symbols:
  DataUtilsTest.test_add_derived_vars_variables_added: test_add_derived_vars_variables_added().
  DataUtilsTest.test_add_derived_vars_existing_vars_not_overridden: test_add_derived_vars_existing_vars_not_overridden().
  DataUtilsTest.test_year_progress_is_zero_at_year_start_or_end: test_year_progress_is_zero_at_year_start_or_end().
  DataUtilsTest.test_year_progress_is_almost_one_before_year_ends: test_year_progress_is_almost_one_before_year_ends().
  DataUtilsTest.test_day_progress_is_zero_at_day_start_or_end: test_day_progress_is_zero_at_day_start_or_end().
  DataUtilsTest.test_add_tisr_var_variable_added: test_add_tisr_var_variable_added().
  DataUtilsTest.test_add_tisr_var_existing_var_not_overridden: test_add_tisr_var_existing_var_not_overridden().
  DataUtilsTest.test_add_tisr_var_works_with_batch_dim_size_one: test_add_tisr_var_works_with_batch_dim_size_one().
  DataUtilsTest.test_day_progress_computes_for_all_times_and_longitudes: test_day_progress_computes_for_all_times_and_longitudes().
  DataUtilsTest.test_day_progress_is_in_between_zero_and_one: test_day_progress_is_in_between_zero_and_one().
  DataUtilsTest.test_day_progress_specific_value: test_day_progress_specific_value().
  DataUtilsTest.test_featurize_progress_valid_values_and_dimensions: test_featurize_progress_valid_values_and_dimensions().
  DataUtilsTest.test_featurize_progress_invalid_dimensions: test_featurize_progress_invalid_dimensions().
  DataUtilsTest.test_add_derived_vars_missing_coordinate_raises_value_error: test_add_derived_vars_missing_coordinate_raises_value_error().
  DataUtilsTest.test_add_tisr_var_fails_with_batch_dim_size_greater_than_one: test_add_tisr_var_fails_with_batch_dim_size_greater_than_one().
  DataUtilsTest: ''
  DataUtilsTest.setUp: setUp().
---
# Module: [`graphcast/data_utils_test.py`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py)

## Classes
### `DataUtilsTest`  ·  implements/extends TestCase
- def: [`graphcast/data_utils_test.py:24`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L24)
- signature: `class DataUtilsTest(parameterized.TestCase):`
- members:
  - `setUp(self)` — [`L26`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L26)
  - `test_add_derived_vars_existing_vars_not_overridden(self)` — [`L182`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L182)
  - `test_add_derived_vars_missing_coordinate_raises_value_error(self, coord_name)` — [`L211`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L211)
  - `test_add_derived_vars_variables_added(self)` — [`L158`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L158)
  - `test_add_tisr_var_existing_var_not_overridden(self)` — [`L243`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L243)
  - `test_add_tisr_var_fails_with_batch_dim_size_greater_than_one(self)` — [`L286`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L286)
  - `test_add_tisr_var_variable_added(self)` — [`L224`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L224)
  - `test_add_tisr_var_works_with_batch_dim_size_one(self)` — [`L264`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L264)
  - `test_day_progress_computes_for_all_times_and_longitudes(self)` — [`L53`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L53)
  - `test_day_progress_is_in_between_zero_and_one(self, year, month, day, hour, minute, second)` — [`L82`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L82)
  - `test_day_progress_is_zero_at_day_start_or_end(self)` — [`L101`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L101)
  - `test_day_progress_specific_value(self)` — [`L112`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L112)
  - `test_featurize_progress_invalid_dimensions(self)` — [`L150`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L150)
  - `test_featurize_progress_valid_values_and_dimensions(self)` — [`L121`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L121)
  - `test_year_progress_is_almost_one_before_year_ends(self)` — [`L41`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L41)
  - `test_year_progress_is_zero_at_year_start_or_end(self)` — [`L31`](../../../../../raw/code/graphcast/graphcast/data_utils_test.py#L31)
- uses (calls/refs, reference-scoped): [`add_derived_vars`](data_utils.md#add_derived_vars), [`SEC_PER_DAY`](data_utils.md#SEC_PER_DAY), [`AVG_SEC_PER_YEAR`](data_utils.md#AVG_SEC_PER_YEAR), [`add_tisr_var`](data_utils.md#add_tisr_var), [`DAY_PROGRESS`](data_utils.md#DAY_PROGRESS), [`YEAR_PROGRESS`](data_utils.md#YEAR_PROGRESS), [`get_year_progress`](data_utils.md#get_year_progress), [`get_day_progress`](data_utils.md#get_day_progress), [`TISR`](data_utils.md#TISR), [`featurize_progress`](data_utils.md#featurize_progress)

