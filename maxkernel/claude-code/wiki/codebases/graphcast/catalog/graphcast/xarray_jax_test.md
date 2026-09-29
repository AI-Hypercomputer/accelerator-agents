---
title: 'Module: graphcast/xarray_jax_test.py'
type: catalog
provenance: extracted
module: graphcast/xarray_jax_test.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.xarray_jax_test`/XarrayJaxTest#
symbols:
  XarrayJaxTest.ops_via_xarray: ops_via_xarray().
  XarrayJaxTest.func: func().
  XarrayJaxTest.fn: fn().
  XarrayJaxTest.test_jit_function_with_xarray_datatree_arguments_and_return: test_jit_function_with_xarray_datatree_arguments_and_return().
  XarrayJaxTest.test_flatten_unflatten_datatree: test_flatten_unflatten_datatree().
  XarrayJaxTest.test_pmap_with_jax_coords: test_pmap_with_jax_coords().
  XarrayJaxTest.test_assign_coords_arg_types: test_assign_coords_arg_types().
  XarrayJaxTest.test_jit_function_with_data_array_and_jax_coords: test_jit_function_with_data_array_and_jax_coords().
  XarrayJaxTest.test_jit_function_with_dataset_and_jax_coords: test_jit_function_with_dataset_and_jax_coords().
  XarrayJaxTest.test_flatten_unflatten_variable: test_flatten_unflatten_variable().
  XarrayJaxTest.test_flatten_unflatten_data_array: test_flatten_unflatten_data_array().
  XarrayJaxTest.test_flatten_unflatten_dataset: test_flatten_unflatten_dataset().
  XarrayJaxTest.test_pmap: test_pmap().
  XarrayJaxTest.test_pmap_with_tree_mix_of_xarray_and_jax_array: test_pmap_with_tree_mix_of_xarray_and_jax_array().
  XarrayJaxTest.test_apply_ufunc_multiple_return_values: test_apply_ufunc_multiple_return_values().
  XarrayJaxTest.test_scan: test_scan().
  XarrayJaxTest.test_scan_no_inputs: test_scan_no_inputs().
  XarrayJaxTest.test_grad_function_with_xarray_variable_arguments: test_grad_function_with_xarray_variable_arguments().
  XarrayJaxTest.test_flatten_unflatten_added_dim: test_flatten_unflatten_added_dim().
  XarrayJaxTest.test_map_added_dim: test_map_added_dim().
  XarrayJaxTest.test_map_remove_dim: test_map_remove_dim().
  XarrayJaxTest.test_pmap_complains_when_dim_not_first: test_pmap_complains_when_dim_not_first().
  XarrayJaxTest.test_apply_ufunc: test_apply_ufunc().
  XarrayJaxTest.f: f().
  XarrayJaxTest.test_jax_array_wrapper_with_numpy_api: test_jax_array_wrapper_with_numpy_api().
  XarrayJaxTest.test_jax_xarray_variable: test_jax_xarray_variable().
  XarrayJaxTest.test_jax_xarray_data_array: test_jax_xarray_data_array().
  XarrayJaxTest.test_jax_xarray_dataset: test_jax_xarray_dataset().
  XarrayJaxTest.test_jit_function_with_xarray_variable_arguments_and_return: test_jit_function_with_xarray_variable_arguments_and_return().
  XarrayJaxTest.test_jit_ahead_of_time_compile_with_xarray: test_jit_ahead_of_time_compile_with_xarray().
  XarrayJaxTest.test_jit_problem_if_convert_to_plain_numpy_array: test_jit_problem_if_convert_to_plain_numpy_array().
  XarrayJaxTest.test_jit_function_with_xarray_data_array_arguments_and_return: test_jit_function_with_xarray_data_array_arguments_and_return().
  XarrayJaxTest.test_jit_function_with_xarray_dataset_arguments_and_return: test_jit_function_with_xarray_dataset_arguments_and_return().
  XarrayJaxTest.test_eval_shape_with_xarray: test_eval_shape_with_xarray().
  XarrayJaxTest.ufunc: ufunc().
  XarrayJaxTest: ''
---
# Module: [`graphcast/xarray_jax_test.py`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py)

## Classes
### `XarrayJaxTest`  ·  implements/extends TestCase
- def: [`graphcast/xarray_jax_test.py:24`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L24)
- signature: `class XarrayJaxTest(absltest.TestCase):`
- members:
  - `f(carry, x)` — [`L600`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L600)
  - `fn(v)` — [`L190`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L190)
  - `func(d)` — [`L448`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L448)
  - `ops_via_xarray(inputs)` — [`L43`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L43)
  - `test_apply_ufunc(self)` — [`L546`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L546)
  - `test_apply_ufunc_multiple_return_values(self)` — [`L562`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L562)
  - `test_assign_coords_arg_types(self)` — [`L656`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L656)
  - `test_eval_shape_with_xarray(self)` — [`L589`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L589)
  - `test_flatten_unflatten_added_dim(self)` — [`L389`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L389)
  - `test_flatten_unflatten_data_array(self)` — [`L334`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L334)
  - `test_flatten_unflatten_dataset(self)` — [`L348`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L348)
  - `test_flatten_unflatten_datatree(self)` — [`L366`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L366)
  - `test_flatten_unflatten_variable(self)` — [`L324`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L324)
  - `test_grad_function_with_xarray_variable_arguments(self)` — [`L161`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L161)
  - `test_jax_array_wrapper_with_numpy_api(self)` — [`L26`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L26)
  - `test_jax_xarray_data_array(self)` — [`L65`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L65)
  - `test_jax_xarray_dataset(self)` — [`L91`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L91)
  - `test_jax_xarray_variable(self)` — [`L42`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L42)
  - `test_jit_ahead_of_time_compile_with_xarray(self)` — [`L144`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L144)
  - `test_jit_function_with_data_array_and_jax_coords(self)` — [`L178`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L178)
  - `test_jit_function_with_dataset_and_jax_coords(self)` — [`L269`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L269)
  - `test_jit_function_with_xarray_data_array_arguments_and_return(self)` — [`L166`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L166)
  - `test_jit_function_with_xarray_dataset_arguments_and_return(self)` — [`L221`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L221)
  - `test_jit_function_with_xarray_datatree_arguments_and_return(self)` — [`L239`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L239)
  - `test_jit_function_with_xarray_variable_arguments_and_return(self)` — [`L123`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L123)
  - `test_jit_problem_if_convert_to_plain_numpy_array(self)` — [`L154`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L154)
  - `test_map_added_dim(self)` — [`L404`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L404)
  - `test_map_remove_dim(self)` — [`L418`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L418)
  - `test_pmap(self)` — [`L440`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L440)
  - `test_pmap_complains_when_dim_not_first(self)` — [`L533`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L533)
  - `test_pmap_with_jax_coords(self)` — [`L466`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L466)
  - `test_pmap_with_tree_mix_of_xarray_and_jax_array(self)` — [`L512`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L512)
  - `test_scan(self)` — [`L599`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L599)
  - `test_scan_no_inputs(self)` — [`L639`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L639)
  - `ufunc(array)` — [`L563`](../../../../../raw/code/graphcast/graphcast/xarray_jax_test.py#L563)
- uses (calls/refs, reference-scoped): [`DataArray`](xarray_jax.md#DataArray), [`Dataset`](xarray_jax.md#Dataset), [`Variable`](xarray_jax.md#Variable), [`scan`](xarray_jax.md#scan), [`assign_coords`](xarray_jax.md#assign_coords), [`JaxArrayWrapper`](xarray_jax.md#JaxArrayWrapper), [`_unflatten_data_array`](xarray_jax.md#_unflatten_data_array), [`_unflatten_dataset`](xarray_jax.md#_unflatten_dataset), [`dims_change_on_unflatten`](xarray_jax.md#dims_change_on_unflatten), [`assign_jax_coords`](xarray_jax.md#assign_jax_coords), [`_unflatten_variable`](xarray_jax.md#_unflatten_variable), [`_flatten_data_array`](xarray_jax.md#_flatten_data_array), [`_flatten_dataset`](xarray_jax.md#_flatten_dataset), [`jax_data`](xarray_jax.md#jax_data), [`pmap`](xarray_jax.md#pmap), [`_flatten_variable`](xarray_jax.md#_flatten_variable), [`apply_ufunc`](xarray_jax.md#apply_ufunc), [`_flatten_datatree`](xarray_jax.md#_flatten_datatree), [`_unflatten_datatree`](xarray_jax.md#_unflatten_datatree)

