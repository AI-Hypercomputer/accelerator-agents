---
title: 'Module: simply/utils/common_test.py'
type: catalog
provenance: extracted
module: simply/utils/common_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.common_test`/
symbols:
  RaggedArrayTest.test_ragged_array: RaggedArrayTest#test_ragged_array().
  RaggedArrayTest.test_nd_ragged: RaggedArrayTest#test_nd_ragged().
  CommonTest.test_annotated_array: CommonTest#test_annotated_array().
  RaggedArrayTest.test_ragged_array_invalid_lens: RaggedArrayTest#test_ragged_array_invalid_lens().
  RaggedArrayTest.test_ragged_concat: RaggedArrayTest#test_ragged_concat().
  CommonTest.test_parameterized_string: CommonTest#test_parameterized_string().
  RaggedArrayTest.test_padding_value: RaggedArrayTest#test_padding_value().
  RaggedArrayTest.test_keep_rows: RaggedArrayTest#test_keep_rows().
  RaggedArrayTest.test_keep_last_ncols: RaggedArrayTest#test_keep_last_ncols().
  RaggedArrayTest.test_when: RaggedArrayTest#test_when().
  RaggedArrayTest._fn: RaggedArrayTest#_fn().
  CommonTest.test_simple_quantize: CommonTest#test_simple_quantize().
  CommonTest.test_quantization_calculation: CommonTest#test_quantization_calculation().
  CommonTest.test_find_unused_argpaths: CommonTest#test_find_unused_argpaths().
  CommonTest.test_sorted_with_indices: CommonTest#test_sorted_with_indices().
  RaggedArrayTest.test_to_padded_dense: RaggedArrayTest#test_to_padded_dense().
  RaggedArrayTest._double: RaggedArrayTest#_double().
  CommonTest.test_convert_array_with_abstract: CommonTest#test_convert_array_with_abstract().
  CommonTest.test_neg_inf: CommonTest#test_neg_inf().
  CommonTest.test_reduce_same: CommonTest#test_reduce_same().
  CommonTest._func: CommonTest#_func().
  CommonTest: CommonTest#
  RaggedArrayTest: RaggedArrayTest#
---
# Module: [`simply/utils/common_test.py`](../../../../../../raw/code/simply/simply/utils/common_test.py)

## Classes
### `CommonTest`  ·  implements/extends TestCase
- def: [`simply/utils/common_test.py:22`](../../../../../../raw/code/simply/simply/utils/common_test.py#L22)
- signature: `class CommonTest(absltest.TestCase):`
- members:
  - `test_annotated_array(self)` — [`L76`](../../../../../../raw/code/simply/simply/utils/common_test.py#L76)
  - `test_convert_array_with_abstract(self)` — [`L131`](../../../../../../raw/code/simply/simply/utils/common_test.py#L131)
  - `test_find_unused_argpaths(self)` — [`L91`](../../../../../../raw/code/simply/simply/utils/common_test.py#L91)
  - `test_neg_inf(self)` — [`L153`](../../../../../../raw/code/simply/simply/utils/common_test.py#L153)
  - `test_parameterized_string(self)` — [`L24`](../../../../../../raw/code/simply/simply/utils/common_test.py#L24)
  - `test_quantization_calculation(self)` — [`L56`](../../../../../../raw/code/simply/simply/utils/common_test.py#L56)
  - `test_reduce_same(self)` — [`L159`](../../../../../../raw/code/simply/simply/utils/common_test.py#L159)
  - `test_simple_quantize(self)` — [`L33`](../../../../../../raw/code/simply/simply/utils/common_test.py#L33)
  - `test_sorted_with_indices(self)` — [`L117`](../../../../../../raw/code/simply/simply/utils/common_test.py#L117)
- protocol/private: `_func`[`L92`](../../../../../../raw/code/simply/simply/utils/common_test.py#L92)
- uses (calls/refs, reference-scoped): [`convert_or_dequantize`](common.md#convert_or_dequantize), [`AnnotatedArray`](common.md#AnnotatedArray), [`create`](common.md#AnnotatedArray.create), [`neg_inf`](common.md#neg_inf), [`transfer_metadata`](common.md#transfer_metadata), [`array`](common.md#AnnotatedArray.array), [`available_parameters`](common.md#ParameterizedString.available_parameters), [`quantize_array`](common.md#quantize_array), [`unsorted`](common.md#unsorted), [`sorted_with_indices`](common.md#sorted_with_indices), [`metadata`](common.md#AnnotatedArray.metadata), [`reduce_same`](common.md#reduce_same), [`find_unused_argpaths`](common.md#find_unused_argpaths), [`convert_array_with_abstract`](common.md#convert_array_with_abstract), [`format`](common.md#ParameterizedString.format), [`ParameterizedString`](common.md#ParameterizedString)

### `RaggedArrayTest`  ·  implements/extends TestCase
- def: [`simply/utils/common_test.py:167`](../../../../../../raw/code/simply/simply/utils/common_test.py#L167)
- signature: `class RaggedArrayTest(absltest.TestCase):`
- members:
  - `test_keep_last_ncols(self)` — [`L282`](../../../../../../raw/code/simply/simply/utils/common_test.py#L282)
  - `test_keep_rows(self)` — [`L271`](../../../../../../raw/code/simply/simply/utils/common_test.py#L271)
  - `test_nd_ragged(self)` — [`L217`](../../../../../../raw/code/simply/simply/utils/common_test.py#L217)
  - `test_padding_value(self)` — [`L255`](../../../../../../raw/code/simply/simply/utils/common_test.py#L255)
  - `test_ragged_array(self)` — [`L169`](../../../../../../raw/code/simply/simply/utils/common_test.py#L169)
  - `test_ragged_array_invalid_lens(self)` — [`L195`](../../../../../../raw/code/simply/simply/utils/common_test.py#L195)
  - `test_ragged_concat(self)` — [`L199`](../../../../../../raw/code/simply/simply/utils/common_test.py#L199)
  - `test_to_padded_dense(self)` — [`L262`](../../../../../../raw/code/simply/simply/utils/common_test.py#L262)
  - `test_when(self)` — [`L293`](../../../../../../raw/code/simply/simply/utils/common_test.py#L293)
- protocol/private: `_double`[`L295`](../../../../../../raw/code/simply/simply/utils/common_test.py#L295), `_fn`[`L302`](../../../../../../raw/code/simply/simply/utils/common_test.py#L302)
- uses (calls/refs, reference-scoped): [`data`](common.md#RaggedArray.data), [`lens`](common.md#RaggedArray.lens), [`row_ids`](common.md#RaggedArray.row_ids), [`concat`](common.md#RaggedArray.concat), [`capacity`](common.md#RaggedArray.capacity), [`RaggedArray`](common.md#RaggedArray), [`intra_offset`](common.md#RaggedArray.intra_offset), [`keep_rows`](common.md#RaggedArray.keep_rows), [`keep_last_ncols`](common.md#RaggedArray.keep_last_ncols), [`batch_size`](common.md#RaggedArray.batch_size), [`set_padding_value`](common.md#RaggedArray.set_padding_value), [`total_length`](common.md#RaggedArray.total_length), [`extend_capacity_to`](common.md#RaggedArray.extend_capacity_to), [`from_numpy_list`](common.md#RaggedArray.from_numpy_list), [`to_numpy_list`](common.md#RaggedArray.to_numpy_list), [`to_padded_dense`](common.md#RaggedArray.to_padded_dense), [`dtype`](common.md#RaggedArray.dtype), [`is_valid`](common.md#RaggedArray.is_valid), [`when`](common.md#when)

