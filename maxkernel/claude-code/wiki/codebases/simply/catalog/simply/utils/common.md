---
title: 'Module: simply/utils/common.py'
type: catalog
provenance: extracted
module: simply/utils/common.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.common`/
symbols:
  PyTree.PyTree: PyTree.PyTree.
  RaggedArray.data: RaggedArray#data.
  RaggedArray.lens: RaggedArray#lens.
  RaggedArray.row_ids: RaggedArray#row_ids().
  RaggedArray.concat: RaggedArray#concat().
  Array.Array: Array.Array.
  RaggedArray.capacity: RaggedArray#capacity().
  get_raw_arrays: get_raw_arrays().
  RaggedArray: RaggedArray#
  convert_or_dequantize: convert_or_dequantize().
  RaggedArray.intra_offset: RaggedArray#intra_offset().
  RaggedArray.keep_rows: RaggedArray#keep_rows().
  RaggedArray.keep_last_ncols: RaggedArray#keep_last_ncols().
  RaggedArray.total_length: RaggedArray#total_length().
  RaggedArray.batch_size: RaggedArray#batch_size().
  RaggedArray.set_padding_value: RaggedArray#set_padding_value().
  AnnotatedArray: AnnotatedArray#
  AnnotatedArray.create: AnnotatedArray#create().
  transfer_metadata._transfer_metadata: transfer_metadata()._transfer_metadata().
  RaggedArray.from_numpy_list: RaggedArray#from_numpy_list().
  RaggedArray.extend_capacity_to: RaggedArray#extend_capacity_to().
  RaggedArray.to_numpy_list: RaggedArray#to_numpy_list().
  neg_inf: neg_inf().
  eval_abstract_output: eval_abstract_output().
  PartitionAnnotation.PartitionAnnotation: PartitionAnnotation.PartitionAnnotation.
  transfer_metadata: transfer_metadata().
  RaggedArray.to_padded_dense: RaggedArray#to_padded_dense().
  AnnotatedArray.array: AnnotatedArray#array.
  RaggedArray.dtype: RaggedArray#dtype().
  ParameterizedString.available_parameters: ParameterizedString#available_parameters().
  RaggedArray.row: RaggedArray#row().
  ParameterizedString.__iter__: ParameterizedString#__iter__().
  RaggedArray.subshape: RaggedArray#subshape().
  quantize_array: quantize_array().
  named_jit: named_jit().
  RaggedArray.row_starts_with_end: RaggedArray#row_starts_with_end().
  unsorted: unsorted().
  RaggedArray.is_valid: RaggedArray#is_valid().
  sorted_with_indices: sorted_with_indices().
  when: when().
  RawT: RawT.
  AnnotatedArray.metadata: AnnotatedArray#metadata.
  AnnotatedArray.tree_flatten: AnnotatedArray#tree_flatten().
  AnnotatedArray.tree_unflatten: AnnotatedArray#tree_unflatten().
  ParameterizedString.__post_init__: ParameterizedString#__post_init__().
  reduce_same: reduce_same().
  _TypeVarT._TypeVarT: _TypeVarT._TypeVarT.
  find_unused_argpaths: find_unused_argpaths().
  RaggedArray.row_starts: RaggedArray#row_starts().
  convert_array_with_abstract: convert_array_with_abstract().
  ParameterizedString.parameter_names: ParameterizedString#parameter_names().
  ParameterizedString.format: ParameterizedString#format().
  when._outer_wrapper: when()._outer_wrapper().
  when._outer_wrapper._inner_wrapper: when()._outer_wrapper()._inner_wrapper().
  ParameterizedString.parameters: ParameterizedString#parameters.
  AnnotatedArray.dim_annotation: AnnotatedArray#dim_annotation().
  AnnotatedArray.shape: AnnotatedArray#shape().
  AnnotatedArray.dtype: AnnotatedArray#dtype().
  RaggedArray.__post_init__: RaggedArray#__post_init__().
  RaggedArray.ndim: RaggedArray#ndim().
  ParameterizedString.template: ParameterizedString#template.
  named_partial_fn: named_partial_fn().
  pad_to_len: pad_to_len().
  when._as_is: when()._as_is().
  AttributeDict: AttributeDict#
  ParameterizedString: ParameterizedString#
  ParameterizedString.PARAMETER_RE: ParameterizedString#PARAMETER_RE.
  convert_rows_to_columns: convert_rows_to_columns().
  round_up_to_base: round_up_to_base().
  BasicType.BasicType: BasicType.BasicType.
  CacheValue: CacheValue.
  THREAD_CONTEXT: THREAD_CONTEXT.
  AttributeDict.__slots__: AttributeDict#__slots__.
  AttributeDict.__setattr__: AttributeDict#__setattr__.
  AttributeDict.__getattr__: AttributeDict#__getattr__().
  convert_columns_to_rows: convert_columns_to_rows().
---
# Module: [`simply/utils/common.py`](../../../../../../raw/code/simply/simply/utils/common.py)

## Classes
### `AnnotatedArray`
- def: [`simply/utils/common.py:61`](../../../../../../raw/code/simply/simply/utils/common.py#L61)
- doc: A wrapper around Array to annotate its metadata.
- signature: `class AnnotatedArray:`
- members:
  - `create(cls, array: Array, **kwargs)` — [`L86`](../../../../../../raw/code/simply/simply/utils/common.py#L86)
  - `dim_annotation(self)` — [`L67`](../../../../../../raw/code/simply/simply/utils/common.py#L67)
  - `dtype(self)` — [`L75`](../../../../../../raw/code/simply/simply/utils/common.py#L75)
  - `shape(self)` — [`L71`](../../../../../../raw/code/simply/simply/utils/common.py#L71)
  - `tree_flatten(self)` — [`L78`](../../../../../../raw/code/simply/simply/utils/common.py#L78)
  - `tree_unflatten(cls, aux_data, children)` — [`L82`](../../../../../../raw/code/simply/simply/utils/common.py#L82)
  - `array` — [`L63`](../../../../../../raw/code/simply/simply/utils/common.py#L63)
  - `metadata` — [`L64`](../../../../../../raw/code/simply/simply/utils/common.py#L64)
- uses (calls/refs, reference-scoped): [`Array`](common.md#Array.Array)
- used by: [`init`](module.md#EinsumLinear.init), [`init`](../model_lib.md#LayerNorm.init), [`init`](vision_encoder.md#PatchEncoder.init), [`get_raw_arrays`](common.md#get_raw_arrays), [`init`](../model_lib.md#PerDimScale.init), [`_transfer_metadata`](common.md#transfer_metadata._transfer_metadata), [`test_annotated_array`](common_test.md#CommonTest.test_annotated_array), [`AnnotatedArray`](vision_encoder.md#AnnotatedArray), [`transfer_metadata`](common.md#transfer_metadata), [`test_dim_annotation`](../model_lib_test.md#ModelLibTest.test_dim_annotation), [`AnnotatedArray`](../model_lib.md#AnnotatedArray), [`AnnotatedArray`](module.md#AnnotatedArray), [`AnnotatedArray`](optimizers.md#AnnotatedArray)

### `AttributeDict`  ·  implements/extends dict
- def: [`simply/utils/common.py:118`](../../../../../../raw/code/simply/simply/utils/common.py#L118)
- doc: A simplfied version of ConfigDict.
- signature: `class AttributeDict(dict):`
- protocol/private: `__getattr__`[`L124`](../../../../../../raw/code/simply/simply/utils/common.py#L124), `__setattr__`[`L122`](../../../../../../raw/code/simply/simply/utils/common.py#L122), `__slots__`[`L121`](../../../../../../raw/code/simply/simply/utils/common.py#L121)
- used by: [`execute_code_patch`](../main.md#execute_code_patch)

### `ParameterizedString`
- def: [`simply/utils/common.py:131`](../../../../../../raw/code/simply/simply/utils/common.py#L131)
- doc: Parameterized string.
- signature: `class ParameterizedString:`
- members:
  - `available_parameters(self)` — [`L162`](../../../../../../raw/code/simply/simply/utils/common.py#L162)
  - `format(self, **kwargs: str)` — [`L165`](../../../../../../raw/code/simply/simply/utils/common.py#L165)
  - `parameter_names(cls, template: str)` — [`L158`](../../../../../../raw/code/simply/simply/utils/common.py#L158)
  - `PARAMETER_RE` — [`L144`](../../../../../../raw/code/simply/simply/utils/common.py#L144)
  - `parameters` — [`L147`](../../../../../../raw/code/simply/simply/utils/common.py#L147)
  - `template` — [`L146`](../../../../../../raw/code/simply/simply/utils/common.py#L146)
- protocol/private: `__iter__`[`L168`](../../../../../../raw/code/simply/simply/utils/common.py#L168), `__post_init__`[`L149`](../../../../../../raw/code/simply/simply/utils/common.py#L149)
- used by: [`test_parameterized_string`](common_test.md#CommonTest.test_parameterized_string)

### `RaggedArray`
- def: [`simply/utils/common.py:321`](../../../../../../raw/code/simply/simply/utils/common.py#L321) — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
- doc: A ragged 2d array.
- signature: `class RaggedArray:`
- members:
  - `batch_size(self)` — [`L357`](../../../../../../raw/code/simply/simply/utils/common.py#L357)
  - `capacity(self)` — [`L349`](../../../../../../raw/code/simply/simply/utils/common.py#L349)
  - `concat(self, other: Self, capacity: int | None = None)` — [`L456`](../../../../../../raw/code/simply/simply/utils/common.py#L456) — Concatenates with another ragged array. — documented in [simply-utils-common](../../../concepts/simply-utils-common.md)
  - `dtype(self)` — [`L353`](../../../../../../raw/code/simply/simply/utils/common.py#L353)
  - `extend_capacity_to(self, capacity: int)` — [`L446`](../../../../../../raw/code/simply/simply/utils/common.py#L446) — documented in [simply-utils-common](../../../concepts/simply-utils-common.md)
  - `from_numpy_list(cls, np_list: Sequence[np.typing.ArrayLike])` — [`L433`](../../../../../../raw/code/simply/simply/utils/common.py#L433) — documented in [simply-utils-common](../../../concepts/simply-utils-common.md)
  - `intra_offset(self)` — [`L388`](../../../../../../raw/code/simply/simply/utils/common.py#L388)
  - `is_valid(self)` — [`L332`](../../../../../../raw/code/simply/simply/utils/common.py#L332)
  - `keep_last_ncols(self, ncols: int)` — [`L508`](../../../../../../raw/code/simply/simply/utils/common.py#L508) — Keeps the last n columns of each row.
  - `keep_rows(self, row_mask: jax.typing.ArrayLike)` — [`L489`](../../../../../../raw/code/simply/simply/utils/common.py#L489) — Keeps the rows that satisfy the row mask. — documented in [simply-utils-common](../../../concepts/simply-utils-common.md)
  - `ndim(self)` — [`L341`](../../../../../../raw/code/simply/simply/utils/common.py#L341)
  - `row(self, idx: jax.typing.ArrayLike)` — [`L379`](../../../../../../raw/code/simply/simply/utils/common.py#L379) — Returns the row at the given index.
  - `row_ids(self)` — [`L369`](../../../../../../raw/code/simply/simply/utils/common.py#L369) — documented in [simply-utils-common](../../../concepts/simply-utils-common.md)
  - `row_starts(self)` — [`L361`](../../../../../../raw/code/simply/simply/utils/common.py#L361)
  - `row_starts_with_end(self)` — [`L365`](../../../../../../raw/code/simply/simply/utils/common.py#L365)
  - `set_padding_value(self, padding_value: jax.typing.ArrayLike)` — [`L438`](../../../../../../raw/code/simply/simply/utils/common.py#L438) — documented in [simply-utils-common](../../../concepts/simply-utils-common.md)
  - `subshape(self)` — [`L345`](../../../../../../raw/code/simply/simply/utils/common.py#L345)
  - `to_numpy_list(self)` — [`L394`](../../../../../../raw/code/simply/simply/utils/common.py#L394)
  - `to_padded_dense(self, max_len: int, padding_value: jax.typing.ArrayLike = 0)` — [`L402`](../../../../../../raw/code/simply/simply/utils/common.py#L402) — Converts to a padded dense array. — documented in [simply-utils-common](../../../concepts/simply-utils-common.md)
  - `total_length(self)` — [`L337`](../../../../../../raw/code/simply/simply/utils/common.py#L337)
  - `data` — [`L324`](../../../../../../raw/code/simply/simply/utils/common.py#L324)
  - `lens` — [`L325`](../../../../../../raw/code/simply/simply/utils/common.py#L325)
- protocol/private: `__post_init__`[`L327`](../../../../../../raw/code/simply/simply/utils/common.py#L327)
- used by: [`apply`](../model_lib.md#Attention.apply), [`update_decode_state_and_compute_attn`](ragged_paged_attention.md#DecodeState.update_decode_state_and_compute_attn), [`test_continue_decode`](ragged_paged_attention_test.md#SamplingStateTest.test_continue_decode), [`test_ragged_issue_tokens`](ragged_paged_attention_test.md#SamplingStateTest.test_ragged_issue_tokens), [`test_update_decode_state_and_compute_attn`](ragged_paged_attention_test.md#DecodeStateTest.test_update_decode_state_and_compute_attn), [`mixed_step`](ragged_paged_attention.md#SamplingState.mixed_step), [`insert`](ragged_paged_attention.md#DecodeState.insert), [`release_for_window`](ragged_paged_attention.md#DecodeState.release_for_window), [`allocate`](ragged_paged_attention.md#DecodeState.allocate), [`update_with_ragged_output`](ragged_paged_attention.md#SamplingState.update_with_ragged_output), [`release`](ragged_paged_attention.md#DecodeState.release), [`test_ragged_array`](common_test.md#RaggedArrayTest.test_ragged_array), [`ragged_issue_tokens`](ragged_paged_attention.md#SamplingState.ragged_issue_tokens), [`test_nd_ragged`](common_test.md#RaggedArrayTest.test_nd_ragged), [`test_ragged_array_invalid_lens`](common_test.md#RaggedArrayTest.test_ragged_array_invalid_lens), [`test_ragged_concat`](common_test.md#RaggedArrayTest.test_ragged_concat), [`RaggedArray`](ragged_paged_attention_test.md#RaggedArray), [`RaggedArray`](ragged_paged_attention.md#RaggedArray), [`test_keep_last_ncols`](common_test.md#RaggedArrayTest.test_keep_last_ncols), [`test_keep_rows`](common_test.md#RaggedArrayTest.test_keep_rows), [`test_padding_value`](common_test.md#RaggedArrayTest.test_padding_value), [`test_to_padded_dense`](common_test.md#RaggedArrayTest.test_to_padded_dense)

## Functions
- `_as_is(*args: Any)` — [`L582`](../../../../../../raw/code/simply/simply/utils/common.py#L582)
- `_inner_wrapper(*args: Any)` — [`L591`](../../../../../../raw/code/simply/simply/utils/common.py#L591)
- `_outer_wrapper(f)` — [`L589`](../../../../../../raw/code/simply/simply/utils/common.py#L589)
- `_transfer_metadata(base, target)` — [`L99`](../../../../../../raw/code/simply/simply/utils/common.py#L99)
- `convert_array_with_abstract(x: jax.Array, abstract: jax.ShapeDtypeStruct)` — [`L517`](../../../../../../raw/code/simply/simply/utils/common.py#L517) — Converts an array to the given abstract specified dtype/sharding.
- `convert_columns_to_rows(columns: Mapping[str, np.typing.ArrayLike])` — [`L242`](../../../../../../raw/code/simply/simply/utils/common.py#L242) — Converts a column view to a sequence of rows.
- `convert_or_dequantize(a: Array | Mapping[str, Array], dtype: jax.typing.DTypeLike = 'bfloat16')` — [`L192`](../../../../../../raw/code/simply/simply/utils/common.py#L192) — Dequantizes an quantized structure if given, otherwise casts dtype. — documented in [simply-utils-module](../../../concepts/simply-utils-module.md)
- `convert_rows_to_columns(rows: Sequence[Mapping[str, np.typing.ArrayLike]])` — [`L231`](../../../../../../raw/code/simply/simply/utils/common.py#L231) — Converts a sequence of rows to a column view.
- `eval_abstract_output(fn: Callable[..., Any], *args, **kwargs)` — [`L208`](../../../../../../raw/code/simply/simply/utils/common.py#L208) — Returns jax.ShapeDtypeStruct tree for given function. — documented in [simply-serving-vanilla_server](../../../concepts/simply-serving-vanilla_server.md)
- `find_unused_argpaths(func: Callable[[Any], Any], argtree: PyTree)` — [`L253`](../../../../../../raw/code/simply/simply/utils/common.py#L253) — Analyzes a JAX function to find args that are not used in the computation.
- `get_raw_arrays(tree: PyTree)` — [`L91`](../../../../../../raw/code/simply/simply/utils/common.py#L91) — documented in [simply-utils-common](../../../concepts/simply-utils-common.md)
- `named_jit(fn: Callable[..., Any], name: str, **kwargs: Any)` — [`L224`](../../../../../../raw/code/simply/simply/utils/common.py#L224) — Returns a jitted function with the given name.
- `named_partial_fn(fn: Callable[..., Any], name: str, **kwargs: Any)` — [`L215`](../../../../../../raw/code/simply/simply/utils/common.py#L215) — Returns a partial function with the given name.
- `neg_inf(dtype: jax.typing.DTypeLike)` — [`L541`](../../../../../../raw/code/simply/simply/utils/common.py#L541)
- `pad_to_len(arr: np.ndarray, seq_len: int, pad_value: Any, dtype: Any)` — [`L561`](../../../../../../raw/code/simply/simply/utils/common.py#L561) — Pads array to target length. Does not truncate.
- `quantize_array(w: Array, symmetric: bool = False)` — [`L179`](../../../../../../raw/code/simply/simply/utils/common.py#L179)
- `reduce_same(seq: Sequence[Any])` — [`L552`](../../../../../../raw/code/simply/simply/utils/common.py#L552) — Reduces a list of same values to a single value.
- `round_up_to_base(x: int, base: int, threshold: int = 128)` — [`L571`](../../../../../../raw/code/simply/simply/utils/common.py#L571) — Rounds up to the nearest base multiple, but not when below threshold.
- `sorted_with_indices(x: Sequence[_TypeVarT], key: Callable[[_TypeVarT], Any] | None = None, reverse: bool = False)` — [`L295`](../../../../../../raw/code/simply/simply/utils/common.py#L295) — Returns a sorted sequence with indices.
- `transfer_metadata(base_tree: PyTree, target_tree: PyTree)` — [`L97`](../../../../../../raw/code/simply/simply/utils/common.py#L97) — Transfer metadata from base to target. — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
- `unsorted(sorted_x: Sequence[_TypeVarT], indices: Sequence[int])` — [`L309`](../../../../../../raw/code/simply/simply/utils/common.py#L309) — Returns a unsorted sequence with the given indices.
- `when(condition: jax.typing.ArrayLike)` — [`L579`](../../../../../../raw/code/simply/simply/utils/common.py#L579) — Returns a decorator that conditionally applies a function.

## Module values
- `Array` — [`L51`](../../../../../../raw/code/simply/simply/utils/common.py#L51)
- `BasicType` — [`L40`](../../../../../../raw/code/simply/simply/utils/common.py#L40)
- `CacheValue` — [`L53`](../../../../../../raw/code/simply/simply/utils/common.py#L53)
- `PartitionAnnotation` — [`L38`](../../../../../../raw/code/simply/simply/utils/common.py#L38)
- `PyTree` — [`L50`](../../../../../../raw/code/simply/simply/utils/common.py#L50) — documented in [simply-model_lib](../../../concepts/simply-model_lib.md)
- `RawT` — [`L52`](../../../../../../raw/code/simply/simply/utils/common.py#L52)
- `THREAD_CONTEXT` — [`L56`](../../../../../../raw/code/simply/simply/utils/common.py#L56)
- `_TypeVarT` — [`L292`](../../../../../../raw/code/simply/simply/utils/common.py#L292)

