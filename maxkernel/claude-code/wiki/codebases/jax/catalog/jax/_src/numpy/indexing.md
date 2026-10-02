---
title: 'Module: jax/_src/numpy/indexing.py'
type: catalog
provenance: extracted
module: jax/_src/numpy/indexing.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.numpy.indexing`/
symbols:
  _index_to_gather: _index_to_gather().
  NDIndexer.to_dynamic_slice: NDIndexer#to_dynamic_slice().
  take_along_axis: take_along_axis().
  ParsedIndex.index: ParsedIndex#index.
  NDIndexer.to_static_slice: NDIndexer#to_static_slice().
  rewriting_take: rewriting_take().
  IndexType: IndexType#
  ParsedIndex.typ: ParsedIndex#typ.
  NDIndexer.normalize_indices: NDIndexer#normalize_indices().
  _take: _take().
  IndexType.from_index: IndexType#from_index().
  _parse_indices: _parse_indices().
  NDIndexer.expand_bool_indices: NDIndexer#expand_bool_indices().
  _static_slice: _static_slice().
  _dynamic_slice: _dynamic_slice().
  put: put().
  NDIndexer.shape: NDIndexer#shape.
  put_along_axis: put_along_axis().
  place: place().
  ParsedIndex.consumed_axes: ParsedIndex#consumed_axes.
  NDIndexer.expand_ellipses: NDIndexer#expand_ellipses().
  NDIndexer.indices: NDIndexer#indices.
  NDIndexer.expand_scalar_bool_indices: NDIndexer#expand_scalar_bool_indices().
  NDIndexer.has_partial_slices: NDIndexer#has_partial_slices().
  NDIndexer.convert_sequences_to_arrays: NDIndexer#convert_sequences_to_arrays().
  _normalize_index: _normalize_index().
  _gather: _gather().
  take: take().
  NDIndexer.validate_static_indices: NDIndexer#validate_static_indices().
  ParsedIndex: ParsedIndex#
  NDIndexer.tree_flatten: NDIndexer#tree_flatten().
  NDIndexer.validate_slices: NDIndexer#validate_slices().
  NDIndexer.is_dynamic: NDIndexer#is_dynamic().
  IndexType.ARRAY: IndexType#ARRAY.
  NDIndexer.is_advanced_int_indexer: NDIndexer#is_advanced_int_indexer().
  _is_boolean_index: _is_boolean_index().
  NDIndexer: NDIndexer#
  IndexType.INTEGER: IndexType#INTEGER.
  IndexType.BOOLEAN: IndexType#BOOLEAN.
  NDIndexer.is_sharded: NDIndexer#is_sharded().
  IndexType.SLICE: IndexType#SLICE.
  NDIndexer.from_raw_indices: NDIndexer#from_raw_indices().
  _StaticSliceIndexer.is_trivial_slice: _StaticSliceIndexer#is_trivial_slice().
  IndexType.NONE: IndexType#NONE.
  IndexType.DYNAMIC_SLICE: IndexType#DYNAMIC_SLICE.
  eliminate_deprecated_list_indexing: eliminate_deprecated_list_indexing().
  export: export.
  NDIndexer.to_gather: NDIndexer#to_gather().
  IndexType.ELLIPSIS: IndexType#ELLIPSIS.
  _make_along_axis_idx: _make_along_axis_idx().
  IndexingStrategy: IndexingStrategy#
  _is_slice_element_none_or_constant_or_symbolic: _is_slice_element_none_or_constant_or_symbolic().
  NDIndexer.tree_unflatten: NDIndexer#tree_unflatten().
  _DynamicSliceIndexer.start_indices: _DynamicSliceIndexer#start_indices.
  _StaticSliceIndexer.start_indices: _StaticSliceIndexer#start_indices.
  _GatherIndexer.gather_indices: _GatherIndexer#gather_indices.
  _GatherIndexer.dnums: _GatherIndexer#dnums.
  _GatherIndexer.slice_sharding: _GatherIndexer#slice_sharding.
  _should_unpack_list_index: _should_unpack_list_index().
  _is_scalar: _is_scalar().
  _StaticSliceIndexer.limit_indices: _StaticSliceIndexer#limit_indices.
  _StaticSliceIndexer.strides: _StaticSliceIndexer#strides.
  _StaticSliceIndexer.rev_axes: _StaticSliceIndexer#rev_axes.
  _StaticSliceIndexer.squeeze_axes: _StaticSliceIndexer#squeeze_axes.
  _StaticSliceIndexer.newaxis_dims: _StaticSliceIndexer#newaxis_dims.
  _DynamicSliceIndexer.rev_axes: _DynamicSliceIndexer#rev_axes.
  _DynamicSliceIndexer.squeeze_axes: _DynamicSliceIndexer#squeeze_axes.
  _DynamicSliceIndexer.newaxis_dims: _DynamicSliceIndexer#newaxis_dims.
  _StaticSliceIndexer: _StaticSliceIndexer#
  _DynamicSliceIndexer: _DynamicSliceIndexer#
  _GatherIndexer: _GatherIndexer#
  IndexingStrategy.AUTO: IndexingStrategy#AUTO.
  _DynamicSliceIndexer.slice_sizes: _DynamicSliceIndexer#slice_sizes.
  _DynamicSliceIndexer.trivial_slicing: _DynamicSliceIndexer#trivial_slicing.
  _DynamicSliceIndexer.normalize_indices: _DynamicSliceIndexer#normalize_indices.
  take_along_axis.replace: take_along_axis().replace().
  _is_integer_index: _is_integer_index().
  IndexingStrategy.STATIC_SLICE: IndexingStrategy#STATIC_SLICE.
  IndexingStrategy.DYNAMIC_SLICE: IndexingStrategy#DYNAMIC_SLICE.
  _GatherIndexer.slice_shape: _GatherIndexer#slice_shape.
  _GatherIndexer.gather_slice_shape: _GatherIndexer#gather_slice_shape.
  _GatherIndexer.unique_indices: _GatherIndexer#unique_indices.
  _GatherIndexer.indices_are_sorted: _GatherIndexer#indices_are_sorted.
  _GatherIndexer.reversed_y_dims: _GatherIndexer#reversed_y_dims.
  _GatherIndexer.newaxis_dims: _GatherIndexer#newaxis_dims.
  _GatherIndexer.scalar_bool_dims: _GatherIndexer#scalar_bool_dims.
  IndexingStrategy.GATHER: IndexingStrategy#GATHER.
  IndexingStrategy.SCATTER: IndexingStrategy#SCATTER.
---
# Module: [`jax/_src/numpy/indexing.py`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py)

## Classes
### `IndexType`  ·  implements/extends Enum
- def: [`jax/_src/numpy/indexing.py:57`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L57)
- doc: Enum for tracking the type of an index.
- signature: `class IndexType(enum.Enum):`
- members:
  - `from_index(cls, idx: Index)` — [`L68`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L68) — Create an IndexType enum from a supported JAX array index.
  - `ARRAY` — [`L64`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L64)
  - `BOOLEAN` — [`L63`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L63)
  - `DYNAMIC_SLICE` — [`L65`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L65)
  - `ELLIPSIS` — [`L61`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L61)
  - `INTEGER` — [`L62`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L62)
  - `NONE` — [`L59`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L59)
  - `SLICE` — [`L60`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L60)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array), [`dtype`](../basearray.md#Array.dtype), [`issubdtype`](../dtypes.md#issubdtype), [`asarray`](array_constructors.md#asarray), [`Slice`](../indexing.md#Slice), [`Index`](../typing.md#Index), [`_is_boolean_index`](indexing.md#_is_boolean_index), [`eval_shape`](../api.md#eval_shape), [`_is_integer_index`](indexing.md#_is_integer_index)
- used by: [`_index_to_gather`](indexing.md#_index_to_gather), [`to_dynamic_slice`](indexing.md#NDIndexer.to_dynamic_slice), [`to_static_slice`](indexing.md#NDIndexer.to_static_slice), [`normalize_indices`](indexing.md#NDIndexer.normalize_indices), [`typ`](indexing.md#ParsedIndex.typ), [`_parse_indices`](indexing.md#_parse_indices), [`expand_bool_indices`](indexing.md#NDIndexer.expand_bool_indices), [`expand_ellipses`](indexing.md#NDIndexer.expand_ellipses), [`expand_scalar_bool_indices`](indexing.md#NDIndexer.expand_scalar_bool_indices), [`has_partial_slices`](indexing.md#NDIndexer.has_partial_slices), [`validate_static_indices`](indexing.md#NDIndexer.validate_static_indices), [`is_dynamic`](indexing.md#NDIndexer.is_dynamic), [`validate_slices`](indexing.md#NDIndexer.validate_slices), [`is_advanced_int_indexer`](indexing.md#NDIndexer.is_advanced_int_indexer)

### `IndexingStrategy`  ·  implements/extends Enum
- def: [`jax/_src/numpy/indexing.py:1072`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1072)
- signature: `class IndexingStrategy(enum.Enum):`
- members:
  - `AUTO` — [`L1073`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1073)
  - `DYNAMIC_SLICE` — [`L1077`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1077)
  - `GATHER` — [`L1074`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1074)
  - `SCATTER` — [`L1075`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1075)
  - `STATIC_SLICE` — [`L1076`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1076)
- used by: [`rewriting_take`](indexing.md#rewriting_take)

### `NDIndexer`
- def: [`jax/_src/numpy/indexing.py:181`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L181)
- doc: Object that implements NumPy-style indexing operations on top of JAX.
- signature: `class NDIndexer:`
- members:
  - `convert_sequences_to_arrays(self)` — [`L307`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L307)
  - `expand_bool_indices(self)` — [`L254`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L254) — Returns a new NDIndexer with boolean indices replaced by array indices.
  - `expand_ellipses(self)` — [`L312`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L312) — Returns a new indexer with ellipsis and implicit trailing slices
  - `expand_scalar_bool_indices(self, sharding_spec: Any = None)` — [`L284`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L284)
  - `from_raw_indices(cls, indices: Index | tuple[Index, ...], shape: tuple[int, ...])` — [`L195`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L195) — Create an NDIndexer object from raw user-supplied indices.
  - `has_partial_slices(self)` — [`L235`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L235) — Check whether the indexer contains partial slices.
  - `is_advanced_int_indexer(self)` — [`L571`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L571) — Returns True if idx should trigger int array indexing, False otherwise.
  - `is_dynamic(i: ParsedIndex)` — [`L583`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L583)
  - `is_sharded(arr)` — [`L231`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L231) — Check whether the array is sharded.
  - `normalize_indices(self)` — [`L330`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L330)
  - `to_dynamic_slice(self, *, arr_is_sharded: bool = False, normalize_indices: bool = True, mode: str | slicing.GatherScatterMode | None)` — [`L459`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L459) — Convert to DynamicSliceIndexer data structure.
  - `to_gather(self, x_sharding: NamedSharding | Any, normalize_indices: bool = True)` — [`L577`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L577)
  - `to_static_slice(self, *, arr_is_sharded: bool = False, normalize_indices: bool = True, mode: str | slicing.GatherScatterMode | None)` — [`L359`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L359) — Convert to StaticSliceIndexer data structure.
  - `tree_flatten(self)` — [`L581`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L581)
  - `tree_unflatten(cls, aux_data, children)` — [`L593`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L593)
  - `validate_slices(self)` — [`L217`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L217) — Check that all slices have static start/stop/step values.
  - `validate_static_indices(self, normalize_indices: bool = True)` — [`L201`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L201) — Check that all static integer indices are in-bounds.
  - `indices` — [`L192`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L192)
  - `shape` — [`L191`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L191)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array), [`ArrayLike`](../basearray.md#ArrayLike), [`shape`](../basearray.md#Array.shape), [`dtype`](../basearray.md#Array.dtype), [`typeof`](../core.md#typeof), [`issubdtype`](../dtypes.md#issubdtype), [`NamedSharding`](../named_sharding.md#NamedSharding), [`select`](../lax/lax.md#select), [`_const`](../lax/lax.md#_const), [`_index_to_gather`](indexing.md#_index_to_gather), [`convert_element_type`](../lax/lax.md#convert_element_type), [`safe_zip`](../util.md#safe_zip), [`asarray`](array_constructors.md#asarray), [`add`](../lax/lax.md#add), [`Slice`](../indexing.md#Slice), [`index`](indexing.md#ParsedIndex.index), [`IndexType`](indexing.md#IndexType), [`GatherScatterMode`](../lax/slicing.md#GatherScatterMode), [`typ`](indexing.md#ParsedIndex.typ), [`ArrayImpl`](../array.md#ArrayImpl), [`start`](../indexing.md#Slice.start), [`sharding`](../array.md#ArrayImpl.sharding), [`size`](../indexing.md#Slice.size), [`definitely_equal`](../core.md#definitely_equal), [`is_constant_dim`](../core.md#is_constant_dim), [`_parse_indices`](indexing.md#_parse_indices), [`stride`](../indexing.md#Slice.stride), [`consumed_axes`](indexing.md#ParsedIndex.consumed_axes), [`is_concrete`](../core.md#is_concrete), [`from_any`](../lax/slicing.md#GatherScatterMode.from_any), [`int_dtype_for_shape`](../lax/utils.md#int_dtype_for_shape), [`ParsedIndex`](indexing.md#ParsedIndex), [`register_pytree_node_class`](../tree_util.md#register_pytree_node_class), [`Index`](../typing.md#Index), [`ARRAY`](indexing.md#IndexType.ARRAY), [`dimension_as_value`](../core.md#dimension_as_value), [`INTEGER`](indexing.md#IndexType.INTEGER), [`num_devices`](../sharding.md#Sharding.num_devices), [`BOOLEAN`](indexing.md#IndexType.BOOLEAN), [`PROMISE_IN_BOUNDS`](../lax/slicing.md#GatherScatterMode.PROMISE_IN_BOUNDS)  (+25 more)
- used by: [`_index_to_gather`](indexing.md#_index_to_gather), [`rewriting_take`](indexing.md#rewriting_take)

### `ParsedIndex`  ·  implements/extends NamedTuple
- def: [`jax/_src/numpy/indexing.py:111`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L111)
- doc: Structure for tracking an indexer parsed within the context of an array shape.
- signature: `class ParsedIndex(NamedTuple):`
- members:
  - `consumed_axes` — [`L115`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L115)
  - `index` — [`L113`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L113)
  - `typ` — [`L114`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L114)
- uses (calls/refs, reference-scoped): [`IndexType`](indexing.md#IndexType), [`Index`](../typing.md#Index)
- used by: [`_index_to_gather`](indexing.md#_index_to_gather), [`to_dynamic_slice`](indexing.md#NDIndexer.to_dynamic_slice), [`to_static_slice`](indexing.md#NDIndexer.to_static_slice), [`normalize_indices`](indexing.md#NDIndexer.normalize_indices), [`_parse_indices`](indexing.md#_parse_indices), [`expand_bool_indices`](indexing.md#NDIndexer.expand_bool_indices), [`expand_ellipses`](indexing.md#NDIndexer.expand_ellipses), [`indices`](indexing.md#NDIndexer.indices), [`expand_scalar_bool_indices`](indexing.md#NDIndexer.expand_scalar_bool_indices), [`has_partial_slices`](indexing.md#NDIndexer.has_partial_slices), [`convert_sequences_to_arrays`](indexing.md#NDIndexer.convert_sequences_to_arrays), [`validate_static_indices`](indexing.md#NDIndexer.validate_static_indices), [`tree_flatten`](indexing.md#NDIndexer.tree_flatten), [`is_dynamic`](indexing.md#NDIndexer.is_dynamic), [`validate_slices`](indexing.md#NDIndexer.validate_slices), [`is_advanced_int_indexer`](indexing.md#NDIndexer.is_advanced_int_indexer), [`tree_unflatten`](indexing.md#NDIndexer.tree_unflatten)

### `_DynamicSliceIndexer`
- def: [`jax/_src/numpy/indexing.py:1257`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1257)
- members:
  - `newaxis_dims` — [`L1262`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1262)
  - `normalize_indices` — [`L1264`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1264)
  - `rev_axes` — [`L1260`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1260)
  - `slice_sizes` — [`L1259`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1259)
  - `squeeze_axes` — [`L1261`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1261)
  - `start_indices` — [`L1258`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1258)
  - `trivial_slicing` — [`L1263`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1263)
- uses (calls/refs, reference-scoped): [`ArrayLike`](../basearray.md#ArrayLike)
- used by: [`to_dynamic_slice`](indexing.md#NDIndexer.to_dynamic_slice), [`_dynamic_slice`](indexing.md#_dynamic_slice)

### `_GatherIndexer`
- def: [`jax/_src/numpy/indexing.py:1267`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1267)
- members:
  - `dnums` — [`L1275`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1275)
  - `gather_indices` — [`L1273`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1273)
  - `gather_slice_shape` — [`L1271`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1271)
  - `indices_are_sorted` — [`L1281`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1281)
  - `newaxis_dims` — [`L1289`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1289)
  - `reversed_y_dims` — [`L1285`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1285)
  - `scalar_bool_dims` — [`L1293`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1293)
  - `slice_shape` — [`L1269`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1269)
  - `slice_sharding` — [`L1296`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1296)
  - `unique_indices` — [`L1280`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1280)
- uses (calls/refs, reference-scoped): [`ArrayLike`](../basearray.md#ArrayLike), [`NamedSharding`](../named_sharding.md#NamedSharding), [`GatherDimensionNumbers`](../lax/slicing.md#GatherDimensionNumbers)
- used by: [`_index_to_gather`](indexing.md#_index_to_gather), [`to_gather`](indexing.md#NDIndexer.to_gather)

### `_StaticSliceIndexer`
- def: [`jax/_src/numpy/indexing.py:1240`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1240)
- members:
  - `is_trivial_slice(self, arr_shape: Sequence[int])` — [`L1248`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1248)
  - `limit_indices` — [`L1242`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1242)
  - `newaxis_dims` — [`L1246`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1246)
  - `rev_axes` — [`L1244`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1244)
  - `squeeze_axes` — [`L1245`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1245)
  - `start_indices` — [`L1241`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1241)
  - `strides` — [`L1243`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1243)
- used by: [`to_static_slice`](indexing.md#NDIndexer.to_static_slice), [`_static_slice`](indexing.md#_static_slice)

## Functions
- `_dynamic_slice(arr: Array, indexer: _DynamicSliceIndexer)` — [`L1174`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1174) — Equivalent of arr[idx] implemented in terms of static :func:`lax.dynamic_slice`.
- `_gather(arr, dynamic_idx, *, treedef, indices_are_sorted, unique_indices, mode, fill_value, normalize_indices)` — [`L1200`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1200)
- `_index_to_gather(indexer: NDIndexer, *, x_sharding: NamedSharding | Any, normalize_indices: bool = True)` — [`L1299`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1299)
- `_is_boolean_index(i)` — [`L1523`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1523)
- `_is_integer_index(idx: Any)` — [`L1068`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1068)
- `_is_scalar(x)` — [`L1542`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1542) — Checks if a Python or NumPy scalar.
- `_is_slice_element_none_or_constant_or_symbolic(elt)` — [`L1533`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1533) — Return True if elt is a constant or None.
- `_make_along_axis_idx(shape, indices, axis)` — [`L970`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L970)
- `_normalize_index(index, axis_size)` — [`L743`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L743) — Normalizes an index value in the range [-N, N) to the range [0, N).
- `_parse_indices(indices: tuple[Index, ...], shape: tuple[int, ...])` — [`L118`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L118) — Parse indices in the context of an array shape.
- `_should_unpack_list_index(x)` — [`L1491`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1491) — Helper for eliminate_deprecated_list_indexing.
- `_static_slice(arr: Array, indexer: _StaticSliceIndexer)` — [`L1154`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1154) — Equivalent of arr[idx] implemented in terms of static :func:`lax.slice` operations.
- `_take(a, indices, axis: int | None = None, out=None, mode=None, unique_indices=False, indices_are_sorted=False, fill_value=None)` — [`L691`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L691)
- `eliminate_deprecated_list_indexing(idx)` — [`L1498`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1498)
- `place(arr: ArrayLike, mask: ArrayLike, vals: ArrayLike, *, inplace: bool = True)` — [`L1551`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1551) — Update array elements based on a mask.
- `put(a: ArrayLike, ind: ArrayLike, v: ArrayLike, mode: str | None = None, *, inplace: bool = True)` — [`L1627`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1627) — Put elements into an array at given indices.
- `put_along_axis(arr: ArrayLike, indices: ArrayLike, values: ArrayLike, axis: int | None, inplace: bool = True, *, mode: str | None = None)` — [`L978`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L978) — Put values into the destination array by matching 1d index and data slices.
- `replace(tup, val)` — [`L867`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L867)
- `rewriting_take(arr: Array, idx: Index | tuple[Index, ...], *, indices_are_sorted: bool = False, unique_indices: bool = False, mode: str | slicing.GatherScatterMode | None = None, fill_value: ArrayLike | None = None, normalize_indices: bool = True, out_sharding: NamedSharding | PartitionSpec | None = None, strategy: IndexingStrategy = IndexingStrategy.AUTO)` — [`L1080`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L1080)
- `take(a: ArrayLike, indices: ArrayLike, axis: int | None = None, out: None = None, mode: str | None = None, unique_indices: bool = False, indices_are_sorted: bool = False, fill_value: StaticScalar | None = None)` — [`L601`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L601) — Take elements from an array.
- `take_along_axis(arr: ArrayLike, indices: ArrayLike, axis: int | None = -1, mode: str | slicing.GatherScatterMode | None = None, fill_value: StaticScalar | None = None, *, wrap_negative_indices: bool | None = None)` — [`L762`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L762) — Take elements from an array.

## Module values
- `export` — [`L52`](../../../../../../../raw/code/jax/jax/_src/numpy/indexing.py#L52)

