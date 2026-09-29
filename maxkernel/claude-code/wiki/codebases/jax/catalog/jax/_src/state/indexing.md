---
title: 'Module: jax/_src/state/indexing.py'
type: catalog
provenance: extracted
module: jax/_src/state/indexing.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.state.indexing`/
symbols:
  NDIndexer: NDIndexer#
  NDIndexer.indices: NDIndexer#indices.
  NDIndexer.from_indices_shape: NDIndexer#from_indices_shape().
  NDIndexer.transform_type: NDIndexer#transform_type().
  NDIndexer.__post_init__: NDIndexer#__post_init__().
  NDIndexer.int_indexer_shape: NDIndexer#int_indexer_shape.
  _pp_slice: _pp_slice().
  NDIndexer.shape: NDIndexer#shape.
  NDIndexer.pretty_print: NDIndexer#pretty_print().
  unpack_ndindexer: unpack_ndindexer().
  NDIndexer._validate_sharding: NDIndexer#_validate_sharding().
  NDIndexer.make_trivial_indexer: NDIndexer#make_trivial_indexer().
  NDIndexer.get_indexer_shape: NDIndexer#get_indexer_shape().
  DShapedArray.update: DShapedArray#update().
  NDIndexer.tree_flatten: NDIndexer#tree_flatten().
  DShapedArray.str_short: DShapedArray#str_short().
  NDIndexer.is_dynamic_size: NDIndexer#is_dynamic_size().
  DShapedArray.__eq__: DShapedArray#__eq__().
  DShapedArray.__hash__: DShapedArray#__hash__().
  DimIndexer: DimIndexer.
  NDIndexer.get_indexer_shape_static: NDIndexer#get_indexer_shape_static().
  DShapedArray.dtype: DShapedArray#dtype.
  DShapedArray.weak_type: DShapedArray#weak_type.
  IntIndexer: IntIndexer.
  _maybe_concretize: _maybe_concretize().
  DShapedArray.shape: DShapedArray#shape.
  DShapedArray.__repr__: DShapedArray#__repr__().
  DShapedArray.__str__: DShapedArray#__str__().
  DShapedArray: DShapedArray#
  indexer_transform_type_registry.indexer_transform_type_registry: indexer_transform_type_registry.indexer_transform_type_registry.
  NDIndexer.tree_unflatten: NDIndexer#tree_unflatten().
  NDIndexer.undo: NDIndexer#undo().
  DShapedArray.broadcast: DShapedArray#broadcast.
  DShapedArray.transpose: DShapedArray#transpose.
  DShapedArray.reshape: DShapedArray#reshape.
  DShapedArray._len: DShapedArray#_len().
  DShapedArray.update_weak_type: DShapedArray#update_weak_type().
  DShapedArray._bool: DShapedArray#_bool.
  DShapedArray._int: DShapedArray#_int.
  DShapedArray._float: DShapedArray#_float.
  DShapedArray._complex: DShapedArray#_complex.
  DShapedArray._hex: DShapedArray#_hex.
  DShapedArray._oct: DShapedArray#_oct.
  NDIndexer.validate: NDIndexer#validate.
  DShapedArray.__init__: DShapedArray#__init__().
  DShapedArray.lower_val: DShapedArray#lower_val().
  DShapedArray.raise_val: DShapedArray#raise_val().
  DShapedArray.lo_ty: DShapedArray#lo_ty().
  DShapedArray.ndim: DShapedArray#ndim.
  DShapedArray.size: DShapedArray#size.
  DShapedArray._iter: DShapedArray#_iter.
  DShapedArray.__ne__: DShapedArray#__ne__().
  DShapedArray._index: DShapedArray#_index.
---
# Module: [`jax/_src/state/indexing.py`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py)

## Classes
### `DShapedArray`
- def: [`jax/_src/state/indexing.py:359`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L359)
- members:
  - `lo_ty(self)` — [`L367`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L367)
  - `lower_val(self, val)` — [`L365`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L365)
  - `raise_val(self, val)` — [`L366`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L366)
  - `str_short(self)` — [`L407`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L407)
  - `update(self, shape=None, dtype=None, weak_type=None)` — [`L369`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L369)
  - `update_weak_type(self, weak_type)` — [`L417`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L417)
  - `broadcast` — [`L383`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L383)
  - `dtype` — [`L362`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L362)
  - `ndim` — [`L378`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L378)
  - `reshape` — [`L385`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L385)
  - `shape` — [`L361`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L361)
  - `size` — [`L379`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L379)
  - `transpose` — [`L384`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L384)
  - `weak_type` — [`L363`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L363)
- protocol/private: `__eq__`[`L388`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L388), `__hash__`[`L393`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L393), `__init__`[`L360`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L360), `__ne__`[`L396`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L396), `__repr__`[`L399`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L399), `__str__`[`L403`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L403), `_bool`[`L420`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L420), `_complex`[`L423`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L423), `_float`[`L422`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L422), `_hex`[`L424`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L424), `_index`[`L426`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L426), `_int`[`L421`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L421), `_iter`[`L386`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L386), `_len`[`L411`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L411), `_oct`[`L425`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L425)
- uses (calls/refs, reference-scoped): [`aval_method`](../core.md#aval_method), [`concretization_function_error`](../core.md#concretization_function_error), [`_dtype_object`](../core.md#_dtype_object)
- used by: [`transform_type`](indexing.md#NDIndexer.transform_type)

### `NDIndexer`
- def: [`jax/_src/state/indexing.py:85`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L85)
- members:
  - `from_indices_shape(cls, indices, shape)` — [`L172`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L172)
  - `get_indexer_shape(self)` — [`L271`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L271)
  - `get_indexer_shape_static(self)` — [`L288`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L288)
  - `is_dynamic_size(self)` — [`L150`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L150)
  - `make_trivial_indexer(cls, shape: tuple[int, ...])` — [`L265`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L265)
  - `pretty_print(self, context: core.JaxprPpContext)` — [`L347`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L347)
  - `transform_type(self, x: core.AbstractValue)` — [`L294`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L294)
  - `tree_flatten(self)` — [`L153`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L153)
  - `tree_unflatten(cls, data, flat_idx)` — [`L161`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L161)
  - `undo(self, x: core.AbstractValue)` — [`L310`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L310)
  - `indices` — [`L86`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L86)
  - `int_indexer_shape` — [`L88`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L88)
  - `shape` — [`L87`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L87)
  - `validate` — [`L90`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L90)
- protocol/private: `__post_init__`[`L92`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L92), `_validate_sharding`[`L313`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L313)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array), [`AbstractRef`](types.md#AbstractRef), [`ShapedArray`](../core.md#ShapedArray), [`typeof`](../core.md#typeof), [`AbstractValue`](../core.md#AbstractValue), [`tree_unflatten`](../tree_util.md#tree_unflatten), [`dtype`](../core.md#ShapedArray.dtype), [`tree_flatten`](../tree_util.md#tree_flatten), [`TransformedRef`](types.md#TransformedRef), [`text`](../pretty_printer.md#text), [`Transform`](types.md#Transform), [`Slice`](../indexing.md#Slice), [`partition_list`](../util.md#partition_list), [`tree_leaves`](../tree_util.md#tree_leaves), [`sharding`](../core.md#ShapedArray.sharding), [`JaxprPpContext`](../core.md#JaxprPpContext), [`canonicalize_shape`](../core.md#canonicalize_shape), [`Doc`](../pretty_printer.md#Doc), [`start`](../indexing.md#Slice.start), [`concat`](../pretty_printer.md#concat), [`update`](../core.md#ShapedArray.update), [`size`](../indexing.md#Slice.size), [`merge_lists`](../util.md#merge_lists), [`pp_var`](../core.md#pp_var), [`shape`](types.md#TransformedRef.shape), [`stride`](../indexing.md#Slice.stride), [`update`](types.md#AbstractRef.update), [`_pp_slice`](indexing.md#_pp_slice), [`weak_type`](../core.md#ShapedArray.weak_type), [`unpack_ndindexer`](indexing.md#unpack_ndindexer), [`join`](../pretty_printer.md#join), [`broadcast_to`](primitives.md#broadcast_to), [`is_dynamic_size`](../indexing.md#Slice.is_dynamic_size), [`from_slice`](../indexing.md#Slice.from_slice), [`DimIndexer`](indexing.md#DimIndexer), [`_maybe_concretize`](indexing.md#_maybe_concretize), [`DShapedArray`](indexing.md#DShapedArray), [`indexer_transform_type_registry`](indexing.md#indexer_transform_type_registry.indexer_transform_type_registry)
- used by: [`interpret_pallas_call`](../pallas/mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`_handle_transforms`](../pallas/mosaic_gpu/lowering.md#_handle_transforms), [`_body`](../pallas/mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call._execute_grid_for_core._body), [`_compute_offsets_from_indices`](../pallas/triton/lowering.md#_compute_offsets_from_indices), [`_masked_load_lowering_rule`](../pallas/triton/lowering.md#_masked_load_lowering_rule), [`_load_lowering_rule`](../pallas/mosaic/lowering.md#_load_lowering_rule), [`_load_lowering_rule`](../pallas/mosaic/sc_lowering.md#_load_lowering_rule), [`_store_lowering_rule`](../pallas/mosaic/sc_lowering.md#_store_lowering_rule), [`get_ref_and_transforms`](primitives.md#get_ref_and_transforms), [`_atomic_lowering_rule`](../pallas/triton/primitives.md#_atomic_lowering_rule), [`_masked_swap_lowering_rule`](../pallas/mosaic/lowering.md#_masked_swap_lowering_rule), [`_swap_discharge_rule`](../pallas/primitives.md#_swap_discharge_rule), [`_get_pull_rule`](../pallas/fuser/block_spec.md#_get_pull_rule), [`transform_swap_array`](discharge.md#transform_swap_array), [`_get_eval_rule`](../pallas/fuser/block_spec.md#_get_eval_rule), [`_commute_transform`](../pallas/mosaic_gpu/lowering.md#_commute_transform), [`transform_type`](types.md#Transform.transform_type), [`_load_discharge_rule`](../pallas/primitives.md#_load_discharge_rule), [`_wgmma_lowering`](../pallas/mosaic_gpu/primitives.md#_wgmma_lowering), [`_prng_key_load_lowering_rule`](../pallas/mosaic/lowering.md#_prng_key_load_lowering_rule), [`undo`](types.md#Transform.undo), [`_swap_eval_rule`](../pallas/fuser/block_spec.md#_swap_eval_rule), [`_transform_ref`](../pallas/mosaic/lowering.md#_transform_ref), [`_convert_to_gather_arrays`](discharge.md#_convert_to_gather_arrays), [`_reinterpret_int4_as_uint8`](../pallas/triton/lowering.md#_reinterpret_int4_as_uint8), [`_store_slice_to_kernel_input`](../pallas/mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call._execute_grid_for_core._body._store_slice_to_kernel_input), [`fetch_and_add`](../pallas/mosaic/sc_primitives.md#fetch_and_add), [`_ndindexer_indices`](../pallas/mosaic_gpu/lowering.md#_ndindexer_indices), [`_masked_swap_lowering_rule`](../pallas/triton/lowering.md#_masked_swap_lowering_rule), [`commute_ndindexer`](../pallas/mosaic_gpu/core.md#UntilingTransform.commute_ndindexer), [`_batch_indexer`](primitives.md#_batch_indexer), [`_extract_gmem_copy_params`](../pallas/mosaic_gpu/primitives.md#_extract_gmem_copy_params), [`transform_array`](discharge.md#transform_array), [`_wgmma_warpgroup_lowering`](../pallas/mosaic_gpu/primitives.md#_wgmma_warpgroup_lowering), [`commute_ndindexer`](../pallas/mosaic_gpu/core.md#ExpandLeadingBatchDimensionsTransform.commute_ndindexer), [`_is_contiguous_int4`](../pallas/triton/lowering.md#_is_contiguous_int4), [`_get_barrier_base_index`](../pallas/mosaic_gpu/primitives.md#_get_barrier_base_index), [`_bubble_up_transforms_for_lowering`](../pallas/mosaic_gpu/lowering.md#_bubble_up_transforms_for_lowering), [`_jaxpr_call`](../pallas/primitives.md#_jaxpr_call), [`pretty_print`](types.md#Transform.pretty_print)  (+29 more)

## Functions
- `_maybe_concretize(x: Any)` — [`L73`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L73)
- `_pp_slice(context: core.JaxprPpContext, dim, slc: Slice)` — [`L35`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L35)
- `unpack_ndindexer(indexer: NDIndexer)` — [`L64`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L64)

## Module values
- `DimIndexer` — [`L62`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L62)
- `IntIndexer` — [`L61`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L61)
- `indexer_transform_type_registry` — [`L81`](../../../../../../../raw/code/jax/jax/_src/state/indexing.py#L81)

