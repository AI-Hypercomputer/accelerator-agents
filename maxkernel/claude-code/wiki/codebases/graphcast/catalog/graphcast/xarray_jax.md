---
title: 'Module: graphcast/xarray_jax.py'
type: catalog
provenance: extracted
module: graphcast/xarray_jax.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.xarray_jax`/
symbols:
  DataArray: DataArray().
  Dataset: Dataset().
  Variable: Variable().
  scan: scan().
  assign_coords: assign_coords().
  JaxArrayWrapper: JaxArrayWrapper#
  dims_change_on_unflatten: dims_change_on_unflatten().
  _unflatten_data_array: _unflatten_data_array().
  _unflatten_dataset: _unflatten_dataset().
  wrap: wrap().
  unwrap: unwrap().
  _HashableCoords._variables: _HashableCoords#_variables.
  _wrapped: _wrapped().
  unwrap_data: unwrap_data().
  assign_jax_coords: assign_jax_coords().
  JaxArrayWrapper.jax_array: JaxArrayWrapper#jax_array.
  _unflatten_variable: _unflatten_variable().
  tree_map_with_dims: tree_map_with_dims().
  _DIMS_CHANGE_ON_UNFLATTEN_FN._DIMS_CHANGE_ON_UNFLATTEN_FN: _DIMS_CHANGE_ON_UNFLATTEN_FN._DIMS_CHANGE_ON_UNFLATTEN_FN.
  _vmap_or_pmap: _vmap_or_pmap().
  _vmap_or_pmap.fn_passed_to_pmap: _vmap_or_pmap().fn_passed_to_pmap().
  _split_static_and_jax_coords: _split_static_and_jax_coords().
  _flatten_data_array: _flatten_data_array().
  _flatten_dataset: _flatten_dataset().
  jax_data: jax_data().
  pmap: pmap().
  get_jax_coords: get_jax_coords().
  _wrapped.wrapped_func: _wrapped().wrapped_func().
  apply_ufunc.wrapped_func: apply_ufunc().wrapped_func().
  tree_map_with_dims.wrapped_func: tree_map_with_dims().wrapped_func().
  _HashableCoords: _HashableCoords#
  _HashableCoords._hash: _HashableCoords#_hash.
  apply_ufunc: apply_ufunc().
  _flatten_variable: _flatten_variable().
  DatasetOrDataArray: DatasetOrDataArray.
  _PyTree: _PyTree.
  _Carry: _Carry.
  unwrap_vars: unwrap_vars().
  _vmap_or_pmap.result_fn: _vmap_or_pmap().result_fn().
  tree_map_variables: tree_map_variables().
  scan.scan_fn: scan().scan_fn().
  _JAX_COORD_ATTR_NAME: _JAX_COORD_ATTR_NAME.
  unwrap_coords: unwrap_coords().
  jax_vars: jax_vars().
  JaxArrayWrapper.__array_ufunc__: JaxArrayWrapper#__array_ufunc__().
  JaxArrayWrapper.__array_function__: JaxArrayWrapper#__array_function__().
  JaxArrayWrapper.__repr__: JaxArrayWrapper#__repr__().
  JaxArrayWrapper.shape: JaxArrayWrapper#shape().
  JaxArrayWrapper.dtype: JaxArrayWrapper#dtype().
  JaxArrayWrapper.ndim: JaxArrayWrapper#ndim().
  JaxArrayWrapper.size: JaxArrayWrapper#size().
  JaxArrayWrapper.real: JaxArrayWrapper#real().
  JaxArrayWrapper.imag: JaxArrayWrapper#imag().
  JaxArrayWrapper.__array__: JaxArrayWrapper#__array__().
  JaxArrayWrapper.__getitem__: JaxArrayWrapper#__getitem__.
  JaxArrayWrapper.astype: JaxArrayWrapper#astype.
  JaxArrayWrapper.transpose: JaxArrayWrapper#transpose.
  JaxArrayWrapper.reshape: JaxArrayWrapper#reshape.
  JaxArrayWrapper.all: JaxArrayWrapper#all.
  vmap: vmap().
  _HashableCoords.__repr__: _HashableCoords#__repr__().
  _HashableCoords.__getitem__: _HashableCoords#__getitem__().
  _HashableCoords.__len__: _HashableCoords#__len__().
  _HashableCoords.__iter__: _HashableCoords#__iter__().
  _HashableCoords.__eq__: _HashableCoords#__eq__().
  _WRAPPED_TYPES: _WRAPPED_TYPES.
  _X: _X.
  _Y: _Y.
  DimsChangeFn: DimsChangeFn.
  _drop_with_none_of_dims: _drop_with_none_of_dims().
  _flatten_datatree: _flatten_datatree().
  _unflatten_datatree: _unflatten_datatree().
  _vmap_or_pmap.fn_passed_to_pmap.check_and_remove_leading_dim: _vmap_or_pmap().fn_passed_to_pmap().check_and_remove_leading_dim().
  tree_map_with_dims.is_leaf: tree_map_with_dims().is_leaf().
  JaxArrayWrapper.__init__: JaxArrayWrapper#__init__().
  _HashableCoords.__init__: _HashableCoords#__init__().
  _HashableCoords.__hash__: _HashableCoords#__hash__().
---
# Module: [`graphcast/xarray_jax.py`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py)

## Classes
### `JaxArrayWrapper`
- def: [`graphcast/xarray_jax.py:425`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L425)
- members:
  - `dtype(self)` — [`L481`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L481)
  - `imag(self)` — [`L497`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L497)
  - `ndim(self)` — [`L485`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L485)
  - `real(self)` — [`L493`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L493)
  - `shape(self)` — [`L477`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L477)
  - `size(self)` — [`L489`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L489)
  - `all` — [`L520`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L520)
  - `astype` — [`L510`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L510)
  - `jax_array` — [`L440`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L440)
  - `reshape` — [`L519`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L519)
  - `transpose` — [`L518`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L518)
- protocol/private: `__array__`[`L504`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L504), `__array_function__`[`L460`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L460), `__array_ufunc__`[`L442`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L442), `__getitem__`[`L507`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L507), `__init__`[`L439`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L439), `__repr__`[`L468`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L468)
- uses (calls/refs, reference-scoped): [`_wrapped`](xarray_jax.md#_wrapped)
- used by: [`func`](xarray_jax_test.md#XarrayJaxTest.func), [`fn`](xarray_jax_test.md#XarrayJaxTest.fn), [`unwrap`](xarray_jax.md#unwrap), [`wrap`](xarray_jax.md#wrap), [`test_jit_function_with_xarray_datatree_arguments_and_return`](xarray_jax_test.md#XarrayJaxTest.test_jit_function_with_xarray_datatree_arguments_and_return), [`test_jit_function_with_data_array_and_jax_coords`](xarray_jax_test.md#XarrayJaxTest.test_jit_function_with_data_array_and_jax_coords), [`test_jit_function_with_dataset_and_jax_coords`](xarray_jax_test.md#XarrayJaxTest.test_jit_function_with_dataset_and_jax_coords), [`_split_static_and_jax_coords`](xarray_jax.md#_split_static_and_jax_coords), [`test_jax_array_wrapper_with_numpy_api`](xarray_jax_test.md#XarrayJaxTest.test_jax_array_wrapper_with_numpy_api)

### `_HashableCoords`
- def: [`graphcast/xarray_jax.py:933`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L933)
- doc: Wraps a dict of xarray Variables as hashable, used for static coordinates.
- signature: `class _HashableCoords(collections.abc.Mapping):`
- protocol/private: `__eq__`[`L966`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L966), `__getitem__`[`L951`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L951), `__hash__`[`L960`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L960), `__init__`[`L945`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L945), `__iter__`[`L957`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L957), `__len__`[`L954`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L954), `__repr__`[`L948`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L948), `_hash`[`L962`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L962), `_variables`[`L946`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L946)
- used by: [`_unflatten_data_array`](xarray_jax.md#_unflatten_data_array), [`_unflatten_dataset`](xarray_jax.md#_unflatten_dataset), [`_flatten_data_array`](xarray_jax.md#_flatten_data_array), [`_flatten_dataset`](xarray_jax.md#_flatten_dataset)

## Functions
- `DataArray(data, coords=None, dims=None, name=None, attrs=None, jax_coords=None)` — [`L129`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L129) — Like xarray.DataArray, but supports using JAX arrays.
- `Dataset(data_vars=None, coords=None, attrs=None, jax_coords=None)` — [`L179`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L179) — Like xarray.Dataset, but can wrap JAX arrays.
- `Variable(dims, data, **kwargs)` — [`L121`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L121) — Like xarray.Variable, but can wrap JAX arrays.
- `_drop_with_none_of_dims(coord_vars: Mapping[Hashable, xarray.Variable], dims: Tuple[Hashable, ...])` — [`L926`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L926)
- `_flatten_data_array(v: xarray.DataArray)` — [`L979`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L979) — Flattens a DataArray for jax.tree_util.
- `_flatten_dataset(dataset: xarray.Dataset)` — [`L1011`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L1011) — Flattens a Dataset for jax.tree_util.
- `_flatten_datatree(datatree: xarray.DataTree)` — [`L1045`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L1045) — Flattens a DataTree for jax.tree_util.
- `_flatten_variable(v: xarray.Variable)` — [`L894`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L894) — Flattens a Variable for jax.tree_util.
- `_split_static_and_jax_coords(coords: xarray.core.coordinates.Coordinates)` — [`L912`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L912)
- `_unflatten_data_array(aux: Tuple[Optional[Hashable], _HashableCoords], children: Tuple[xarray.Variable, Mapping[Hashable, xarray.Variable]])` — [`L991`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L991) — Unflattens a DataArray for jax.tree_util.
- `_unflatten_dataset(aux: _HashableCoords, children: Tuple[Mapping[Hashable, xarray.Variable], Mapping[Hashable, xarray.Variable]])` — [`L1027`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L1027) — Unflattens a Dataset for jax.tree_util.
- `_unflatten_datatree(aux: str | None, children: Tuple[Mapping[str, xarray.DataTree], xarray.Dataset])` — [`L1057`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L1057) — Unflattens a DataTree for jax.tree_util.
- `_unflatten_variable(aux: Tuple[Hashable, ...], children: Tuple[jax.typing.ArrayLike])` — [`L902`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L902) — Unflattens a Variable for jax.tree_util.
- `_vmap_or_pmap(fn: Callable[..., Any], dim: str, axis_name: Optional[str] = None, devices=None, backend=None, is_vmap: bool = False)` — [`L606`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L606) — See pmap documentations.
- `_wrapped(func)` — [`L375`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L375) — Surrounds a function with JAX array unwrapping/wrapping.
- `apply_ufunc(func, *args, require_jax=False, **apply_ufunc_kwargs)` — [`L523`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L523) — Like xarray.apply_ufunc but for jax-specific ufuncs.
- `assign_coords(x: DatasetOrDataArray, *, coords: Optional[Mapping[Hashable, Any]] = None, jax_coords: Optional[Mapping[Hashable, Any]] = None)` — [`L244`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L244) — Replacement for assign_coords which works in presence of jax_coords.
- `assign_jax_coords(x: DatasetOrDataArray, jax_coords: Optional[Mapping[Hashable, Any]] = None, **jax_coords_kwargs)` — [`L346`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L346) — Assigns only jax_coords, with same API as xarray's assign_coords.
- `check_and_remove_leading_dim(dims)` — [`L622`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L622)
- `dims_change_on_unflatten(dims_change_fn: DimsChangeFn)` — [`L858`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L858) — Can be used to change the dims used when unflattening arrays into xarrays.
- `fn_passed_to_pmap(*flat_args)` — [`L619`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L619)
- `get_jax_coords(x: DatasetOrDataArray)` — [`L339`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L339)
- `is_leaf(x)` — [`L744`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L744)
- `jax_data(value: Union[xarray.Variable, xarray.DataArray])` — [`L412`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L412) — Like unwrap_data, but will complain if not a jax array.
- `jax_vars(dataset: Mapping[Hashable, xarray.DataArray])` — [`L419`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L419) — Like unwrap_vars, but will complain if vars are not all jax arrays.
- `pmap(fn: Callable[..., Any], dim: str, axis_name: Optional[str] = None, devices=None, backend=None)` — [`L557`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L557) — Wraps a subset of jax.pmap functionality to handle xarray input/output.
- `result_fn(*args)` — [`L654`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L654)
- `scan(f: Callable[[_Carry, _X], tuple[_Carry, _Y]], init: _Carry, dim: str, xs: _X | None = None, length: int | None = None, reverse: bool = False, unroll: int | bool = 1)` — [`L763`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L763) — Like jax.lax.scan but supports xarray data.
- `scan_fn(carry, x_leaves)` — [`L821`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L821)
- `tree_map_variables(func: Callable[[xarray.Variable], xarray.Variable], tree_data: _PyTree)` — [`L670`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L670) — Like jax.tree.map but operates with Variables as leaves.
- `tree_map_with_dims(func: Callable[[jax.typing.ArrayLike, tuple[str, ...] | None], jax.typing.ArrayLike], data: _PyTree)` — [`L701`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L701) — Like jax.tree.map but also passes in xarray dimensions where known.
- `unwrap(value, require_jax=False)` — [`L363`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L363) — Unwraps wrapped JAX arrays used in xarray, passing through other values.
- `unwrap_coords(dataset: Union[xarray.Dataset, xarray.DataArray], require_jax: bool = False)` — [`L403`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L403) — The unwrapped data (see unwrap) of the coords in a Dataset or DataArray.
- `unwrap_data(value: Union[xarray.Variable, xarray.DataArray], require_jax: bool = False)` — [`L384`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L384) — The unwrapped (see unwrap) data of a an xarray.Variable or DataArray.
- `unwrap_vars(dataset: Mapping[Hashable, xarray.DataArray], require_jax: bool = False)` — [`L392`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L392) — The unwrapped data (see unwrap) of the variables in a dataset.
- `vmap(fn: Callable[..., Any], dim: str, axis_name: Optional[str] = None)` — [`L595`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L595) — Similar to pmap, but for vmap.
- `wrap(value)` — [`L355`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L355) — Wraps JAX arrays for use in xarray, passing through other values.
- `wrapped_func(*args, **kwargs)` — [`L377`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L377)
- `wrapped_func(*maybe_wrapped_args)` — [`L549`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L549)
- `wrapped_func(x)` — [`L747`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L747)

## Module values
- `DatasetOrDataArray` — [`L240`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L240)
- `DimsChangeFn` — [`L852`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L852)
- `_Carry` — [`L758`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L758)
- `_DIMS_CHANGE_ON_UNFLATTEN_FN` — [`L853`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L853)
- `_JAX_COORD_ATTR_NAME` — [`L126`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L126)
- `_PyTree` — [`L667`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L667)
- `_WRAPPED_TYPES` — [`L117`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L117)
- `_X` — [`L759`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L759)
- `_Y` — [`L760`](../../../../../raw/code/graphcast/graphcast/xarray_jax.py#L760)

