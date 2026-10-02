---
title: 'Module: jax/_src/api_util.py'
type: catalog
provenance: extracted
module: jax/_src/api_util.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.api_util`/
symbols:
  debug_info: debug_info().
  flatten_axes: flatten_axes().
  flatten_fun_nokwargs: flatten_fun_nokwargs().
  _ensure_index_tuple: _ensure_index_tuple().
  _check_no_aliased_closed_over_refs: _check_no_aliased_closed_over_refs().
  argnums_partial: argnums_partial().
  flatten_axis_resources: flatten_axis_resources().
  check_no_aliased_ref_args: check_no_aliased_ref_args().
  resolve_argnums: resolve_argnums().
  check_no_transformed_refs_args: check_no_transformed_refs_args().
  infer_argnums_and_argnames: infer_argnums_and_argnames().
  check_callable: check_callable().
  argnums_partial2: argnums_partial2().
  flatten_fun: flatten_fun().
  _flat_out_axes: _flat_out_axes().
  fun_sourceinfo: fun_sourceinfo().
  prepend_static_args: prepend_static_args().
  flat_out_axes: flat_out_axes().
  _validate_argnames: _validate_argnames().
  resolve_kwargs: resolve_kwargs().
  fun_signature: fun_signature().
  _ensure_index: _ensure_index().
  donation_vector: donation_vector().
  _non_static_arg_names: _non_static_arg_names().
  _ensure_str_tuple: _ensure_str_tuple().
  save_wrapped_fun_debug_info: save_wrapped_fun_debug_info().
  _HashableWithStrictTypeEquality: _HashableWithStrictTypeEquality#
  InternalFloatingPointError.__init__: InternalFloatingPointError#__init__().
  _validate_argnums: _validate_argnums().
  maybe_recursive_nan_check: maybe_recursive_nan_check().
  InternalFloatingPointError: InternalFloatingPointError#
  InternalFloatingPointError.ty: InternalFloatingPointError#ty.
  _ensure_inbounds: _ensure_inbounds().
  _HashableWithStrictTypeEquality.__init__: _HashableWithStrictTypeEquality#__init__().
  _HashableWithStrictTypeEquality.__hash__: _HashableWithStrictTypeEquality#__hash__().
  _HashableWithStrictTypeEquality.__eq__: _HashableWithStrictTypeEquality#__eq__().
  _dtype: _dtype().
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  rebase_donate_argnums: rebase_donate_argnums().
  _POSITIONAL_OR_KEYWORD: _POSITIONAL_OR_KEYWORD.
  _raise_no_nan_in_deoptimized: _raise_no_nan_in_deoptimized().
  PytreeLeaf: PytreeLeaf#
  _ensure_str: _ensure_str().
  _POSITIONAL_ARGUMENTS: _POSITIONAL_ARGUMENTS.
  _INVALID_KEYWORD_ARGUMENTS: _INVALID_KEYWORD_ARGUMENTS.
  _KEYWORD_ARGUMENTS: _KEYWORD_ARGUMENTS.
  argnums_partial2.f_wrapped: argnums_partial2().f_wrapped().
  _prepend_static_args: _prepend_static_args().
  _argnums_partial: _argnums_partial().
  is_hashable: is_hashable().
  _assert_no_intersection: _assert_no_intersection().
  _fun_name_re: _fun_name_re.
  InternalFloatingPointError.name: InternalFloatingPointError#name.
  _HashableWithStrictTypeEquality.__slots__: _HashableWithStrictTypeEquality#__slots__.
  SENTINEL: SENTINEL.
  PytreeLeaf.__repr__: PytreeLeaf#__repr__().
  api_hook: api_hook().
---
# Module: [`jax/_src/api_util.py`](../../../../../../raw/code/jax/jax/_src/api_util.py)

## Classes
### `InternalFloatingPointError`  ·  implements/extends Exception
- def: [`jax/_src/api_util.py:759`](../../../../../../raw/code/jax/jax/_src/api_util.py#L759)
- signature: `class InternalFloatingPointError(Exception):`
- members:
  - `name` — [`L760`](../../../../../../raw/code/jax/jax/_src/api_util.py#L760)
  - `ty` — [`L761`](../../../../../../raw/code/jax/jax/_src/api_util.py#L761)
- protocol/private: `__init__`[`L763`](../../../../../../raw/code/jax/jax/_src/api_util.py#L763)
- used by: [`_run_python_pjit`](pjit.md#_run_python_pjit), [`_pjit_transpose_fancy`](pjit.md#_pjit_transpose_fancy), [`_nan_check_posthook`](api.md#_nan_check_posthook), [`_maybe_check_special`](shard_map.md#_maybe_check_special)

### `PytreeLeaf`
- def: [`jax/_src/api_util.py:374`](../../../../../../raw/code/jax/jax/_src/api_util.py#L374)
- protocol/private: `__repr__`[`L375`](../../../../../../raw/code/jax/jax/_src/api_util.py#L375)
- used by: [`flatten_axes`](api_util.md#flatten_axes), [`flatten_axis_resources`](api_util.md#flatten_axis_resources)

### `_HashableWithStrictTypeEquality`
- def: [`jax/_src/api_util.py:88`](../../../../../../raw/code/jax/jax/_src/api_util.py#L88)
- doc: Box object used when comparing static arguments as a jit key.
- signature: `class _HashableWithStrictTypeEquality:`
- protocol/private: `__eq__`[`L100`](../../../../../../raw/code/jax/jax/_src/api_util.py#L100), `__hash__`[`L97`](../../../../../../raw/code/jax/jax/_src/api_util.py#L97), `__init__`[`L94`](../../../../../../raw/code/jax/jax/_src/api_util.py#L94), `__slots__`[`L92`](../../../../../../raw/code/jax/jax/_src/api_util.py#L92)
- used by: [`argnums_partial`](api_util.md#argnums_partial)

## Functions
- `_argnums_partial(_fun: Callable, _dyn_argnums: Sequence[int], _fixed_args: Sequence, *dyn_args, **kwargs)` — [`L244`](../../../../../../raw/code/jax/jax/_src/api_util.py#L244)
- `_assert_no_intersection(static_argnames, donate_argnames)` — [`L553`](../../../../../../raw/code/jax/jax/_src/api_util.py#L553)
- `_check_no_aliased_closed_over_refs(dbg: core.DebugInfo, consts, args)` — [`L733`](../../../../../../raw/code/jax/jax/_src/api_util.py#L733)
- `_dtype(x)` — [`L582`](../../../../../../raw/code/jax/jax/_src/api_util.py#L582)
- `_ensure_inbounds(allow_invalid: bool, num_args: int, argnums: Sequence[int])` — [`L228`](../../../../../../raw/code/jax/jax/_src/api_util.py#L228) — Ensure argnum is within bounds. Also resolves negative argnums.
- `_ensure_index(x: Any)` — [`L42`](../../../../../../raw/code/jax/jax/_src/api_util.py#L42) — Ensure x is either an index or a tuple of indices.
- `_ensure_index_tuple(x: Any)` — [`L50`](../../../../../../raw/code/jax/jax/_src/api_util.py#L50) — Convert x to a tuple of indices.
- `_ensure_str(x: str)` — [`L58`](../../../../../../raw/code/jax/jax/_src/api_util.py#L58)
- `_ensure_str_tuple(x: str | Iterable[str])` — [`L63`](../../../../../../raw/code/jax/jax/_src/api_util.py#L63) — Convert x to a tuple of strings.
- `_flat_out_axes(_fun, _store, _leaves, _treedef, *args, **kwargs)` — [`L444`](../../../../../../raw/code/jax/jax/_src/api_util.py#L444)
- `_non_static_arg_names(fn_signature: inspect.Signature | None, args: Sequence[Any], kwargs: Mapping[str, Any], static_argnums: Sequence[int], static_argnames: Sequence[str])` — [`L664`](../../../../../../raw/code/jax/jax/_src/api_util.py#L664) — Returns the names of the non-static arguments.
- `_prepend_static_args(f, static_args, *args, **kwargs)` — [`L222`](../../../../../../raw/code/jax/jax/_src/api_util.py#L222)
- `_raise_no_nan_in_deoptimized(e)` — [`L780`](../../../../../../raw/code/jax/jax/_src/api_util.py#L780)
- `_validate_argnames(sig: inspect.Signature, argnames: tuple[str, ...], argnames_name: str)` — [`L138`](../../../../../../raw/code/jax/jax/_src/api_util.py#L138) — Validate that the argnames are sensible for a given function.
- `_validate_argnums(sig: inspect.Signature, argnums: tuple[int, ...], argnums_name: str)` — [`L108`](../../../../../../raw/code/jax/jax/_src/api_util.py#L108) — Validate that the argnums are sensible for a given function.
- `api_hook(fun, tag: str)` — [`L591`](../../../../../../raw/code/jax/jax/_src/api_util.py#L591)
- `argnums_partial(f: lu.WrappedFun, dyn_argnums: int | Sequence[int], args: Sequence, require_static_args_hashable=True)` — [`L176`](../../../../../../raw/code/jax/jax/_src/api_util.py#L176)
- `argnums_partial2(f: Callable, dyn_argnums: int | Sequence[int], args: Sequence, kwargs: dict)` — [`L197`](../../../../../../raw/code/jax/jax/_src/api_util.py#L197)
- `check_callable(fun)` — [`L460`](../../../../../../raw/code/jax/jax/_src/api_util.py#L460)
- `check_no_aliased_ref_args(dbg_fn: Callable[[], core.DebugInfo], maybe_avals, args)` — [`L716`](../../../../../../raw/code/jax/jax/_src/api_util.py#L716)
- `check_no_transformed_refs_args(dbg_fn: Callable[[], core.DebugInfo], args_flat)` — [`L746`](../../../../../../raw/code/jax/jax/_src/api_util.py#L746)
- `debug_info(traced_for: str, fun: Callable, args: Sequence[Any], kwargs: Mapping[str, Any], *, static_argnums: Sequence[int] = (), static_argnames: Sequence[str] = (), result_paths_thunk: Callable[[], tuple[str, ...]] | core.InitialResultPaths = core.initial_result_paths, sourceinfo: str | None = None, signature: inspect.Signature | None = None)` — [`L595`](../../../../../../raw/code/jax/jax/_src/api_util.py#L595) — Construct core.DebugInfo for a function given example args and kwargs.
- `donation_vector(donate_argnums, donate_argnames, in_tree, kws: bool = True)` — [`L258`](../../../../../../raw/code/jax/jax/_src/api_util.py#L258) — Returns a tuple with a boolean value for each leaf in args and kwargs.
- `f_wrapped(*dyn_args_)` — [`L209`](../../../../../../raw/code/jax/jax/_src/api_util.py#L209)
- `flat_out_axes(f: lu.WrappedFun, out_spec: Any)` — [`L436`](../../../../../../raw/code/jax/jax/_src/api_util.py#L436)
- `flatten_axes(name, treedef, axis_tree, *, kws=False, tupled_args=False)` — [`L332`](../../../../../../raw/code/jax/jax/_src/api_util.py#L332)
- `flatten_axis_resources(what, tree, shardings, tupled_args)` — [`L379`](../../../../../../raw/code/jax/jax/_src/api_util.py#L379)
- `flatten_fun(f: Callable, store: lu.Store, in_tree: PyTreeDef, *args_flat)` — [`L71`](../../../../../../raw/code/jax/jax/_src/api_util.py#L71)
- `flatten_fun_nokwargs(f: Callable, store: lu.Store, in_tree: PyTreeDef, *args_flat)` — [`L80`](../../../../../../raw/code/jax/jax/_src/api_util.py#L80)
- `fun_signature(fun: Callable)` — [`L629`](../../../../../../raw/code/jax/jax/_src/api_util.py#L629)
- `fun_sourceinfo(fun: Callable)` — [`L642`](../../../../../../raw/code/jax/jax/_src/api_util.py#L642)
- `infer_argnums_and_argnames(sig: inspect.Signature, argnums: int | Iterable[int] | None, argnames: str | Iterable[str] | None)` — [`L472`](../../../../../../raw/code/jax/jax/_src/api_util.py#L472) — Infer missing argnums and argnames for a function with inspect.
- `is_hashable(arg)` — [`L321`](../../../../../../raw/code/jax/jax/_src/api_util.py#L321)
- `maybe_recursive_nan_check(e: Exception, fun: Callable, args, kwargs)` — [`L767`](../../../../../../raw/code/jax/jax/_src/api_util.py#L767)
- `prepend_static_args(f, static_args)` — [`L217`](../../../../../../raw/code/jax/jax/_src/api_util.py#L217)
- `rebase_donate_argnums(donate_argnums, static_argnums)` — [`L286`](../../../../../../raw/code/jax/jax/_src/api_util.py#L286) — Shifts donate to account for static.
- `resolve_argnums(fun: Callable, signature: inspect.Signature | None, donate_argnums: int | Sequence[int] | None, donate_argnames: str | Iterable[str] | None, static_argnums: int | Sequence[int] | None, static_argnames: str | Iterable[str] | None)` — [`L504`](../../../../../../raw/code/jax/jax/_src/api_util.py#L504) — Validates and completes the argnum/argname specification for a jit.
- `resolve_kwargs(fun: Callable, args, kwargs)` — [`L561`](../../../../../../raw/code/jax/jax/_src/api_util.py#L561) — Resolve input arguments to positional following a function's signature.
- `save_wrapped_fun_debug_info(wrapper: Callable, dbg: core.DebugInfo)` — [`L635`](../../../../../../raw/code/jax/jax/_src/api_util.py#L635)

## Module values
- `SENTINEL` — [`L329`](../../../../../../raw/code/jax/jax/_src/api_util.py#L329)
- `_INVALID_KEYWORD_ARGUMENTS` — [`L128`](../../../../../../raw/code/jax/jax/_src/api_util.py#L128)
- `_KEYWORD_ARGUMENTS` — [`L134`](../../../../../../raw/code/jax/jax/_src/api_util.py#L134)
- `_POSITIONAL_ARGUMENTS` — [`L103`](../../../../../../raw/code/jax/jax/_src/api_util.py#L103)
- `_POSITIONAL_OR_KEYWORD` — [`L470`](../../../../../../raw/code/jax/jax/_src/api_util.py#L470)
- `_fun_name_re` — [`L639`](../../../../../../raw/code/jax/jax/_src/api_util.py#L639)
- `unsafe_map` — [`L39`](../../../../../../raw/code/jax/jax/_src/api_util.py#L39)
- `unsafe_zip` — [`L40`](../../../../../../raw/code/jax/jax/_src/api_util.py#L40)

