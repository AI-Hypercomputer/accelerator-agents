---
title: 'Module: jax/_src/numpy/ufunc_api.py'
type: catalog
provenance: extracted
module: jax/_src/numpy/ufunc_api.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.numpy.ufunc_api`/
symbols:
  ufunc._reduce_via_scan: ufunc#_reduce_via_scan().
  ufunc._accumulate_via_scan: ufunc#_accumulate_via_scan().
  ufunc._at_via_scan: ufunc#_at_via_scan().
  ufunc._reduceat_via_scan: ufunc#_reduceat_via_scan().
  ufunc.reduce: ufunc#reduce().
  ufunc.accumulate: ufunc#accumulate().
  ufunc.reduceat: ufunc#reduceat().
  ufunc.outer: ufunc#outer().
  ufunc.__eq__: ufunc#__eq__().
  ufunc.at: ufunc#at().
  ufunc.__hash__: ufunc#__hash__().
  ufunc: ufunc#
  ufunc.__call__: ufunc#__call__().
  ufunc.nin: ufunc#nin.
  ufunc.nout: ufunc#nout.
  ufunc._call_vectorized: ufunc#_call_vectorized().
  ufunc.loop_body: ufunc#loop_body().
  ufunc._func: ufunc#_func.
  ufunc.scan_fun: ufunc#scan_fun().
  ufunc.body_fun: ufunc#body_fun().
  ufunc.__static_props: ufunc#__static_props.
  ufunc.identity: ufunc#identity.
  frompyfunc: frompyfunc().
  export: export.
  _AT_INPLACE_WARNING: _AT_INPLACE_WARNING.
  ufunc.nargs: ufunc#nargs.
  ufunc.__init__: ufunc#__init__().
  ufunc.__repr__: ufunc#__repr__().
---
# Module: [`jax/_src/numpy/ufunc_api.py`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py)

## Classes
### `ufunc`
- def: [`jax/_src/numpy/ufunc_api.py:49`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L49)
- doc: Universal functions which operation element-by-element on arrays.
- signature: `class ufunc:`
- members:
  - `accumulate(self, a: ArrayLike, axis: int = 0, dtype: DTypeLike | None = None, out: None = None)` — [`L322`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L322) — Accumulate operation derived from binary ufunc.
  - `at(self, a: ArrayLike, indices: Any, b: ArrayLike | None = None,, *, inplace: bool = True)` — [`L403`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L403) — Update elements of an array via the specified unary or binary ufunc.
  - `body_fun(i, val)` — [`L300`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L300)
  - `loop_body(i, out)` — [`L544`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L544)
  - `outer(self, A: ArrayLike, B: ArrayLike,)` — [`L551`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L551) — Apply the function to all pairs of values in ``A`` and ``B``.
  - `reduce(self, a: ArrayLike, axis: int | None = 0, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L189`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L189) — Reduction operation derived from a binary function.
  - `reduceat(self, a: ArrayLike, indices: Any, axis: int = 0, dtype: DTypeLike | None = None, out: None = None)` — [`L475`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L475) — Reduce an array between specified indices via a binary ufunc.
  - `scan_fun(carry, _)` — [`L395`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L395)
  - `identity` — [`L158`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L158)
  - `nargs` — [`L157`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L157)
  - `nin` — [`L155`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L155)
  - `nout` — [`L156`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L156)
- protocol/private: `__call__`[`L175`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L175), `__eq__`[`L167`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L167), `__hash__`[`L160`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L160), `__init__`[`L124`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L124), `__repr__`[`L172`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L172), `__static_props`[`L141`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L141), `_accumulate_via_scan`[`L379`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L379), `_at_via_scan`[`L446`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L446), `_call_vectorized`[`L185`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L185), `_func`[`L154`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L154), `_reduce_via_scan`[`L259`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L259), `_reduceat_via_scan`[`L523`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L523)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array), [`ArrayLike`](../basearray.md#ArrayLike), [`shape`](../basearray.md#Array.shape), [`jit`](../api.md#jit), [`dtype`](../basearray.md#Array.dtype), [`DTypeLike`](../typing.md#DTypeLike), [`astype`](../basearray.md#Array.astype), [`ndim`](../basearray.md#Array.ndim), [`scan`](../lax/control_flow/loops.md#scan), [`canonicalize_axis`](../util.md#canonicalize_axis), [`size`](../basearray.md#Array.size), [`full`](../lax/lax.md#full), [`fori_loop`](../lax/control_flow/loops.md#fori_loop), [`_where`](util.md#_where), [`vmap`](../api.md#vmap), [`reshape`](../basearray.md#Array.reshape), [`check_arraylike`](util.md#check_arraylike), [`expand_dims`](../lax/lax.md#expand_dims), [`reshape`](../lax/lax.md#reshape), [`asarray`](../lax/lax.md#asarray), [`at`](../basearray.md#Array.at), [`slice_in_dim`](../lax/slicing.md#slice_in_dim), [`_broadcast_to`](util.md#_broadcast_to), [`ravel`](../basearray.md#Array.ravel), [`_dtype`](../lax/lax.md#_dtype._dtype), [`broadcast_shapes`](../lax/lax.md#broadcast_shapes), [`_one`](../lax/lax.md#_one), [`take`](indexing.md#take), [`append`](lax_numpy.md#append), [`_moveaxis`](reductions.md#_moveaxis), [`get`](../basearray.md#_IndexUpdateRef.get), [`eval_shape`](../api.md#eval_shape), [`vectorize`](vectorize.md#vectorize), [`eliminate_deprecated_list_indexing`](indexing.md#eliminate_deprecated_list_indexing), [`_AT_INPLACE_WARNING`](ufunc_api.md#_AT_INPLACE_WARNING)
- used by: [`binary_ufunc`](ufuncs.md#binary_ufunc), [`decorator`](ufuncs.md#binary_ufunc.decorator), [`unary_ufunc`](ufuncs.md#unary_ufunc), [`frompyfunc`](ufunc_api.md#frompyfunc)

## Functions
- `frompyfunc(func: Callable[..., Any],, nin: int, nout: int, *, identity: Any = None)` — [`L599`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L599) — Create a JAX ufunc from an arbitrary JAX-compatible scalar function.

## Module values
- `_AT_INPLACE_WARNING` — [`L41`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L41)
- `export` — [`L39`](../../../../../../../raw/code/jax/jax/_src/numpy/ufunc_api.py#L39)

