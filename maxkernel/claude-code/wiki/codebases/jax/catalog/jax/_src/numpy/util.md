---
title: 'Module: jax/_src/numpy/util.py'
type: catalog
provenance: extracted
module: jax/_src/numpy/util.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.numpy.util`/
symbols:
  ensure_arraylike: ensure_arraylike().
  _where: _where().
  promote_args_inexact: promote_args_inexact().
  check_arraylike: check_arraylike().
  promote_args: promote_args().
  promote_dtypes_inexact: promote_dtypes_inexact().
  _broadcast_to: _broadcast_to().
  promote_dtypes: promote_dtypes().
  promote_args_numeric: promote_args_numeric().
  _broadcast_arrays: _broadcast_arrays().
  promote_shapes: promote_shapes().
  choose_device_or_out_sharding: choose_device_or_out_sharding().
  promote_args_complex: promote_args_complex().
  size: size().
  promote_dtypes_numeric: promote_dtypes_numeric().
  ensure_arraylike_tuple: ensure_arraylike_tuple().
  promote_dtypes_complex: promote_dtypes_complex().
  shape: shape().
  ndim: ndim().
  check_for_prngkeys: check_for_prngkeys().
  _arraylike: _arraylike().
  canonicalize_device_to_sharding: canonicalize_device_to_sharding().
  _arraylike_asarray: _arraylike_asarray().
  check_no_float0s: check_no_float0s().
  _check_no_float0s: _check_no_float0s.
  _rank_promotion_warning_or_error: _rank_promotion_warning_or_error().
  export: export.
  check_arraylike_or_none: check_arraylike_or_none().
  _check_jax_array_protocol: _check_jax_array_protocol().
  _dtype: _dtype.
  _arraylike_types: _arraylike_types.
  unsafe_zip: unsafe_zip.
  unsafe_map: unsafe_map.
---
# Module: [`jax/_src/numpy/util.py`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py)

## Functions
- `_arraylike(x: ArrayLike)` — [`L120`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L120)
- `_arraylike_asarray(x: Any)` — [`L125`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L125) — Convert an array-like object to an array.
- `_broadcast_arrays(*args: ArrayLike)` — [`L263`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L263) — Like Numpy's broadcast_arrays but doesn't return views.
- `_broadcast_to(arr: ArrayLike, shape: DimSize | Shape, sharding=None)` — [`L274`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L274)
- `_check_jax_array_protocol(x: Any)` — [`L133`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L133)
- `_rank_promotion_warning_or_error(fun_name: str, shapes: Sequence[Shape])` — [`L63`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L63)
- `_where(condition: ArrayLike, x: ArrayLike, y: ArrayLike)` — [`L289`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L289)
- `canonicalize_device_to_sharding(device: xc.Device | Sharding | None)` — [`L309`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L309)
- `check_arraylike(fun_name: str, *args: Any, emit_warning=False, stacklevel=3)` — [`L171`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L171) — Check if all args fit JAX's definition of arraylike.
- `check_arraylike_or_none(fun_name: str, *args: Any)` — [`L185`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L185)
- `check_for_prngkeys(fun_name: str, *args: Any)` — [`L208`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L208) — Check if args don't match and none of the args have typed prng dtype
- `check_no_float0s(fun_name: str, *args: Any)` — [`L194`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L194) — Check if none of the args have dtype float0.
- `choose_device_or_out_sharding(device: xc.Device | Sharding | None, out_sharding: NamedSharding | P | None, name: str)` — [`L315`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L315)
- `ensure_arraylike(fun_name: str,)` — [`L139`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L139)
- `ensure_arraylike_tuple(fun_name: str, tup: Sequence[Any])` — [`L162`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L162) — Check that argument elements are arraylike and convert to a tuple of arrays.
- `ndim(a: ArrayLike | SupportsNdim)` — [`L330`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L330) — Return the number of dimensions of an array.
- `promote_args(fun_name: str, *args: ArrayLike)` — [`L223`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L223) — Convenience function to apply Numpy argument shape and dtype promotion.
- `promote_args_complex(fun_name: str, *args: ArrayLike)` — [`L251`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L251) — Convenience function to apply Numpy argument shape and dtype promotion.
- `promote_args_inexact(fun_name: str, *args: ArrayLike)` — [`L240`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L240) — Convenience function to apply Numpy argument shape and dtype promotion.
- `promote_args_numeric(fun_name: str, *args: ArrayLike)` — [`L232`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L232)
- `promote_dtypes(*args: ArrayLike)` — [`L78`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L78) — Convenience function to apply Numpy argument dtype promotion.
- `promote_dtypes_complex(*args: ArrayLike)` — [`L108`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L108) — Convenience function to apply Numpy argument dtype promotion.
- `promote_dtypes_inexact(*args: ArrayLike)` — [`L88`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L88) — Convenience function to apply Numpy argument dtype promotion.
- `promote_dtypes_numeric(*args: ArrayLike)` — [`L98`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L98) — Convenience function to apply Numpy argument dtype promotion.
- `promote_shapes(fun_name: str, *args: ArrayLike)` — [`L45`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L45) — Apply NumPy-style broadcasting, making args shape-compatible for lax.py.
- `shape(a: ArrayLike | SupportsShape)` — [`L375`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L375) — Return the shape an array.
- `size(a: ArrayLike | SupportsSize | SupportsShape, axis: int | Sequence[int] | None = None)` — [`L420`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L420) — Return number of elements along a given axis.

## Module values
- `_arraylike_types` — [`L118`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L118)
- `_check_no_float0s` — [`L205`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L205)
- `_dtype` — [`L42`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L42)
- `export` — [`L40`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L40)
- `unsafe_map` — [`L38`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L38)
- `unsafe_zip` — [`L37`](../../../../../../../raw/code/jax/jax/_src/numpy/util.py#L37)

