---
title: 'Module: jax/_src/numpy/array_creation.py'
type: catalog
provenance: extracted
module: jax/_src/numpy/array_creation.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.numpy.array_creation`/
symbols:
  _linspace: _linspace().
  full: full().
  full_like: full_like().
  zeros: zeros().
  ones: ones().
  zeros_like: zeros_like().
  _geomspace: _geomspace().
  ones_like: ones_like().
  empty: empty().
  _logspace: _logspace().
  empty_like: empty_like().
  linspace: linspace().
  logspace: logspace().
  geomspace: geomspace().
  export: export.
  canonicalize_shape: canonicalize_shape().
  _check_forgot_shape_tuple: _check_forgot_shape_tuple().
---
# Module: [`jax/_src/numpy/array_creation.py`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py)

## Functions
- `_check_forgot_shape_tuple(name, shape, dtype)` — [`L201`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L201)
- `_geomspace(start: ArrayLike, stop: ArrayLike, num: int = 50, endpoint: bool = True, dtype: DTypeLike | None = None, axis: int = 0)` — [`L773`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L773) — Implementation of geomspace differentiable in start and stop args.
- `_linspace(start: ArrayLike, stop: ArrayLike, num: int = 50, endpoint: bool = True, retstep: bool = False, dtype: DTypeLike | None = None, axis: int = 0, *, device: xc.Device | Sharding | None = None)` — [`L576`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L576) — Implementation of linspace differentiable in start and stop args.
- `_logspace(start: ArrayLike, stop: ArrayLike, num: int = 50, endpoint: bool = True, base: ArrayLike = 10, dtype: DTypeLike | None = None, axis: int = 0)` — [`L702`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L702) — Implementation of logspace differentiable in start and stop args.
- `canonicalize_shape(shape: Any, context: str = "")` — [`L41`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L41)
- `empty(shape: Any, dtype: DTypeLike | None = None, *, device: xc.Device | Sharding | None = None, out_sharding: NamedSharding | P | None = None)` — [`L148`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L148) — Create an empty array.
- `empty_like(prototype: ArrayLike | DuckTypedArray, dtype: DTypeLike | None = None, shape: Any = None, *, device: xc.Device | Sharding | None = None)` — [`L367`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L367) — Create an empty array with the same shape and dtype as an array.
- `full(shape: Any, fill_value: ArrayLike, dtype: DTypeLike | None = None, *, device: xc.Device | Sharding | None = None, out_sharding: NamedSharding | P | None = None)` — [`L210`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L210) — Create an array full of a specified value.
- `full_like(a: ArrayLike | DuckTypedArray, fill_value: ArrayLike, dtype: DTypeLike | None = None, shape: Any = None, *, device: xc.Device | Sharding | None = None, out_sharding: NamedSharding | P | None = None)` — [`L417`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L417) — Create an array full of a specified value with the same shape and dtype as an array.
- `geomspace(start: ArrayLike, stop: ArrayLike, num: int = 50, endpoint: bool = True, dtype: DTypeLike | None = None, axis: int = 0)` — [`L720`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L720) — Generate geometrically-spaced values.
- `linspace(start: ArrayLike, stop: ArrayLike, num: int = 50, endpoint: bool = True, retstep: Literal[False] = False, dtype: DTypeLike | None = None, axis: int = 0, *, device: xc.Device | Sharding | None = None)` — [`L484`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L484)
- `logspace(start: ArrayLike, stop: ArrayLike, num: int = 50, endpoint: bool = True, base: ArrayLike = 10, dtype: DTypeLike | None = None, axis: int = 0)` — [`L634`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L634) — Generate logarithmically-spaced values.
- `ones(shape: Any, dtype: DTypeLike | None = None, *, device: xc.Device | Sharding | None = None, out_sharding: NamedSharding | P | None = None)` — [`L99`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L99) — Create an array full of ones.
- `ones_like(a: ArrayLike | DuckTypedArray, dtype: DTypeLike | None = None, shape: Any = None, *, device: xc.Device | Sharding | None = None, out_sharding: NamedSharding | P | None = None)` — [`L317`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L317) — Create an array of ones with the same shape and dtype as an array.
- `zeros(shape: Any, dtype: DTypeLike | None = None, *, device: xc.Device | Sharding | None = None, out_sharding: NamedSharding | P | None = None)` — [`L50`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L50) — Create an array full of zeros.
- `zeros_like(a: ArrayLike | DuckTypedArray, dtype: DTypeLike | None = None, shape: Any = None, *, device: xc.Device | Sharding | None = None, out_sharding: NamedSharding | P | None = None)` — [`L267`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L267) — Create an array full of zeros with the same shape and dtype as an array.

## Module values
- `export` — [`L36`](../../../../../../../raw/code/jax/jax/_src/numpy/array_creation.py#L36)

