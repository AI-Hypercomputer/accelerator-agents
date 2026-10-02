---
title: 'Module: jax/_src/numpy/setops.py'
type: catalog
provenance: extracted
module: jax/_src/numpy/setops.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.numpy.setops`/
symbols:
  _unique: _unique().
  _intersect1d_size: _intersect1d_size().
  _in1d: _in1d().
  _setxor1d_size: _setxor1d_size().
  _unique_sorted_mask: _unique_sorted_mask().
  unique: unique().
  setdiff1d: setdiff1d().
  setxor1d: setxor1d().
  intersect1d: intersect1d().
  _intersect1d_sorted_mask: _intersect1d_sorted_mask().
  unique_all: unique_all().
  _concat_unique: _concat_unique().
  union1d: union1d().
  unique_counts: unique_counts().
  unique_inverse: unique_inverse().
  export: export.
  isin: isin().
  unique_values: unique_values().
  UNIQUE_SIZE_HINT: UNIQUE_SIZE_HINT.
  _UniqueInverseResult: _UniqueInverseResult#
  _UniqueAllResult.values: _UniqueAllResult#values.
  _UniqueAllResult.indices: _UniqueAllResult#indices.
  _UniqueAllResult.inverse_indices: _UniqueAllResult#inverse_indices.
  _UniqueAllResult.counts: _UniqueAllResult#counts.
  _UniqueCountsResult.values: _UniqueCountsResult#values.
  _UniqueCountsResult.counts: _UniqueCountsResult#counts.
  _UniqueInverseResult.values: _UniqueInverseResult#values.
  _UniqueInverseResult.inverse_indices: _UniqueInverseResult#inverse_indices.
  _UniqueAllResult: _UniqueAllResult#
  _UniqueCountsResult: _UniqueCountsResult#
---
# Module: [`jax/_src/numpy/setops.py`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py)

## Classes
### `_UniqueAllResult`  ·  implements/extends NamedTuple
- def: [`jax/_src/numpy/setops.py:864`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L864)
- doc: Struct returned by `jax.numpy.unique_all`.
- signature: `class _UniqueAllResult(NamedTuple):`
- members:
  - `counts` — [`L869`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L869)
  - `indices` — [`L867`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L867)
  - `inverse_indices` — [`L868`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L868)
  - `values` — [`L866`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L866)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array)
- used by: [`unique_all`](setops.md#unique_all)

### `_UniqueCountsResult`  ·  implements/extends NamedTuple
- def: [`jax/_src/numpy/setops.py:872`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L872)
- doc: Struct returned by `jax.numpy.unique_counts`.
- signature: `class _UniqueCountsResult(NamedTuple):`
- members:
  - `counts` — [`L875`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L875)
  - `values` — [`L874`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L874)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array)
- used by: [`unique_counts`](setops.md#unique_counts)

### `_UniqueInverseResult`  ·  implements/extends NamedTuple
- def: [`jax/_src/numpy/setops.py:878`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L878)
- doc: Struct returned by `jax.numpy.unique_inverse`.
- signature: `class _UniqueInverseResult(NamedTuple):`
- members:
  - `inverse_indices` — [`L881`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L881)
  - `values` — [`L880`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L880)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array), [`export`](setops.md#export)
- used by: [`unique_inverse`](setops.md#unique_inverse)

## Functions
- `_concat_unique(arr1: Array, arr2: Array)` — [`L81`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L81) — Utility to concatenate the unique values from two arrays.
- `_in1d(ar1: ArrayLike, ar2: ArrayLike, invert: bool, method='auto', assume_unique=False)` — [`L45`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L45)
- `_intersect1d_size(arr1: Array, arr2: Array, fill_value: ArrayLike | None, assume_unique: bool, size: int, return_indices: bool)` — [`L361`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L361) — Jit-compatible helper function for intersect1d with size specified.
- `_intersect1d_sorted_mask(arr1: Array, arr2: Array, return_indices: bool)` — [`L344`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L344) — JIT-compatible helper function for intersect1d
- `_setxor1d_size(arr1: Array, arr2: Array, fill_value: ArrayLike | None, *, assume_unique: bool, size: int)` — [`L257`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L257)
- `_unique(ar: Array, axis: int, return_index: bool = False, return_inverse: bool = False, return_counts: bool = False, equal_nan: bool = True, size: int | None = None, fill_value: ArrayLike | None = None, return_true_size: bool = False)` — [`L606`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L606) — Find the unique elements of an array along a particular axis.
- `_unique_sorted_mask(ar: Array, axis: int, equal_nan: bool)` — [`L580`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L580)
- `intersect1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False, return_indices: bool = False, *, size: int | None = None, fill_value: ArrayLike | None = None)` — [`L430`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L430) — Compute the set intersection of two 1D arrays.
- `isin(element: ArrayLike, test_elements: ArrayLike, assume_unique: bool = False, invert: bool = False, *, method='auto')` — [`L539`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L539) — Determine whether elements in ``element`` appear in ``test_elements``.
- `setdiff1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False, *, size: int | None = None, fill_value: ArrayLike | None = None)` — [`L93`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L93) — Compute the set difference of two 1D arrays.
- `setxor1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False, *, size: int | None = None, fill_value: ArrayLike | None = None)` — [`L288`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L288) — Compute the set-wise xor of elements in two arrays.
- `union1d(ar1: ArrayLike, ar2: ArrayLike, *, size: int | None = None, fill_value: ArrayLike | None = None)` — [`L184`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L184) — Compute the set union of two 1D arrays.
- `unique(ar: ArrayLike, return_index: bool = False, return_inverse: bool = False, return_counts: bool = False, axis: int | None = None, *, equal_nan: bool = True, size: int | None = None, fill_value: ArrayLike | None = None, sorted: bool = True)` — [`L668`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L668) — Return the unique values from an array.
- `unique_all(x: ArrayLike,, *, size: int | None = None, fill_value: ArrayLike | None = None)` — [`L885`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L885) — Return unique values from x, along with indices, inverse indices, and counts.
- `unique_counts(x: ArrayLike,, *, size: int | None = None, fill_value: ArrayLike | None = None)` — [`L968`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L968) — Return unique values from x, along with counts.
- `unique_inverse(x: ArrayLike,, *, size: int | None = None, fill_value: ArrayLike | None = None)` — [`L1029`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L1029) — Return unique values from x, along with indices, inverse indices, and counts.
- `unique_values(x: ArrayLike,, *, size: int | None = None, fill_value: ArrayLike | None = None)` — [`L1095`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L1095) — Return unique values from x, along with indices, inverse indices, and counts.

## Module values
- `UNIQUE_SIZE_HINT` — [`L575`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L575)
- `export` — [`L41`](../../../../../../../raw/code/jax/jax/_src/numpy/setops.py#L41)

