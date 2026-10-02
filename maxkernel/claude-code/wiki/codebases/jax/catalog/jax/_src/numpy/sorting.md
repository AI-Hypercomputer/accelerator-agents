---
title: 'Module: jax/_src/numpy/sorting.py'
type: catalog
provenance: extracted
module: jax/_src/numpy/sorting.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.numpy.sorting`/
symbols:
  argsort: argsort().
  argpartition: argpartition().
  partition: partition().
  lexsort: lexsort().
  sort: sort().
  sort_complex: sort_complex().
  export: export.
---
# Module: [`jax/_src/numpy/sorting.py`](../../../../../../../raw/code/jax/jax/_src/numpy/sorting.py)

## Functions
- `argpartition(a: ArrayLike, kth: int, axis: int = -1)` — [`L252`](../../../../../../../raw/code/jax/jax/_src/numpy/sorting.py#L252) — Returns indices that partially sort an array.
- `argsort(a: ArrayLike, axis: int | None = -1, *, kind: None = None, order: None = None, stable: bool = True, descending: bool = False, dtype: DTypeLike | None = None)` — [`L93`](../../../../../../../raw/code/jax/jax/_src/numpy/sorting.py#L93) — Return indices that sort an array.
- `lexsort(keys: Array | np.ndarray | Sequence[ArrayLike], axis: int = -1)` — [`L367`](../../../../../../../raw/code/jax/jax/_src/numpy/sorting.py#L367) — Sort a sequence of keys in lexicographic order.
- `partition(a: ArrayLike, kth: int, axis: int = -1)` — [`L182`](../../../../../../../raw/code/jax/jax/_src/numpy/sorting.py#L182) — Returns a partially-sorted copy of an array.
- `sort(a: ArrayLike, axis: int | None = -1, *, kind: None = None, order: None = None, stable: bool = True, descending: bool = False)` — [`L32`](../../../../../../../raw/code/jax/jax/_src/numpy/sorting.py#L32) — Return a sorted copy of an array.
- `sort_complex(a: ArrayLike)` — [`L329`](../../../../../../../raw/code/jax/jax/_src/numpy/sorting.py#L329) — Return a sorted copy of complex array.

## Module values
- `export` — [`L28`](../../../../../../../raw/code/jax/jax/_src/numpy/sorting.py#L28)

