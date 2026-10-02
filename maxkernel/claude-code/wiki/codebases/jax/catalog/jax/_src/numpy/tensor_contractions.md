---
title: 'Module: jax/_src/numpy/tensor_contractions.py'
type: catalog
provenance: extracted
module: jax/_src/numpy/tensor_contractions.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.numpy.tensor_contractions`/
symbols:
  matmul: matmul().
  tensordot: tensordot().
  vdot: vdot().
  dot: dot().
  vecdot: vecdot().
  inner: inner().
  vecmat: vecmat().
  outer: outer().
  matvec: matvec().
  export: export.
---
# Module: [`jax/_src/numpy/tensor_contractions.py`](../../../../../../../raw/code/jax/jax/_src/numpy/tensor_contractions.py)

## Functions
- `dot(a: ArrayLike, b: ArrayLike, *, precision: lax.PrecisionLike = None, preferred_element_type: DTypeLike | None = None, out_sharding=None)` — [`L39`](../../../../../../../raw/code/jax/jax/_src/numpy/tensor_contractions.py#L39) — Compute the dot product of two arrays.
- `inner(a: ArrayLike, b: ArrayLike, *, precision: lax.PrecisionLike = None, preferred_element_type: DTypeLike | None = None)` — [`L591`](../../../../../../../raw/code/jax/jax/_src/numpy/tensor_contractions.py#L591) — Compute the inner product of two arrays.
- `matmul(a: ArrayLike, b: ArrayLike, *, precision: lax.PrecisionLike = None, preferred_element_type: DTypeLike | None = None, out_sharding: NamedSharding | P | None = None)` — [`L136`](../../../../../../../raw/code/jax/jax/_src/numpy/tensor_contractions.py#L136) — Perform a matrix multiplication.
- `matvec(x1: ArrayLike, x2: ArrayLike,)` — [`L281`](../../../../../../../raw/code/jax/jax/_src/numpy/tensor_contractions.py#L281) — Batched matrix-vector product.
- `outer(a: ArrayLike, b: ArrayLike, out: None = None)` — [`L649`](../../../../../../../raw/code/jax/jax/_src/numpy/tensor_contractions.py#L649) — Compute the outer product of two arrays.
- `tensordot(a: ArrayLike, b: ArrayLike, axes: int | Sequence[int] | Sequence[Sequence[int]] = 2, *, precision: lax.PrecisionLike = None, preferred_element_type: DTypeLike | None = None, out_sharding: NamedSharding | P | None = None)` — [`L470`](../../../../../../../raw/code/jax/jax/_src/numpy/tensor_contractions.py#L470) — Compute the tensor dot product of two N-dimensional arrays.
- `vdot(a: ArrayLike, b: ArrayLike, *, precision: lax.PrecisionLike = None, preferred_element_type: DTypeLike | None = None)` — [`L364`](../../../../../../../raw/code/jax/jax/_src/numpy/tensor_contractions.py#L364) — Perform a conjugate multiplication of two 1D vectors.
- `vecdot(x1: ArrayLike, x2: ArrayLike,, *, axis: int = -1, precision: lax.PrecisionLike = None, preferred_element_type: DTypeLike | None = None)` — [`L412`](../../../../../../../raw/code/jax/jax/_src/numpy/tensor_contractions.py#L412) — Perform a conjugate multiplication of two batched vectors.
- `vecmat(x1: ArrayLike, x2: ArrayLike,)` — [`L322`](../../../../../../../raw/code/jax/jax/_src/numpy/tensor_contractions.py#L322) — Batched conjugate vector-matrix product.

## Module values
- `export` — [`L34`](../../../../../../../raw/code/jax/jax/_src/numpy/tensor_contractions.py#L34)

