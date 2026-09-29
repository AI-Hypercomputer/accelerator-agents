---
title: 'Module: simply/utils/masked.py'
type: catalog
provenance: extracted
module: simply/utils/masked.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.masked`/
symbols:
  Array: Array.
  masked: masked().
  masked_mean: masked_mean().
  masked_mean_var: masked_mean_var().
  masked_mean_std: masked_mean_std().
  masked_var: masked_var().
  masked_std: masked_std().
  masked_max: masked_max().
  masked_min: masked_min().
  masked_sum: masked_sum().
  _EPS: _EPS.
---
# Module: [`simply/utils/masked.py`](../../../../../../raw/code/simply/simply/utils/masked.py)

## Functions
- `masked(x: jax.Array, mask: jax.Array, padding_value: float = 0)` — [`L26`](../../../../../../raw/code/simply/simply/utils/masked.py#L26)
- `masked_max(x: Array, mask: Array, axis: int | Sequence[int] | None = None, keepdims: bool = False)` — [`L32`](../../../../../../raw/code/simply/simply/utils/masked.py#L32)
- `masked_mean(x: Array, mask: Array, axis: int | Sequence[int] | None = None, keepdims: bool = False)` — [`L71`](../../../../../../raw/code/simply/simply/utils/masked.py#L71)
- `masked_mean_std(x: Array, mask: Array, axis: int | Sequence[int] | None = None, ddof: int = 0, keepdims: bool = False)` — [`L120`](../../../../../../raw/code/simply/simply/utils/masked.py#L120)
- `masked_mean_var(x: Array, mask: Array, axis: int | Sequence[int] | None = None, ddof: int = 0, keepdims: bool = False)` — [`L104`](../../../../../../raw/code/simply/simply/utils/masked.py#L104) — Computes the mean and variance of a masked tensor.
- `masked_min(x: Array, mask: Array, axis: int | Sequence[int] | None = None, keepdims: bool = False)` — [`L47`](../../../../../../raw/code/simply/simply/utils/masked.py#L47)
- `masked_std(x: Array, mask: Array, axis: int | Sequence[int] | None = None, ddof: int = 0, keepdims: bool = False)` — [`L93`](../../../../../../raw/code/simply/simply/utils/masked.py#L93)
- `masked_sum(x: Array, mask: Array, axis: int | Sequence[int] | None = None, keepdims: bool = False)` — [`L62`](../../../../../../raw/code/simply/simply/utils/masked.py#L62)
- `masked_var(x: Array, mask: Array, axis: int | Sequence[int] | None = None, ddof: int = 0, keepdims: bool = False)` — [`L82`](../../../../../../raw/code/simply/simply/utils/masked.py#L82)

## Module values
- `Array` — [`L21`](../../../../../../raw/code/simply/simply/utils/masked.py#L21)
- `_EPS` — [`L23`](../../../../../../raw/code/simply/simply/utils/masked.py#L23)

