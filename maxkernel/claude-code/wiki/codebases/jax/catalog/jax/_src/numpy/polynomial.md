---
title: 'Module: jax/_src/numpy/polynomial.py'
type: catalog
provenance: extracted
module: jax/_src/numpy/polynomial.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.numpy.polynomial`/
symbols:
  polyfit: polyfit().
  polyint: polyint().
  roots: roots().
  polydiv: polydiv().
  polyval: polyval().
  poly: poly().
  polyder: polyder().
  polyadd: polyadd().
  polymul: polymul().
  _roots_no_zeros: _roots_no_zeros().
  _roots_with_zeros: _roots_with_zeros().
  polysub: polysub().
  export: export.
---
# Module: [`jax/_src/numpy/polynomial.py`](../../../../../../../raw/code/jax/jax/_src/numpy/polynomial.py)

## Functions
- `_roots_no_zeros(p: Array)` — [`L44`](../../../../../../../raw/code/jax/jax/_src/numpy/polynomial.py#L44)
- `_roots_with_zeros(p: Array, num_leading_zeros: Array | int)` — [`L54`](../../../../../../../raw/code/jax/jax/_src/numpy/polynomial.py#L54)
- `poly(seq_of_zeros: ArrayLike)` — [`L308`](../../../../../../../raw/code/jax/jax/_src/numpy/polynomial.py#L308) — Returns the coefficients of a polynomial for the given sequence of roots.
- `polyadd(a1: ArrayLike, a2: ArrayLike)` — [`L455`](../../../../../../../raw/code/jax/jax/_src/numpy/polynomial.py#L455) — Returns the sum of the two polynomials.
- `polyder(p: ArrayLike, m: int = 1)` — [`L582`](../../../../../../../raw/code/jax/jax/_src/numpy/polynomial.py#L582) — Returns the coefficients of the derivative of specified order of a polynomial.
- `polydiv(u: ArrayLike, v: ArrayLike, *, trim_leading_zeros: bool = False)` — [`L699`](../../../../../../../raw/code/jax/jax/_src/numpy/polynomial.py#L699) — Returns the quotient and remainder of polynomial division.
- `polyfit(x: ArrayLike, y: ArrayLike, deg: int, rcond: float | None = None, full: bool = False, w: ArrayLike | None = None, cov: bool = False)` — [`L128`](../../../../../../../raw/code/jax/jax/_src/numpy/polynomial.py#L128) — Least squares polynomial fit to data.
- `polyint(p: ArrayLike, m: int = 1, k: int | ArrayLike | None = None)` — [`L513`](../../../../../../../raw/code/jax/jax/_src/numpy/polynomial.py#L513) — Returns the coefficients of the integration of specified order of a polynomial.
- `polymul(a1: ArrayLike, a2: ArrayLike, *, trim_leading_zeros: bool = False)` — [`L632`](../../../../../../../raw/code/jax/jax/_src/numpy/polynomial.py#L632) — Returns the product of two polynomials.
- `polysub(a1: ArrayLike, a2: ArrayLike)` — [`L760`](../../../../../../../raw/code/jax/jax/_src/numpy/polynomial.py#L760) — Returns the difference of two polynomials.
- `polyval(p: ArrayLike, x: ArrayLike, *, unroll: int = 16)` — [`L391`](../../../../../../../raw/code/jax/jax/_src/numpy/polynomial.py#L391) — Evaluates the polynomial at specific values.
- `roots(p: ArrayLike, *, strip_zeros: bool = True)` — [`L66`](../../../../../../../raw/code/jax/jax/_src/numpy/polynomial.py#L66) — Returns the roots of a polynomial given the coefficients ``p``.

## Module values
- `export` — [`L40`](../../../../../../../raw/code/jax/jax/_src/numpy/polynomial.py#L40)

