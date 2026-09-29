---
title: 'Module: jax/_src/lax/other.py'
type: catalog
provenance: extracted
module: jax/_src/lax/other.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.lax.other`/
symbols:
  logaddexp: logaddexp().
  logaddexp2: logaddexp2().
  conv_general_dilated_patches: conv_general_dilated_patches().
  _replace_inf: _replace_inf().
  conv_general_dilated_local: conv_general_dilated_local().
  _logaddexp_jvp: _logaddexp_jvp().
  _logaddexp2_jvp: _logaddexp2_jvp().
  _wrap_between: _wrap_between().
  DType: DType.
---
# Module: [`jax/_src/lax/other.py`](../../../../../../../raw/code/jax/jax/_src/lax/other.py)

## Functions
- `_logaddexp2_jvp(primals, tangents)` — [`L316`](../../../../../../../raw/code/jax/jax/_src/lax/other.py#L316)
- `_logaddexp_jvp(primals, tangents)` — [`L283`](../../../../../../../raw/code/jax/jax/_src/lax/other.py#L283)
- `_replace_inf(x: Array)` — [`L255`](../../../../../../../raw/code/jax/jax/_src/lax/other.py#L255)
- `_wrap_between(x, _a)` — [`L245`](../../../../../../../raw/code/jax/jax/_src/lax/other.py#L245) — Wraps `x` between `[-a, a]`.
- `conv_general_dilated_local(lhs: ArrayLike, rhs: ArrayLike, window_strides: Sequence[int], padding: str | Sequence[tuple[int, int]], filter_shape: Sequence[int], lhs_dilation: Sequence[int] | None = None, rhs_dilation: Sequence[int] | None = None, dimension_numbers: convolution.ConvGeneralDilatedDimensionNumbers | None = None, precision: lax.PrecisionLike = None)` — [`L126`](../../../../../../../raw/code/jax/jax/_src/lax/other.py#L126) — General n-dimensional unshared convolution operator with optional dilation.
- `conv_general_dilated_patches(lhs: ArrayLike, filter_shape: Sequence[int], window_strides: Sequence[int], padding: str | Sequence[tuple[int, int]], lhs_dilation: Sequence[int] | None = None, rhs_dilation: Sequence[int] | None = None, dimension_numbers: convolution.ConvGeneralDilatedDimensionNumbers | None = None, precision: lax.Precision | None = None, preferred_element_type: DType | None = None)` — [`L31`](../../../../../../../raw/code/jax/jax/_src/lax/other.py#L31) — Extract patches subject to the receptive field of `conv_general_dilated`.
- `logaddexp(x1: ArrayLike, x2: ArrayLike,)` — [`L262`](../../../../../../../raw/code/jax/jax/_src/lax/other.py#L262) — Compute log(exp(x1) + exp(x2)) avoiding overflow.
- `logaddexp2(x1: ArrayLike, x2: ArrayLike,)` — [`L294`](../../../../../../../raw/code/jax/jax/_src/lax/other.py#L294) — Compute log2(exp2(x1) + exp2(x2)) avoiding overflow.

## Module values
- `DType` — [`L29`](../../../../../../../raw/code/jax/jax/_src/lax/other.py#L29)

