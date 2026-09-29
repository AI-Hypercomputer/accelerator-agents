---
title: 'Module: qwix/_src/core/ragged_dot.py'
type: catalog
provenance: extracted
module: qwix/_src/core/ragged_dot.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.core.ragged_dot`/
symbols:
  _fast_ragged_dot_general: _fast_ragged_dot_general().
  ragged_dot_general: ragged_dot_general().
  ragged_dot: ragged_dot().
  _slow_ragged_dot_general: _slow_ragged_dot_general().
  _BASIC_RAGGED_DOT_DIMENSION_NUMBERS: _BASIC_RAGGED_DOT_DIMENSION_NUMBERS.
  _apply_tiling: _apply_tiling().
  _apply_group_channelwise_scale: _apply_group_channelwise_scale().
  _ragged_get_scale_transpose: _ragged_get_scale_transpose().
---
# Module: [`qwix/_src/core/ragged_dot.py`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot.py)

## Functions
- `_apply_group_channelwise_scale(rhs_scale: jax.Array, lhs_shape: tuple[int, ...], group_sizes: jax.Array, dimension_numbers: jax.lax.RaggedDotDimensionNumbers, precision: jax.lax.PrecisionLike, group_offset: jax.Array | None)` — [`L32`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot.py#L32) — Expands the group dimension of rhs_scale using a gather-like op.
- `_apply_tiling(contracting_axes: Sequence[int], batch_axes: Sequence[int], tiled_axes: Collection[int])` — [`L60`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot.py#L60) — Apply tiling to dimension numbers.
- `_fast_ragged_dot_general(lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray, group_sizes: jax.Array, dimension_numbers: jax.lax.RaggedDotDimensionNumbers, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, group_offset: jax.Array | None = None)` — [`L123`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot.py#L123) — Quantized ragged_dot_general with a fast path.
- `_ragged_get_scale_transpose(dimension_numbers: jax.lax.RaggedDotDimensionNumbers, ndims: tuple[int, int])` — [`L92`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot.py#L92) — Calculates the transpose permutation for lhs_scale and rhs_scale.
- `_slow_ragged_dot_general(lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray, group_sizes: jax.Array, dimension_numbers: jax.lax.RaggedDotDimensionNumbers, **kwargs)` — [`L229`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot.py#L229) — A ragged_dot_general which dequantizes first.
- `ragged_dot(lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray, group_sizes: jax.Array, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, group_offset: jax.Array | None = None)` — [`L316`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot.py#L316) — Computes a ragged dot product with support for ``QArray`` inputs.
- `ragged_dot_general(lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray, group_sizes: jax.Array, dimension_numbers: jax.lax.RaggedDotDimensionNumbers, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, group_offset: jax.Array | None = None)` — [`L244`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot.py#L244) — Computes a general ragged dot product with support for ``QArray`` inputs.

## Module values
- `_BASIC_RAGGED_DOT_DIMENSION_NUMBERS` — [`L25`](../../../../../../../raw/code/qwix/qwix/_src/core/ragged_dot.py#L25)

