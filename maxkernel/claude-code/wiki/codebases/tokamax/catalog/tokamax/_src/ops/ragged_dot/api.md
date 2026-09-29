---
title: 'Module: tokamax/_src/ops/ragged_dot/api.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/api.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.api`/
symbols:
  _DEFAULT_IMPLEMENTATIONS: _DEFAULT_IMPLEMENTATIONS.
  ragged_dot: ragged_dot().
  ragged_dot_general: ragged_dot_general().
  IMPLEMENTATIONS.IMPLEMENTATIONS: IMPLEMENTATIONS.IMPLEMENTATIONS.
  _IMPLEMENTATIONS: _IMPLEMENTATIONS.
  QArray: QArray.
  Implementation.Implementation: Implementation.Implementation.
---
# Module: [`tokamax/_src/ops/ragged_dot/api.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api.py)

## Functions
- `ragged_dot(lhs: Float[Array | QArray, "M K"], rhs: Float[Array | QArray, "G K N"], group_sizes: Int[Array, G] | base.GroupSizes, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, group_offset: Array | None = None, activation: base.ActivationFunction | None = None, manual_axis_type: ManualAxisType | None = None, *, implementation: (Implementation | Sequence[Implementation | Callable[..., jax.Array]] | None) = None)` — [`L71`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api.py#L71) — Ragged matrix multiplication.
- `ragged_dot_general(lhs: jax.Array | QArray, rhs: jax.Array | QArray, group_sizes: Int[Array, G] | base.GroupSizes, ragged_dot_dimension_numbers: jax.lax.RaggedDotDimensionNumbers, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, group_offset: Array | None = None, activation: base.ActivationFunction | None = None, manual_axis_type: ManualAxisType | None = None, *, implementation: (Implementation | Sequence[Implementation | Callable[..., jax.Array]] | None) = None)` — [`L131`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api.py#L131) — Ragged matrix multiplication.

## Module values
- `IMPLEMENTATIONS` — [`L65`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api.py#L65)
- `Implementation` — [`L29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api.py#L29)
- `QArray` — [`L28`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api.py#L28)
- `_DEFAULT_IMPLEMENTATIONS` — [`L32`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api.py#L32)
- `_IMPLEMENTATIONS` — [`L31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api.py#L31)

