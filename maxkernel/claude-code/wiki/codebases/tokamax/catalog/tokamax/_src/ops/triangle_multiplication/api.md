---
title: 'Module: tokamax/_src/ops/triangle_multiplication/api.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/triangle_multiplication/api.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.triangle_multiplication.api`/
symbols:
  triangle_multiplication: triangle_multiplication().
  _DEFAULT_IMPLEMENTATIONS._DEFAULT_IMPLEMENTATIONS: _DEFAULT_IMPLEMENTATIONS._DEFAULT_IMPLEMENTATIONS.
  Implementation.Implementation: Implementation.Implementation.
  IMPLEMENTATIONS: IMPLEMENTATIONS.
---
# Module: [`tokamax/_src/ops/triangle_multiplication/api.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/api.py)

## Functions
- `triangle_multiplication(x: Float[Array, "N N C"], mask: Bool[Array, "N N"], projection_in_weights: Float[Array, "C 2 H"], gate_in_weights: Float[Array, "C 2 H"], projection_out_weights: Float[Array, "H D"], gate_out_weights: Float[Array, "C D"], layernorm_in_scale: Float[Array, C], layernorm_in_offset: Float[Array, C], layernorm_out_scale: Float[Array, H], layernorm_out_offset: Float[Array, H], triangle_type: Literal["incoming", "outgoing"], *, precision: jax.lax.PrecisionLike = None, epsilon: float = 0.000001, implementation: Implementation | Sequence[Implementation] | None = None)` — [`L33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/api.py#L33) — Triangle multiplicative update.

## Module values
- `IMPLEMENTATIONS` — [`L28`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/api.py#L28)
- `Implementation` — [`L26`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/api.py#L26)
- `_DEFAULT_IMPLEMENTATIONS` — [`L29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/triangle_multiplication/api.py#L29)

