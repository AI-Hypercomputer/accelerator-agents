---
title: 'Module: tokamax/_src/ops/attention/api.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/api.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.api`/
symbols:
  dot_product_attention: dot_product_attention().
  _DEFAULT_IMPLEMENTATIONS: _DEFAULT_IMPLEMENTATIONS.
  _IMPLEMENTATIONS: _IMPLEMENTATIONS.
  IMPLEMENTATIONS.IMPLEMENTATIONS: IMPLEMENTATIONS.IMPLEMENTATIONS.
  QArray: QArray.
  Implementation.Implementation: Implementation.Implementation.
  _XLA_CHUNK_SIZE._XLA_CHUNK_SIZE: _XLA_CHUNK_SIZE._XLA_CHUNK_SIZE.
---
# Module: [`tokamax/_src/ops/attention/api.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api.py)

## Functions
- `dot_product_attention(query: Float[Array | QArray, "*B T N H"], key: Float[Array | QArray, "*B S K H"], value: Float[Array | QArray, "*B S K h"], bias: Float[Array, "*#B #N #T #S"] | None = None, mask: Bool[Array, "*#B #N #T #S"] | None = None, *, scale: float | None = None, is_causal: bool = False, query_seq_lengths: Int[Array, "*#B"] | None = None, key_value_seq_lengths: Int[Array, "*#B"] | None = None, local_window_size: int | tuple[int, int] | None = None, logits_soft_cap: float | None = None, precision: jax.lax.PrecisionLike = None, implementation: Implementation | Sequence[Implementation] | None = None, q_sharding: jax.sharding.NamedSharding | None = None)` — [`L83`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api.py#L83) — Scaled dot product attention function.

## Module values
- `IMPLEMENTATIONS` — [`L77`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api.py#L77)
- `Implementation` — [`L30`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api.py#L30)
- `QArray` — [`L29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api.py#L29)
- `_DEFAULT_IMPLEMENTATIONS` — [`L46`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api.py#L46)
- `_IMPLEMENTATIONS` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api.py#L37)
- `_XLA_CHUNK_SIZE` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/api.py#L35)

