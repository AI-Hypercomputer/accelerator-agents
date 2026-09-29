---
title: 'Module: aqt/jax/v2/aqt_ragged_dot.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/aqt_ragged_dot.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.aqt_ragged_dot`/
symbols:
  RaggedDot: RaggedDot#
  RaggedDot.group_sizes: RaggedDot#group_sizes.
  RaggedDot.__call__: RaggedDot#__call__().
  ragged_dot: ragged_dot().
---
# Module: [`aqt/jax/v2/aqt_ragged_dot.py`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_ragged_dot.py)

## Classes
### `RaggedDot`
- def: [`aqt/jax/v2/aqt_ragged_dot.py:24`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_ragged_dot.py#L24)
- doc: Flax slot for jax.lax.ragged_dot.
- signature: `class RaggedDot:`
- members:
  - `group_sizes` — [`L30`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_ragged_dot.py#L30)
- protocol/private: `__call__`[`L32`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_ragged_dot.py#L32)

## Functions
- `ragged_dot(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, precision: jax.lax.PrecisionLike=None, preferred_element_type: jnp.dtype | None=None, cfg: aqt_dot_general.DotGeneral=config.config_v4())` — [`L50`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_ragged_dot.py#L50) — Quantized version of jax.lax.ragged_dot.

