---
title: 'Module: easydel/layers/rotary/_rotary.py'
type: catalog
provenance: extracted
module: easydel/layers/rotary/_rotary.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.rotary._rotary`/get_
symbols:
  get_frequencies: frequencies().
  get_rope: rope().
  get_inv_frequencies: inv_frequencies().
---
# Module: [`easydel/layers/rotary/_rotary.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_rotary.py)

## Functions
- `get_frequencies(head_size: int, rotary_dim: int, max_position: int, base: int, rope_scaling: dict[str, tp.Any] | None = None, partial_rotary_factor: float = 1)` — [`L269`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_rotary.py#L269) — Computes and returns the RoPE frequency cache based on configuration.
- `get_inv_frequencies(head_size: int, rotary_dim: int, max_position: int, base: int, rope_scaling: dict[str, tp.Any] | None = None, partial_rotary_factor: float = 1)` — [`L447`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_rotary.py#L447) — Computes and returns just the inverse frequencies for RoPE based on configuration.
- `get_rope(head_size: int, rotary_dim: int, max_position: int, base: int, is_neox_style: bool = True, rope_scaling: dict[str, tp.Any] | None = None, dtype: jnp.dtype | None = None, partial_rotary_factor: float = 1)` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_rotary.py#L81) — Factory function to create and return a RotaryEmbedding instance based on configuration.

