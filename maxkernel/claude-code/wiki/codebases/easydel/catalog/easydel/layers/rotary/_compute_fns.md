---
title: 'Module: easydel/layers/rotary/_compute_fns.py'
type: catalog
provenance: extracted
module: easydel/layers/rotary/_compute_fns.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.rotary._compute_fns`/
symbols:
  compute_deepseek_frequencies: compute_deepseek_frequencies().
  compute_basic_inv_frequencies: compute_basic_inv_frequencies().
  apply_basic_rope: apply_basic_rope().
  compute_yarn_frequencies: compute_yarn_frequencies().
  compute_basic_frequencies: compute_basic_frequencies().
  compute_yarn_inv_frequencies: compute_yarn_inv_frequencies().
  compute_linear_frequencies: compute_linear_frequencies().
  compute_dynamic_frequencies: compute_dynamic_frequencies().
  compute_llama3_frequencies: compute_llama3_frequencies().
  compute_llama3_inv_frequencies: compute_llama3_inv_frequencies().
  compute_phi3_frequencies: compute_phi3_frequencies().
  apply_phi3_rope: apply_phi3_rope().
---
# Module: [`easydel/layers/rotary/_compute_fns.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_compute_fns.py)

## Functions
- `apply_basic_rope(query: jax.Array, key: jax.Array, positions: jax.Array, frequencies: jax.Array, rotary_dim: int, is_neox_style: bool, offsets: jax.Array | None = None, dtype: jnp.dtype = jnp.float32)` — [`L534`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_compute_fns.py#L534) — Applies standard or partially applied RoPE to query and key tensors.
- `apply_phi3_rope(query, key, positions, frequencies, offsets: jax.Array | None = None, dtype: jnp.dtype = jnp.float32)` — [`L493`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_compute_fns.py#L493) — Applies Phi-3 LongRoPE to query and key tensors.
- `compute_basic_frequencies(base: int, rotary_dim: int, max_position_embeddings: int)` — [`L168`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_compute_fns.py#L168) — Computes the basic RoPE frequencies (cos and sin values) for all positions.
- `compute_basic_inv_frequencies(base: int, rotary_dim: int)` — [`L63`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_compute_fns.py#L63) — Computes the inverse frequencies for standard RoPE.
- `compute_deepseek_frequencies(base, rotary_dim, scaling_factor, extrapolation_factor, beta_fast, beta_slow, max_position_embeddings, mscale, mscale_all_dim, attn_factor)` — [`L432`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_compute_fns.py#L432) — Computes RoPE frequencies using the Deepseek-YaRN scaling method.
- `compute_dynamic_frequencies(base: int, rotary_dim: int, max_position_embeddings: int, scaling_factor: float)` — [`L250`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_compute_fns.py#L250) — Computes RoPE frequencies using Dynamic NTK scaling.
- `compute_linear_frequencies(base: int, rotary_dim: int, max_position_embeddings: int, scaling_factors: list[float])` — [`L193`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_compute_fns.py#L193) — Computes RoPE frequencies using linear scaling for potentially multiple factors.
- `compute_llama3_frequencies(base, rotary_dim, low_freq_factor, high_freq_factor, scaling_factor, max_position_embeddings: int)` — [`L390`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_compute_fns.py#L390) — Computes RoPE frequencies using the Llama3 scaling method.
- `compute_llama3_inv_frequencies(base, rotary_dim, low_freq_factor, high_freq_factor, orig_max_position, scaling_factor)` — [`L122`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_compute_fns.py#L122) — Computes the inverse frequencies for Llama3-style scaled RoPE.
- `compute_phi3_frequencies(base, head_size, rotary_dim, max_position_embeddings, original_max_position_embeddings, short_factor, long_factor)` — [`L329`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_compute_fns.py#L329) — Computes RoPE frequencies using the Phi-3 LongRoPE scaling method.
- `compute_yarn_frequencies(base: float, rotary_dim: int, beta_fast: float, beta_slow: float, max_position_embeddings: int, scaling_factor: float, extrapolation_factor: float, attn_factor: float)` — [`L282`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_compute_fns.py#L282) — Computes RoPE frequencies using the YaRN scaling method.
- `compute_yarn_inv_frequencies(base: float, rotary_dim: int, beta_fast: float, beta_slow: float, max_position_embeddings: int, scaling_factor: float, extrapolation_factor: float)` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/layers/rotary/_compute_fns.py#L78) — Computes the inverse frequencies for YaRN scaled RoPE.

