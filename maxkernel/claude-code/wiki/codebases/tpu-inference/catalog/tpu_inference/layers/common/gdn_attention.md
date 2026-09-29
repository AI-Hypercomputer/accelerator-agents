---
title: 'Module: tpu_inference/layers/common/gdn_attention.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/gdn_attention.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.gdn_attention`/run_jax_gdn_attention().
symbols:
  run_jax_gdn_attention: ''
---
# Module: [`tpu_inference/layers/common/gdn_attention.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/gdn_attention.py)

## Functions
- `run_jax_gdn_attention(j_mixed_qkv: jnp.ndarray, j_b: jnp.ndarray, j_a: jnp.ndarray, conv_state: jnp.ndarray, recurrent_state: jnp.ndarray, j_conv_weight: jnp.ndarray, j_conv_bias: Optional[jnp.ndarray], j_A_log: jnp.ndarray, j_dt_bias: jnp.ndarray, state_indices: jnp.ndarray, query_start_loc: jnp.ndarray, distribution: jnp.ndarray, seq_lens: jnp.ndarray, n_kq: int, n_v: int, d_k: int, d_v: int, kernel_size: int, mesh: jax.sharding.Mesh)` — [`L30`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/gdn_attention.py#L30) — Runs the Jax GDN attention mechanism.

