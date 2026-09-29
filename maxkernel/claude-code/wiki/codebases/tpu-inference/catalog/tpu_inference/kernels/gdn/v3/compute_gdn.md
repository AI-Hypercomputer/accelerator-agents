---
title: 'Module: tpu_inference/kernels/gdn/v3/compute_gdn.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/v3/compute_gdn.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.v3.compute_gdn`/
symbols:
  l2_norm: l2_norm().
  get_mask_dtype: get_mask_dtype().
  invert_triangular_matrix: invert_triangular_matrix().
  fused_transpose_broadcast: fused_transpose_broadcast().
  chunked_gdn_per_seq: chunked_gdn_per_seq().
  chunked_gdn: chunked_gdn().
  recurrent_gdn_per_seq: recurrent_gdn_per_seq().
  recurrent_gdn: recurrent_gdn().
---
# Module: [`tpu_inference/kernels/gdn/v3/compute_gdn.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/compute_gdn.py)

## Functions
- `chunked_gdn(real_sizes: jax.Array, q_large: jax.Array, k_large: jax.Array, v_large: jax.Array, b_large: jax.Array, a_large: jax.Array, state_prev: jax.Array, a_log: jax.Array, dt_bias: jax.Array, cfg: config.GDNConfig)` — [`L256`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/compute_gdn.py#L256) — Perform chunked GDN over input [seq, num_heads, chunk, head_dim].
- `chunked_gdn_per_seq(q_large: jax.Array, k_large: jax.Array, v_large: jax.Array, gating_log: jax.Array, beta: jax.Array, state_prev: jax.Array, cfg: config.GDNConfig)` — [`L106`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/compute_gdn.py#L106) — Perform chunked GDN over input [num_heads, chunk, head_dim].
- `fused_transpose_broadcast(x: jax.Array, src_dim: int, dst_dim: int)` — [`L90`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/compute_gdn.py#L90) — Perform 1D transpose where results are broadcasted along src_dim.
- `get_mask_dtype(dtype: jnp.dtype)` — [`L27`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/compute_gdn.py#L27)
- `invert_triangular_matrix(t: jax.Array, block_size=16)` — [`L38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/compute_gdn.py#L38) — Compute invert matrix of a given triauglar matrix.
- `l2_norm(x: jax.Array, eps: float=1e-06)` — [`L21`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/compute_gdn.py#L21)
- `recurrent_gdn(real_sizes: jax.Array, q_compact: jax.Array, k_compact: jax.Array, v_compact: jax.Array, b_compact: jax.Array, a_compact: jax.Array, state_prev: jax.Array, a_log: jax.Array, dt_bias: jax.Array, cfg: config.GDNConfig)` — [`L389`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/compute_gdn.py#L389) — Perform recurrent GDN over input [seq, num_heads, chunk, 1, head_dim].
- `recurrent_gdn_per_seq(q_compact: jax.Array, k_compact: jax.Array, k_compact_t: jax.Array, v_compact: jax.Array, gating_log: jax.Array, beta: jax.Array, state: jax.Array, cfgs: config.GDNConfig)` — [`L323`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/compute_gdn.py#L323) — Perform recurrent GDN over input [num_heads, chunk, 1, head_dim].

