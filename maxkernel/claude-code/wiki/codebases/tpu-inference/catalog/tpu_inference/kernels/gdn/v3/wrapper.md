---
title: 'Module: tpu_inference/kernels/gdn/v3/wrapper.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/v3/wrapper.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.v3.wrapper`/
symbols:
  fused_conv1d_gdn: fused_conv1d_gdn().
  inner_kernel: inner_kernel().
  outer_kernel: outer_kernel().
---
# Module: [`tpu_inference/kernels/gdn/v3/wrapper.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/wrapper.py)

## Functions
- `fused_conv1d_gdn(qkv: jax.Array, b: jax.Array, a: jax.Array, conv_state: jax.Array, recurrent_state: jax.Array, conv_weight: jax.Array, conv_bias: jax.Array | None, a_log: jax.Array, dt_bias: jax.Array, query_start_loc: jax.Array, state_indices: jax.Array, distribution: jax.Array, seq_lens: jax.Array, *, n_kq: int, n_v: int, d_k: int, d_v: int, kernel_size: int, zero_initialize_out: bool=True, compute_precision: jnp.dtype=jnp.float32.dtype, decode_tile_size: int=4, mixed_tile_size: int=64)` — [`L271`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/wrapper.py#L271) — Perform conv1d and gdn in a single fused kernel.
- `inner_kernel(qkv_slot_ref: jax.Array, b_slot_ref: jax.Array, a_slot_ref: jax.Array, conv_state_slot_ref: jax.Array, recurrent_slot_ref: jax.Array, out_slot_ref: jax.Array, metadata_ref: memory_ref.MetadataRef, weights_ref: memory_ref.WeightRefs, carry_conv_scratch_ref: jax.Array | None, carry_recurrent_scratch_ref: jax.Array | None, *, cfg: config.GDNConfig)` — [`L26`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/wrapper.py#L26) — Orchestrates computation of Conv1D and GDN for a single tile.
- `outer_kernel(metadata_ref: memory_ref.MetadataRef, qkv_ref: jax.Array, b_ref: jax.Array, a_ref: jax.Array, conv_state_ref: jax.Array, recurrent_state_ref: jax.Array, _: jax.Array, weights_ref: memory_ref.WeightRefs, out_ref: jax.Array, conv_state_out_ref: jax.Array, recurrent_state_out_ref: jax.Array, carry_conv_scratch_ref: jax.Array | None, carry_recurrent_scratch_ref: jax.Array | None, *, cfg: config.GDNConfig)` — [`L176`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/wrapper.py#L176) — Setup memory allocations and emit pipeline for running inner_kernel.

