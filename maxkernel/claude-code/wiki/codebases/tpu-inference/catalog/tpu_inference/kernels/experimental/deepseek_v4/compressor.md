---
title: 'Module: tpu_inference/kernels/experimental/deepseek_v4/compressor.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/experimental/deepseek_v4/compressor.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.experimental.deepseek_v4.compressor`/
symbols:
  compressor_forward: compressor_forward().
  compressor_forward_indexer: compressor_forward_indexer().
  save_partial_states: save_partial_states().
---
# Module: [`tpu_inference/kernels/experimental/deepseek_v4/compressor.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compressor.py)

## Functions
- `compressor_forward(kv_score: jax.Array, ape: jax.Array, norm_weight: jax.Array, cos_sin_cache: jax.Array, positions: jax.Array, slot_mapping: jax.Array, block_table: jax.Array, token_to_req_indices: jax.Array, kv_slot_mapping: jax.Array, cache: jax.Array, state_block_size: int, head_dim: int, rope_head_dim: int, compress_ratio: int, overlap: bool, rms_eps: float, quant_block: int)` — [`L70`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compressor.py#L70) — head_dim=512 path: project, save state, compress, store into one buffer.
- `compressor_forward_indexer(kv_score: jax.Array, ape: jax.Array, norm_weight: jax.Array, cos_sin_cache: jax.Array, positions: jax.Array, slot_mapping: jax.Array, block_table: jax.Array, token_to_req_indices: jax.Array, kv_slot_mapping: jax.Array, cache: jax.Array, state_block_size: int, head_dim: int, rope_head_dim: int, compress_ratio: int, overlap: bool, rms_eps: float, quant_block: int)` — [`L125`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compressor.py#L125) — head_dim=128 indexer path: same as ``compressor_forward``, head_dim=128.
- `save_partial_states(kv_score: jax.Array, ape: jax.Array, positions: jax.Array, state_cache: jax.Array, slot_mapping: jax.Array, head_dim: int, overlap: bool, compress_ratio: int)` — [`L24`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compressor.py#L24) — Scatter ``[kv | score + ape]`` into ``state_cache``; skip ``slot < 0``.

