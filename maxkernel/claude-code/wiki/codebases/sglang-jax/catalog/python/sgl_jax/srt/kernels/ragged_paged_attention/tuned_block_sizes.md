---
title: 'Module: python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes`/
symbols:
  get_tuned_block_sizes: get_tuned_block_sizes().
  get_simplified_key: get_simplified_key().
  TUNED_BLOCK_SIZES: TUNED_BLOCK_SIZES.
  logger: logger.
---
# Module: [`python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes.py)

## Functions
- `get_simplified_key(page_size, q_dtype, kv_dtype, num_q_heads, num_kv_heads, head_dim, max_num_tokens)` — [`L1491`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes.py#L1491) — Get the simplified key to reduce the number of combinations.
- `get_tuned_block_sizes(q_dtype, kv_dtype, actual_num_q_heads, actual_num_kv_heads, head_dim, page_size, max_num_tokens, pages_per_seq, causal=True)` — [`L1429`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes.py#L1429)

## Module values
- `TUNED_BLOCK_SIZES` — [`L23`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes.py#L23)
- `logger` — [`L11`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes.py#L11)

