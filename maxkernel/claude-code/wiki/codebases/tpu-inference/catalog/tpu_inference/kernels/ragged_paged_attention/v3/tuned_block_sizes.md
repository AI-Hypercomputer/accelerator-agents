---
title: 'Module: tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes`/
symbols:
  logger: logger.
  TUNED_BLOCK_SIZES: TUNED_BLOCK_SIZES.
  get_tuned_block_sizes: get_tuned_block_sizes().
  get_lookup_keys: get_lookup_keys().
  get_simplified_raw_key: get_simplified_raw_key().
---
# Module: [`tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py)

## Functions
- `get_lookup_keys(page_size, q_dtype, kv_dtype, num_q_heads, num_kv_heads, head_dim, max_model_len, sliding_window)` — [`L4394`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py#L4394) — Get the lookup keys for tuned block sizes.
- `get_simplified_raw_key(page_size, q_dtype, kv_dtype, actual_num_q_heads, actual_num_kv_heads, head_dim, max_model_len, sliding_window)` — [`L4434`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py#L4434) — Get the simplified key.
- `get_tuned_block_sizes(q_dtype, kv_dtype, actual_num_q_heads, actual_num_kv_heads, head_dim, page_size, max_num_tokens, pages_per_seq, sliding_window=None)` — [`L4339`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py#L4339) — Search tuned values for (num_kv_pages_per_blk, num_queries_per_blk).

## Module values
- `TUNED_BLOCK_SIZES` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py#L33)
- `logger` — [`L23`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py#L23)

