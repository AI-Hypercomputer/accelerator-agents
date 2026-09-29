---
title: 'Module: tpu_inference/kernels/ragged_paged_attention/v2/tuned_block_sizes.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/ragged_paged_attention/v2/tuned_block_sizes.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.ragged_paged_attention.v2.tuned_block_sizes`/
symbols:
  MAX_PAGES_PER_SEQ: MAX_PAGES_PER_SEQ.
  TUNED_BLOCK_SIZES: TUNED_BLOCK_SIZES.
  next_power_of_2: next_power_of_2().
  simplify_key: simplify_key().
  get_tpu_version: get_tpu_version().
  get_device_name: get_device_name().
  get_tuned_block_sizes: get_tuned_block_sizes().
  get_min_page_size: get_min_page_size().
---
# Module: [`tpu_inference/kernels/ragged_paged_attention/v2/tuned_block_sizes.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/tuned_block_sizes.py)

## Functions
- `get_device_name(num_devices: int | None=None)` — [`L1433`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/tuned_block_sizes.py#L1433)
- `get_min_page_size(max_model_len, min_page_size=16)` — [`L1479`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/tuned_block_sizes.py#L1479) — Recommended min page size for high-performance kernel.
- `get_tpu_version()` — [`L1422`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/tuned_block_sizes.py#L1422) — Returns the numeric version of the TPU, or -1 if not on TPU.
- `get_tuned_block_sizes(q_dtype, kv_dtype, num_q_heads_per_blk, num_kv_heads_per_blk, head_dim, page_size, max_num_batched_tokens, pages_per_seq)` — [`L1440`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/tuned_block_sizes.py#L1440) — Look up for the best (num_kv_pages_per_blk, num_queries_per_blk) from auto-tuned table.
- `next_power_of_2(x: int)` — [`L1383`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/tuned_block_sizes.py#L1383) — Finds the smallest power of 2 >= x using bit manipulation.
- `simplify_key(key)` — [`L1398`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/tuned_block_sizes.py#L1398) — Simplify the key to reduce the number of combinations.

## Module values
- `MAX_PAGES_PER_SEQ` — [`L21`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/tuned_block_sizes.py#L21)
- `TUNED_BLOCK_SIZES` — [`L35`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/tuned_block_sizes.py#L35)

