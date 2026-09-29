---
title: 'Module: jax/experimental/pallas/ops/tpu/ragged_paged_attention/tuned_block_sizes.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/tpu/ragged_paged_attention/tuned_block_sizes.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes`/
symbols:
  get_tuned_block_sizes: get_tuned_block_sizes().
  next_power_of_2: next_power_of_2().
  simplify_key: simplify_key().
  get_min_page_size: get_min_page_size().
  get_tpu_version: get_tpu_version().
  get_device_name: get_device_name().
  TUNED_BLOCK_SIZES: TUNED_BLOCK_SIZES.
  MAX_PAGES_PER_SEQ: MAX_PAGES_PER_SEQ.
---
# Module: [`jax/experimental/pallas/ops/tpu/ragged_paged_attention/tuned_block_sizes.py`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/tuned_block_sizes.py)

## Functions
- `get_device_name(num_devices: int | None = None)` — [`L1436`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/tuned_block_sizes.py#L1436)
- `get_min_page_size(max_model_len, min_page_size=16)` — [`L1482`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/tuned_block_sizes.py#L1482) — Recommended min page size for high-performance kernel.
- `get_tpu_version()` — [`L1423`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/tuned_block_sizes.py#L1423) — Returns the numeric version of the TPU, or -1 if not on TPU.
- `get_tuned_block_sizes(q_dtype, kv_dtype, num_q_heads_per_blk, num_kv_heads_per_blk, head_dim, page_size, max_num_batched_tokens, pages_per_seq)` — [`L1443`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/tuned_block_sizes.py#L1443) — Look up for the best (num_kv_pages_per_blk, num_queries_per_blk) from auto-tuned table.
- `next_power_of_2(x: int)` — [`L1384`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/tuned_block_sizes.py#L1384) — Finds the smallest power of 2 >= x using bit manipulation.
- `simplify_key(key)` — [`L1399`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/tuned_block_sizes.py#L1399) — Simplify the key to reduce the number of combinations.

## Module values
- `MAX_PAGES_PER_SEQ` — [`L22`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/tuned_block_sizes.py#L22)
- `TUNED_BLOCK_SIZES` — [`L36`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/tuned_block_sizes.py#L36)

