---
title: 'Module: ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_utils.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_utils.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ragged_page_attention_v2._utils`/
symbols:
  get_tuned_block_sizes: get_tuned_block_sizes().
  next_power_of_2: next_power_of_2().
  get_min_page_size: get_min_page_size().
  simplify_key: simplify_key().
  TUNED_BLOCK_SIZES: TUNED_BLOCK_SIZES.
  MAX_PAGES_PER_SEQ: MAX_PAGES_PER_SEQ.
  get_tpu_version: get_tpu_version().
  get_device_name: get_device_name().
---
# Module: [`ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_utils.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_utils.py)

## Functions
- `get_device_name(num_devices: int | None = None)` — [`L471`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_utils.py#L471) — Get a standardized device name string for block size lookup.
- `get_min_page_size(max_model_len, min_page_size=16)` — [`L547`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_utils.py#L547) — Compute the recommended minimum page size for the attention kernel.
- `get_tpu_version()` — [`L453`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_utils.py#L453) — Detect and return the numeric TPU generation version.
- `get_tuned_block_sizes(q_dtype, kv_dtype, num_q_heads_per_blk, num_kv_heads_per_blk, head_dim, page_size, max_num_batched_tokens, pages_per_seq)` — [`L489`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_utils.py#L489) — Look up optimal block sizes from the auto-tuned configuration table.
- `next_power_of_2(x: int)` — [`L399`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_utils.py#L399) — Finds the smallest power of 2 >= x using bit manipulation.
- `simplify_key(key)` — [`L414`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_utils.py#L414) — Normalize workload parameters to reduce the tuning space.

## Module values
- `MAX_PAGES_PER_SEQ` — [`L76`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_utils.py#L76)
- `TUNED_BLOCK_SIZES` — [`L79`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_utils.py#L79)

