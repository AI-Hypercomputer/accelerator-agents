---
title: 'Module: python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes_v3.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes_v3.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes_v3`/
symbols:
  get_tuned_block_sizes_v3: get_tuned_block_sizes_v3().
  _WARNED_MISSES._WARNED_MISSES: _WARNED_MISSES._WARNED_MISSES.
  logger: logger.
  TUNED_BLOCK_SIZES_V3.TUNED_BLOCK_SIZES_V3: TUNED_BLOCK_SIZES_V3.TUNED_BLOCK_SIZES_V3.
---
# Module: [`python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes_v3.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes_v3.py)

## Functions
- `get_tuned_block_sizes_v3(stage: str, q_dtype, kv_dtype, actual_num_q_heads: int, actual_num_kv_heads: int, head_dim: int, page_size: int, max_num_tokens: int, sliding_window: int | None = None)` — [`L2127`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes_v3.py#L2127) — Look up (bq_sz, bkv_sz, bq_csz, bkv_csz) from the v3 tuned table.

## Module values
- `TUNED_BLOCK_SIZES_V3` — [`L37`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes_v3.py#L37)
- `_WARNED_MISSES` — [`L34`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes_v3.py#L34)
- `logger` — [`L28`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes_v3.py#L28)

