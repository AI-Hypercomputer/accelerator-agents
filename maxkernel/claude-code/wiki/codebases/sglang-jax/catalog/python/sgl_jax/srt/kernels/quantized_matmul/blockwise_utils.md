---
title: 'Module: python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.quantized_matmul.blockwise_utils`/
symbols:
  get_safe_blockwise_tuned_value: get_safe_blockwise_tuned_value().
  _get_blockwise_tuning_api: _get_blockwise_tuning_api().
  get_blockwise_kernel: get_blockwise_kernel().
  expand_block_scale: expand_block_scale().
  _BLOCKWISE_KERNEL: _BLOCKWISE_KERNEL.
  _BLOCKWISE_TUNED_VALUE_CLS: _BLOCKWISE_TUNED_VALUE_CLS.
  _BLOCKWISE_GET_TUNED_BLOCK_SIZES: _BLOCKWISE_GET_TUNED_BLOCK_SIZES.
  _BLOCKWISE_TUNED_BLOCK_SIZES: _BLOCKWISE_TUNED_BLOCK_SIZES.
  _nearest_power_of_two_multiple: _nearest_power_of_two_multiple().
  logger: logger.
  _TRIED_LOADING_BLOCKWISE_KERNEL: _TRIED_LOADING_BLOCKWISE_KERNEL.
  _TRIED_LOADING_BLOCKWISE_TUNING: _TRIED_LOADING_BLOCKWISE_TUNING.
  _floor_multiple: _floor_multiple().
  convert_block_scale_to_kernel_layout: convert_block_scale_to_kernel_layout().
  _next_multiple: _next_multiple().
  _nearest_power_of_two_multiple._candidate: _nearest_power_of_two_multiple()._candidate().
  should_use_blockwise_kernel: should_use_blockwise_kernel().
  _get_current_tpu_version: _get_current_tpu_version().
  _iter_blockwise_tuned_candidates: _iter_blockwise_tuned_candidates().
---
# Module: [`python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py)

## Functions
- `_candidate(units_value: int)` — [`L141`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L141)
- `_floor_multiple(x: int, m: int)` — [`L119`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L119) — Round ``x`` down to a positive multiple of ``m``.
- `_get_blockwise_tuning_api()` — [`L78`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L78) — Lazily load tuned-size helpers for the blockwise kernel.
- `_get_current_tpu_version()` — [`L154`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L154) — Return the current TPU major version, or ``-1`` when unavailable.
- `_iter_blockwise_tuned_candidates(tuned_block_sizes: dict | None, n_batch: int, n_out: int, n_in: int, x_q_dtype: jnp.dtype, w_q_dtype: jnp.dtype, tpu_version: int)` — [`L166`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L166) — Return compatible tuned-size candidates ordered by closeness.
- `_nearest_power_of_two_multiple(x: int, base: int, upper_bound: int)` — [`L126`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L126) — Snap ``x`` to a nearby power-of-two multiple of ``base``.
- `_next_multiple(x: int, m: int)` — [`L112`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L112) — Round ``x`` up to the next multiple of ``m``.
- `convert_block_scale_to_kernel_layout(w_scale: jax.Array, out_dim: int, in_dim: int, block_size_out: int, block_size_in: int)` — [`L275`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L275) — Convert our block-scale layout to the TPU kernel layout.
- `expand_block_scale(scale_2d: jax.Array, n_out: int, block_size_out: int, channel_to_block: jax.Array | None = None)` — [`L235`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L235) — Expand a 2D block scale to the 3D kernel-ready layout.
- `get_blockwise_kernel()` — [`L60`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L60) — Lazily load the blockwise kernel implementation.
- `get_safe_blockwise_tuned_value(n_batch: int, n_out: int, n_in: int, x_q_dtype: jnp.dtype, w_q_dtype: jnp.dtype, block_size_in: int)` — [`L301`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L301) — Build a safe tuned value for the blockwise kernel on TPU.
- `should_use_blockwise_kernel(*, out_dim: int, block_size_out: int)` — [`L218`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L218) — Guard known-bad narrow-N TPU blockwise cases.

## Module values
- `_BLOCKWISE_GET_TUNED_BLOCK_SIZES` — [`L55`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L55)
- `_BLOCKWISE_KERNEL` — [`L49`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L49)
- `_BLOCKWISE_TUNED_BLOCK_SIZES` — [`L56`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L56)
- `_BLOCKWISE_TUNED_VALUE_CLS` — [`L54`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L54)
- `_TRIED_LOADING_BLOCKWISE_KERNEL` — [`L50`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L50)
- `_TRIED_LOADING_BLOCKWISE_TUNING` — [`L57`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L57)
- `logger` — [`L38`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/blockwise_utils.py#L38)

