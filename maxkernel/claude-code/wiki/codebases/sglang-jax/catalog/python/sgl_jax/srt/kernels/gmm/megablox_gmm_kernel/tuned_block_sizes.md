---
title: 'Module: python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/tuned_block_sizes.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/tuned_block_sizes.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.tuned_block_sizes`/
symbols:
  get_tuned_block_sizes: get_tuned_block_sizes().
  get_default_gmm_block_sizes: get_default_gmm_block_sizes().
  TUNED_BLOCK_SIZES: TUNED_BLOCK_SIZES.
  round_up_to_multiple_of_128_within_limit: round_up_to_multiple_of_128_within_limit().
  logger: logger.
---
# Module: [`python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/tuned_block_sizes.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/tuned_block_sizes.py)

## Functions
- `get_default_gmm_block_sizes(m: int, k: int, n: int, g: int)` — [`L541`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/tuned_block_sizes.py#L541) — Heuristic-based defaults for GMM tiling.
- `get_tuned_block_sizes(m: int, k: int, n: int, num_total_groups: int, num_current_groups: int, lhs_dtype: str, rhs_dtype: str, quant_block_size: int)` — [`L574`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/tuned_block_sizes.py#L574) — Retrieves optimized (TM, TK, TN) tiling parameters for the GMM kernel.
- `round_up_to_multiple_of_128_within_limit(x: int, limit: int)` — [`L507`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/tuned_block_sizes.py#L507) — Rounds the given integer `x` up to the nearest multiple of 128, without

## Module values
- `TUNED_BLOCK_SIZES` — [`L32`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/tuned_block_sizes.py#L32)
- `logger` — [`L17`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/tuned_block_sizes.py#L17)

