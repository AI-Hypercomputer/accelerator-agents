---
title: 'Module: tpu_inference/kernels/megablox/tuned_block_sizes.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/megablox/tuned_block_sizes.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.megablox.tuned_block_sizes`/
symbols:
  get_tuned_block_sizes: get_tuned_block_sizes().
  round_up_to_multiple_of_128_within_limit: round_up_to_multiple_of_128_within_limit().
  logger: logger.
  get_default_gmm_block_sizes: get_default_gmm_block_sizes().
  TUNED_BLOCK_SIZES: TUNED_BLOCK_SIZES.
---
# Module: [`tpu_inference/kernels/megablox/tuned_block_sizes.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/tuned_block_sizes.py)

## Functions
- `get_default_gmm_block_sizes(m: int, k: int, n: int, g: int)` — [`L321`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/tuned_block_sizes.py#L321) — Heuristic-based defaults for GMM tiling.
- `get_tuned_block_sizes(m: int, k: int, n: int, num_total_groups: int, num_current_groups: int, lhs_dtype: str, rhs_dtype: str, quant_block_size: int)` — [`L362`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/tuned_block_sizes.py#L362) — Retrieves optimized (TM, TK, TN) tiling parameters for the GMM kernel.
- `round_up_to_multiple_of_128_within_limit(x: int, limit: int)` — [`L287`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/tuned_block_sizes.py#L287) — Rounds the given integer `x` up to the nearest multiple of 128, without

## Module values
- `TUNED_BLOCK_SIZES` — [`L32`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/tuned_block_sizes.py#L32)
- `logger` — [`L17`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/tuned_block_sizes.py#L17)

