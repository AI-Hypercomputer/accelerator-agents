---
title: 'Module: tpu_inference/kernels/fused_moe/v1/tuned_block_sizes.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/fused_moe/v1/tuned_block_sizes.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.fused_moe.v1.tuned_block_sizes`/
symbols:
  get_tuned_block_sizes: get_tuned_block_sizes().
  align_to: align_to().
  get_default_block_sizes: get_default_block_sizes().
  logger: logger.
  TUNED_BLOCK_SIZES: TUNED_BLOCK_SIZES.
  cdiv: cdiv().
---
# Module: [`tpu_inference/kernels/fused_moe/v1/tuned_block_sizes.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/tuned_block_sizes.py)

## Functions
- `align_to(x, a)` — [`L27`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/tuned_block_sizes.py#L27)
- `cdiv(a, b)` — [`L22`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/tuned_block_sizes.py#L22)
- `get_default_block_sizes(hidden_size: int, intermediate_size: int, num_experts: int, top_k: int, t_packing: int, w_packing: int, num_tokens: int, ep_size: int)` — [`L353`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/tuned_block_sizes.py#L353)
- `get_tuned_block_sizes(hidden_size: int, intermediate_size: int, num_experts: int, top_k: int, t_packing: int, w_packing: int, num_tokens: int, ep_size: int)` — [`L391`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/tuned_block_sizes.py#L391)

## Module values
- `TUNED_BLOCK_SIZES` — [`L49`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/tuned_block_sizes.py#L49)
- `logger` — [`L18`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/fused_moe/v1/tuned_block_sizes.py#L18)

