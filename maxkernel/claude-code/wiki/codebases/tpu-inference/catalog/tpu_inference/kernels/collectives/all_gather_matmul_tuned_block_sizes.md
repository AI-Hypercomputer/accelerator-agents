---
title: 'Module: tpu_inference/kernels/collectives/all_gather_matmul_tuned_block_sizes.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/collectives/all_gather_matmul_tuned_block_sizes.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.collectives.all_gather_matmul_tuned_block_sizes`/
symbols:
  TUNED_BLOCK_SIZES: TUNED_BLOCK_SIZES.
  get_tpu_version: get_tpu_version().
  get_key: get_key().
  get_tuned_block_sizes: get_tuned_block_sizes().
---
# Module: [`tpu_inference/kernels/collectives/all_gather_matmul_tuned_block_sizes.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/all_gather_matmul_tuned_block_sizes.py)

## Functions
- `get_key(m, n, k, dtype, tp_size)` — [`L44`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/all_gather_matmul_tuned_block_sizes.py#L44) — Returns the key for the given parameters.
- `get_tpu_version()` — [`L30`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/all_gather_matmul_tuned_block_sizes.py#L30) — Returns the numeric version of the TPU, or -1 if not on TPU.
- `get_tuned_block_sizes(m, n, k, dtype_name, tp_size)` — [`L62`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/all_gather_matmul_tuned_block_sizes.py#L62) — Returns the tuned block sizes for the given parameters.

## Module values
- `TUNED_BLOCK_SIZES` — [`L18`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/collectives/all_gather_matmul_tuned_block_sizes.py#L18)

