---
title: 'Module: tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.quantized_matmul.tuned_block_sizes`/
symbols:
  get_tuned_block_sizes: get_tuned_block_sizes().
  TUNED_BLOCK_SIZES.TUNED_BLOCK_SIZES: TUNED_BLOCK_SIZES.TUNED_BLOCK_SIZES.
  get_device_vmem_limit: get_device_vmem_limit().
  TunedValue: TunedValue#
  get_key: get_key().
  logger: logger.
  TunedKey: TunedKey#
  DEVICE_VMEM_LIMIT: DEVICE_VMEM_LIMIT.
  get_tpu_version: get_tpu_version().
  TunedValue.batch_block_size: TunedValue#batch_block_size.
  TunedValue.out_block_size: TunedValue#out_block_size.
  TunedValue.in_block_size: TunedValue#in_block_size.
  TUNED_BLOCK_SIZES_RAW: TUNED_BLOCK_SIZES_RAW.
  TunedKey.tpu_version: TunedKey#tpu_version.
  TunedKey.n_batch: TunedKey#n_batch.
  TunedKey.n_out: TunedKey#n_out.
  TunedKey.n_in: TunedKey#n_in.
  TunedKey.x_q_dtype: TunedKey#x_q_dtype.
  TunedKey.w_q_dtype: TunedKey#w_q_dtype.
  TunedValue.n_lane_multiplier: TunedValue#n_lane_multiplier.
---
# Module: [`tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py)

## Classes
### `TunedKey`  ·  implements/extends NamedTuple
- def: [`tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py:14`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L14)
- signature: `class TunedKey(NamedTuple):`
- members:
  - `n_batch` — [`L16`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L16)
  - `n_in` — [`L18`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L18)
  - `n_out` — [`L17`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L17)
  - `tpu_version` — [`L15`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L15)
  - `w_q_dtype` — [`L20`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L20)
  - `x_q_dtype` — [`L19`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L19)
- used by: [`TUNED_BLOCK_SIZES`](tuned_block_sizes.md#TUNED_BLOCK_SIZES.TUNED_BLOCK_SIZES), [`get_key`](tuned_block_sizes.md#get_key)

### `TunedValue`  ·  implements/extends NamedTuple
- def: [`tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py:23`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L23)
- signature: `class TunedValue(NamedTuple):`
- members:
  - `batch_block_size` — [`L24`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L24)
  - `in_block_size` — [`L26`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L26)
  - `n_lane_multiplier` — [`L27`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L27)
  - `out_block_size` — [`L25`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L25)
- used by: [`get_kernel_name`](util.md#get_kernel_name), [`get_tuned_block_sizes`](tuned_block_sizes.md#get_tuned_block_sizes), [`TUNED_BLOCK_SIZES`](tuned_block_sizes.md#TUNED_BLOCK_SIZES.TUNED_BLOCK_SIZES)

## Functions
- `get_device_vmem_limit()` — [`L629`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L629)
- `get_key(n_batch: int, n_out: int, n_in: int, x_q_dtype: str, w_q_dtype: str)` — [`L648`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L648) — Returns the key for the given parameters.
- `get_tpu_version()` — [`L639`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L639) — Returns the numeric version of the TPU, or -1 if not on TPU.
- `get_tuned_block_sizes(n_batch: int, n_out: int, n_in: int, x_q_dtype: str, w_q_dtype: str)` — [`L666`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L666) — Retrieve the tuned block sizes for the given parameters.

## Module values
- `DEVICE_VMEM_LIMIT` — [`L626`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L626)
- `TUNED_BLOCK_SIZES` — [`L621`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L621)
- `TUNED_BLOCK_SIZES_RAW` — [`L30`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L30)
- `logger` — [`L11`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/tuned_block_sizes.py#L11)

