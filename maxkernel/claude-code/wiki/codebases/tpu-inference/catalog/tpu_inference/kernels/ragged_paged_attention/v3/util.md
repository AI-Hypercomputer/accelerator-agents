---
title: 'Module: tpu_inference/kernels/ragged_paged_attention/v3/util.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/ragged_paged_attention/v3/util.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.ragged_paged_attention.v3.util`/
symbols:
  align_to: align_to().
  get_dtype_packing: get_dtype_packing().
  cdiv: cdiv().
  next_power_of_2: next_power_of_2().
  get_tpu_version: get_tpu_version().
  get_dtype_bitwidth: get_dtype_bitwidth().
---
# Module: [`tpu_inference/kernels/ragged_paged_attention/v3/util.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/util.py)

## Functions
- `align_to(x, a)` — [`L24`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/util.py#L24)
- `cdiv(a, b)` — [`L19`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/util.py#L19)
- `get_dtype_bitwidth(dtype)` — [`L28`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/util.py#L28)
- `get_dtype_packing(dtype)` — [`L32`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/util.py#L32)
- `get_tpu_version()` — [`L52`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/util.py#L52) — Returns the numeric version of the TPU, or -1 if not on TPU.
- `next_power_of_2(x: int)` — [`L37`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/util.py#L37) — Finds the smallest power of 2 >= x using bit manipulation.

