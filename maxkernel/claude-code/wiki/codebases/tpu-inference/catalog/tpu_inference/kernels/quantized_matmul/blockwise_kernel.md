---
title: 'Module: tpu_inference/kernels/quantized_matmul/blockwise_kernel.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/quantized_matmul/blockwise_kernel.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.quantized_matmul.blockwise_kernel`/
symbols:
  quantize_tensor: quantize_tensor.
  MXU_SIZE: MXU_SIZE.
  quantized_matmul_kernel: quantized_matmul_kernel().
---
# Module: [`tpu_inference/kernels/quantized_matmul/blockwise_kernel.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/blockwise_kernel.py)

## Functions
- `quantized_matmul_kernel(x: jax.Array, w_q: jax.Array, w_scale: jax.Array, w_zp: jax.Array | None=None, block_size: int | None=None, x_q_dtype: jnp.dtype | None=None, *, tuned_value: TunedValue | None=None)` — [`L25`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/blockwise_kernel.py#L25) — Quantized matmul kernel with blockwise support.

## Module values
- `MXU_SIZE` — [`L17`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/blockwise_kernel.py#L17)
- `quantize_tensor` — [`L16`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/blockwise_kernel.py#L16)

