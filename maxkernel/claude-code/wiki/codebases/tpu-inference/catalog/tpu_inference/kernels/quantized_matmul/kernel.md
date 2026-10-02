---
title: 'Module: tpu_inference/kernels/quantized_matmul/kernel.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/quantized_matmul/kernel.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.quantized_matmul.kernel`/
symbols:
  quantize_tensor: quantize_tensor.
  matmul_kernel: matmul_kernel().
  quantized_matmul_kernel: quantized_matmul_kernel().
---
# Module: [`tpu_inference/kernels/quantized_matmul/kernel.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/kernel.py)

## Functions
- `matmul_kernel(x_ref: jax.Array, w_q_ref: jax.Array, w_scale_ref: jax.Array, x_abs_max_ref: jax.Array, out_ref: jax.Array, acc_scratch: jax.Array, x_q_scratch: jax.Array, x_scale_scratch: jax.Array, *, x_q_dtype: jnp.dtype, save_acc: bool, save_x_q: bool)` — [`L21`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/kernel.py#L21)
- `quantized_matmul_kernel(x: jax.Array, w_q: jax.Array, w_scale: jax.Array, w_zp: jax.Array | None=None, block_size: int | None=None, x_q_dtype: jnp.dtype | None=None, *, tuned_value: TunedValue | None=None)` — [`L119`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/kernel.py#L119) — Quantized matmul kernel.

## Module values
- `quantize_tensor` — [`L18`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/kernel.py#L18)

