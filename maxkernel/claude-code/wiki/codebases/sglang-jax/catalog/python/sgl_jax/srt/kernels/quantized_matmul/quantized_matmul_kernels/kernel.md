---
title: 'Module: python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/kernel.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/kernel.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels.kernel`/
symbols:
  quantized_matmul_kernel: quantized_matmul_kernel().
  matmul_kernel: matmul_kernel().
  matmul_kernel.matmul_body: matmul_kernel().matmul_body().
  quantize_tensor: quantize_tensor.
---
# Module: [`python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/kernel.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/kernel.py)

## Functions
- `matmul_body(quant: bool, is_first_step: bool, is_last_step: bool)` — [`L64`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/kernel.py#L64)
- `matmul_kernel(x_ref: jax.Array, w_q_ref: jax.Array, w_scale_ref: jax.Array, x_abs_max_ref: jax.Array, out_ref: jax.Array, acc_scratch: jax.Array, x_q_scratch: jax.Array, x_scale_scratch: jax.Array, *, x_q_dtype: jnp.dtype, save_acc: bool, save_x_q: bool)` — [`L19`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/kernel.py#L19)
- `quantized_matmul_kernel(x: jax.Array, w_q: jax.Array, w_scale: jax.Array, w_zp: jax.Array | None = None, block_size: int | None = None, x_q_dtype: jnp.dtype | None = None, *, tuned_value: TunedValue | None = None)` — [`L119`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/kernel.py#L119) — Quantized matmul kernel.

## Module values
- `quantize_tensor` — [`L16`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/kernel.py#L16)

