---
title: 'Module: python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/blockwise_kernel.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/blockwise_kernel.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels.blockwise_kernel`/
symbols:
  quantized_matmul_kernel: quantized_matmul_kernel().
  quantized_matmul_kernel.kernel: quantized_matmul_kernel().kernel().
  quantized_matmul_kernel.kernel.accum: quantized_matmul_kernel().kernel().accum().
  quantize_tensor: quantize_tensor.
  MXU_SIZE: MXU_SIZE.
---
# Module: [`python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/blockwise_kernel.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/blockwise_kernel.py)

## Functions
- `accum(is_first_step, is_last_step)` — [`L144`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/blockwise_kernel.py#L144)
- `kernel(lhs_ref, rhs_ref, w_scales_ref, out_ref, acc_scratch)` — [`L139`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/blockwise_kernel.py#L139)
- `quantized_matmul_kernel(x: jax.Array, w_q: jax.Array, w_scale: jax.Array, w_zp: jax.Array | None = None, block_size: int | None = None, x_q_dtype: jnp.dtype | None = None, *, tuned_value: TunedValue | None = None)` — [`L25`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/blockwise_kernel.py#L25) — Quantized matmul kernel with blockwise support.

## Module values
- `MXU_SIZE` — [`L15`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/blockwise_kernel.py#L15)
- `quantize_tensor` — [`L14`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/blockwise_kernel.py#L14)

