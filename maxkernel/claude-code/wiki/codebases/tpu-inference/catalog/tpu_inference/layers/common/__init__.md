---
title: 'Module: tpu_inference/layers/common/__init__.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/__init__.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.quantization`/
symbols:
  quantize_kv: quantize_kv().
  quantize_tensor: quantize_tensor().
  dequantize_tensor: dequantize_tensor().
  dequantize_tensor_from_mxfp4_packed: dequantize_tensor_from_mxfp4_packed().
  u8_unpack_e2m1: u8_unpack_e2m1().
  awq_u32_unpack_u4: awq_u32_unpack_u4().
  static_per_tensor_quantize_tensor: static_per_tensor_quantize_tensor().
  quantize_tensor_to_mxfp4_packed: quantize_tensor_to_mxfp4_packed().
  e8m0_to_fp32: e8m0_to_fp32().
  u32_unpack_i4: u32_unpack_i4().
  MXFP4_BLOCK_SIZE: MXFP4_BLOCK_SIZE.
---
# Module: [`tpu_inference/layers/common/__init__.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/__init__.py)

## Functions
- `awq_u32_unpack_u4(awq_u32_packed: jax.Array)` — [`L73`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/__init__.py#L73) — Unpack u4 tensor that was packed into u32 in awq ordering.
- `dequantize_tensor(tensor_q: jax.Array, scale: jax.Array, axis: int | None | tuple = -1, out_dtype: jnp.dtype = jnp.bfloat16, block_size: tuple[int, ...] | None = None)` — [`L99`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/__init__.py#L99) — Dequantize a quantized tensor
- `dequantize_tensor_from_mxfp4_packed(tensor_q: jax.Array, scale: jax.Array, axis: int | tuple = -1, out_dtype: jnp.dtype = jnp.bfloat16)` — [`L166`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/__init__.py#L166) — Dequantize packed mxfp4 tensor.
- `e8m0_to_fp32(u8: jax.Array)` — [`L63`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/__init__.py#L63) — Convert e8m0 (that was bitcasted to u8) into fp32.
- `quantize_kv(dtype: jnp.dtype, key: jax.Array, value: jax.Array | None = None, k_scale: float = 1, v_scale: float = 1)` — [`L294`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/__init__.py#L294) — Static quantize key and value tensors. — documented in [tpu_inference-layers-jax-attention](../../../../concepts/tpu_inference-layers-jax-attention.md)
- `quantize_tensor(dtype: jnp.dtype, tensor: jax.Array, axis: int | tuple | None = -1, block_size: int | None = None, clip_percentile: float | None = None)` — [`L194`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/__init__.py#L194) — Quantize tensor.
- `quantize_tensor_to_mxfp4_packed(tensor: jax.Array, axis: int | tuple = -1)` — [`L24`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/__init__.py#L24) — Quantize a tensor to mxfp4 and pack it into uint8.
- `static_per_tensor_quantize_tensor(dtype: jnp.dtype, tensor: jax.Array, scale: float)` — [`L278`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/__init__.py#L278)
- `u32_unpack_i4(u32_packed_i4: jax.Array)` — [`L85`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/__init__.py#L85) — Unpack i4 tensor that was packed into u32.
- `u8_unpack_e2m1(u8_packed_e2m1: jax.Array)` — [`L54`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/__init__.py#L54) — Unpack e2m1 tensor that was packed into u8.

## Module values
- `MXFP4_BLOCK_SIZE` — [`L21`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/__init__.py#L21)

