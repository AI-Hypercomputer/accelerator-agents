---
title: 'Module: python/sgl_jax/srt/utils/quantization/quantization_utils.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/utils/quantization/quantization_utils.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.utils.quantization.quantization_utils`/
symbols:
  apply_linear_quantization: apply_linear_quantization().
  quantize_tensor: quantize_tensor().
  apply_moe_quantization: apply_moe_quantization().
  apply_linear_quantization._replace_linear_recursive: apply_linear_quantization()._replace_linear_recursive().
  apply_moe_quantization._quantize_moe_recursive: apply_moe_quantization()._quantize_moe_recursive().
  logger: logger.
  quantize_tensor_simple: quantize_tensor_simple().
  _get_block_reshape_sharding: _get_block_reshape_sharding().
  _get_safe_block_quant_input_sharding: _get_safe_block_quant_input_sharding().
  apply_linear_quantization._find_matching_rule: apply_linear_quantization()._find_matching_rule().
  apply_moe_quantization._is_ignored: apply_moe_quantization()._is_ignored().
  dequantize_tensor: dequantize_tensor().
---
# Module: [`python/sgl_jax/srt/utils/quantization/quantization_utils.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/quantization/quantization_utils.py)

## Functions
- `_find_matching_rule(path: str)` — [`L151`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/quantization/quantization_utils.py#L151) — Find the first rule that matches the given module path.
- `_get_block_reshape_sharding(tensor: jax.Array, quantized_axes: list[int])` — [`L22`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/quantization/quantization_utils.py#L22) — Extend sharding specs for block reshapes.
- `_get_safe_block_quant_input_sharding(tensor: jax.Array, quantized_axes: list[int])` — [`L50`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/quantization/quantization_utils.py#L50) — Drop sharding on axes that cannot be safely split for block quant.
- `_is_ignored(log_path: str)` — [`L250`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/quantization/quantization_utils.py#L250)
- `_quantize_moe_recursive(obj, path: str = "", visited=None)` — [`L261`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/quantization/quantization_utils.py#L261)
- `_replace_linear_recursive(obj, path: str = "", visited: set | None = None)` — [`L158`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/quantization/quantization_utils.py#L158) — Recursively walk the model and replace LinearBase with QuantizedLinear.
- `apply_linear_quantization(model_config: ModelConfig, model: nnx.Module, is_static_input: bool = False)` — [`L78`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/quantization/quantization_utils.py#L78) — Apply quantization to linear layers based on regex rules. — documented in [python-sgl_jax-srt-configs-model_config](../../../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
- `apply_moe_quantization(model_config: ModelConfig, model: nnx.Module, is_static_input: bool = False)` — [`L224`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/quantization/quantization_utils.py#L224) — Quantize MoE weights in-place. — documented in [python-sgl_jax-srt-configs-model_config](../../../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
- `dequantize_tensor(tensor_q: jax.Array, scale: jax.Array, axis: int | None | tuple = -1, out_dtype: jnp.dtype = jnp.bfloat16)` — [`L425`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/quantization/quantization_utils.py#L425) — Dequantize a quantized tensor
- `quantize_tensor(dtype: jnp.dtype, tensor: jax.Array, axis: int | tuple | None = -1, block_size: int | None = None, pad_tensor: bool = False)` — [`L317`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/quantization/quantization_utils.py#L317) — Quantize tensor.
- `quantize_tensor_simple(x: jax.Array, dtype: jnp.dtype, dim: int = -1, out_dtype: jnp.dtype = jnp.float32)` — [`L295`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/quantization/quantization_utils.py#L295) — Simple per-token quantization for activations.

## Module values
- `logger` — [`L19`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/utils/quantization/quantization_utils.py#L19)

