---
title: 'Module: tpu_inference/layers/common/quantization/fp8.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/quantization/fp8.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.quantization.fp8`/
symbols:
  process_blockwise_fp8_linear_weights: process_blockwise_fp8_linear_weights().
  Fp8LinearMethod._apply_fused: Fp8LinearMethod#_apply_fused().
  Fp8LinearMethod.linear_config: Fp8LinearMethod#linear_config.
  Fp8LinearMethod: Fp8LinearMethod#
  Fp8LinearMethod._apply_split: Fp8LinearMethod#_apply_split().
  Fp8LinearMethod.__init__: Fp8LinearMethod#__init__().
  logger: logger.
---
# Module: [`tpu_inference/layers/common/quantization/fp8.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/fp8.py)

## Classes
### `Fp8LinearMethod`
- def: [`tpu_inference/layers/common/quantization/fp8.py:37`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/fp8.py#L37)
- doc: Implements the forward method for fp8 linear layers.
- signature: `class Fp8LinearMethod:`
- members:
  - `linear_config` — [`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/fp8.py#L44)
- protocol/private: `__init__`[`L43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/fp8.py#L43), `_apply_fused`[`L46`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/fp8.py#L46), `_apply_split`[`L61`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/fp8.py#L61)
- uses (calls/refs, reference-scoped): [`sharded_quantized_matmul`](../linear.md#sharded_quantized_matmul), [`QuantLinearConfig`](configs.md#QuantLinearConfig), [`slice_sharded_tensor_for_concatenation`](../utils.md#slice_sharded_tensor_for_concatenation), [`output_sizes`](configs.md#QuantLinearConfig.output_sizes), [`n_shards`](configs.md#QuantLinearConfig.n_shards), [`Fp8BlockwiseLinearMethod`](../../jax/quantization/fp8.md#Fp8BlockwiseLinearMethod), [`Fp8TensorwiseLinearMethod`](../../jax/quantization/fp8.md#Fp8TensorwiseLinearMethod), [`weight_sharding`](configs.md#QuantLinearConfig.weight_sharding), [`mesh`](configs.md#QuantLinearConfig.mesh)
- used by: [`process_weights_after_loading`](../../jax/quantization/fp8.md#Fp8BlockwiseLinearMethod.process_weights_after_loading), [`create_weights_jax`](../../jax/quantization/fp8.md#Fp8BlockwiseLinearMethod.create_weights_jax), [`apply_jax`](../../jax/quantization/fp8.md#Fp8BlockwiseLinearMethod.apply_jax), [`create_weights_jax`](../../jax/quantization/fp8.md#Fp8TensorwiseLinearMethod.create_weights_jax), [`apply_jax`](../../jax/quantization/fp8.md#Fp8TensorwiseLinearMethod.apply_jax), [`__init__`](../../jax/quantization/fp8.md#Fp8BlockwiseLinearMethod.__init__), [`apply`](../../vllm/quantization/fp8.md#VllmFp8LinearMethod.apply), [`__init__`](../../jax/quantization/fp8.md#Fp8TensorwiseLinearMethod.__init__), [`Fp8BlockwiseLinearMethod`](../../jax/quantization/fp8.md#Fp8BlockwiseLinearMethod), [`Fp8TensorwiseLinearMethod`](../../jax/quantization/fp8.md#Fp8TensorwiseLinearMethod), [`VllmFp8LinearMethod`](../../vllm/quantization/fp8.md#VllmFp8LinearMethod)

## Functions
- `process_blockwise_fp8_linear_weights(weight: jax.Array, weight_scale: jax.Array, *, bias: jax.Array | None, weight_block_size: Sequence[int], requant_block_size, output_sizes, requant_weight_dtype, fuse_matmuls, n_shards, enable_kernel: bool = False)` — [`L91`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/fp8.py#L91)

## Module values
- `logger` — [`L34`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/fp8.py#L34)

