---
title: 'Module: tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.custom_ops.experimental.deepseek_v4.deepseek_v4_compressor`/
symbols:
  VllmDeepseekCompressor.forward: VllmDeepseekCompressor#forward().
  VllmDeepseekCompressor: VllmDeepseekCompressor#
  logger: logger.
  VllmCompressorStateCache.get_kv_cache_spec: VllmCompressorStateCache#get_kv_cache_spec().
  VllmDeepseekCompressor.__init__: VllmDeepseekCompressor#__init__().
  VllmCompressorStateCache: VllmCompressorStateCache#
  VllmCompressorStateCache.block_size: VllmCompressorStateCache#block_size.
  VllmDeepseekCompressor._compress: VllmDeepseekCompressor#_compress().
  VllmCompressorStateCache.__init__: VllmCompressorStateCache#__init__().
  VllmCompressorStateCache.head_dim: VllmCompressorStateCache#head_dim.
---
# Module: [`tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py)

## Classes
### `VllmCompressorStateCache`
- def: [`tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py:39`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py#L39)
- doc: TPU-compatible compressor state cache.
- signature: `class VllmCompressorStateCache(CompressorStateCache):`
- members:
  - `get_kv_cache_spec(self, vllm_config)` — [`L74`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py#L74)
  - `block_size` — [`L71`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py#L71)
  - `head_dim` — [`L61`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py#L61)
- protocol/private: `__init__`[`L47`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py#L47)
- used by: [`__init__`](deepseek_v4_compressor.md#VllmDeepseekCompressor.__init__)

### `VllmDeepseekCompressor`
- def: [`tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py:85`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py#L85)
- doc: TPU-compatible DeepSeek-V4 compressor.
- signature: `class VllmDeepseekCompressor(DeepseekCompressor):`
- members:
  - `forward(self, kv_score: torch.Tensor, positions: torch.Tensor, rotary_emb)` — [`L106`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py#L106)
- protocol/private: `__init__`[`L94`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py#L94), `_compress`[`L168`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py#L168)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../../../common/sharding.md#ShardingAxisName), [`get_vllm_model_wrapper_context`](../../../../../models/vllm/vllm_model_wrapper_context.md#get_vllm_model_wrapper_context), [`compressor_forward`](../../../../../kernels/experimental/deepseek_v4/compressor.md#compressor_forward), [`compressor_forward_indexer`](../../../../../kernels/experimental/deepseek_v4/compressor.md#compressor_forward_indexer), [`kv_caches`](../../../../../models/vllm/vllm_model_wrapper_context.md#VllmModelWrapperContext.kv_caches), [`layer_name_to_kvcache_index`](../../../../../models/vllm/vllm_model_wrapper_context.md#VllmModelWrapperContext.layer_name_to_kvcache_index), [`mesh`](../../../../../models/vllm/vllm_model_wrapper_context.md#VllmModelWrapperContext.mesh), [`VllmCompressorStateCache`](deepseek_v4_compressor.md#VllmCompressorStateCache)
- used by: [`__init__`](deepseek_v4_attention.md#VllmDeepseekV4MLAAttention.__init__), [`__init__`](deepseek_v4_indexer.md#VllmDeepseekV4Indexer.__init__)

## Module values
- `logger` — [`L36`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_compressor.py#L36)

