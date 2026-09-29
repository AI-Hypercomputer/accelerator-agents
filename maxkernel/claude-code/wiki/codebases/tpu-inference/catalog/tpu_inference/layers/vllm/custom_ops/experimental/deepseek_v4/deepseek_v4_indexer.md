---
title: 'Module: tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.custom_ops.experimental.deepseek_v4.deepseek_v4_indexer`/
symbols:
  VllmDeepseekV4Indexer.forward: VllmDeepseekV4Indexer#forward().
  VllmDeepseekV4Indexer.__init__: VllmDeepseekV4Indexer#__init__().
  align_to: align_to().
  fused_indexer_q_rope_quant: fused_indexer_q_rope_quant().
  VllmDeepseekV4Indexer._streamindex_topk: VllmDeepseekV4Indexer#_streamindex_topk().
  logger: logger.
  VllmDeepseekV4IndexerCache.get_kv_cache_spec: VllmDeepseekV4IndexerCache#get_kv_cache_spec().
  VllmDeepseekV4Indexer: VllmDeepseekV4Indexer#
  cdiv: cdiv().
  VllmDeepseekV4IndexerCache: VllmDeepseekV4IndexerCache#
---
# Module: [`tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py)

## Classes
### `VllmDeepseekV4Indexer`
- def: [`tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py:114`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py#L114)
- doc: TPU-compatible DeepSeek-V4 Lightning Indexer with StreamIndex.
- signature: `class VllmDeepseekV4Indexer(DeepseekV4Indexer):`
- members:
  - `forward(self, hidden_states: torch.Tensor, query: torch.Tensor, compressed_kv_score: torch.Tensor, indexer_weights: torch.Tensor, positions: torch.Tensor, rotary_emb: nn.Module, slot_mapping: Optional[torch.Tensor] = None)` — [`L133`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py#L133)
- protocol/private: `__init__`[`L119`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py#L119), `_streamindex_topk`[`L180`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py#L180)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../../../common/sharding.md#ShardingAxisName), [`streamindex_topk`](../../../../../kernels/experimental/deepseek_v4/streamindex_topk.md#streamindex_topk), [`get_vllm_model_wrapper_context`](../../../../../models/vllm/vllm_model_wrapper_context.md#get_vllm_model_wrapper_context), [`kv_caches`](../../../../../models/vllm/vllm_model_wrapper_context.md#VllmModelWrapperContext.kv_caches), [`layer_name_to_kvcache_index`](../../../../../models/vllm/vllm_model_wrapper_context.md#VllmModelWrapperContext.layer_name_to_kvcache_index), [`VllmDeepseekCompressor`](deepseek_v4_compressor.md#VllmDeepseekCompressor), [`mesh`](../../../../../models/vllm/vllm_model_wrapper_context.md#VllmModelWrapperContext.mesh), [`fused_indexer_q_rope_quant`](deepseek_v4_indexer.md#fused_indexer_q_rope_quant), [`VllmDeepseekV4IndexerCache`](deepseek_v4_indexer.md#VllmDeepseekV4IndexerCache)
- used by: [`__init__`](deepseek_v4_attention.md#VllmDeepseekV4MLAAttention.__init__)

### `VllmDeepseekV4IndexerCache`
- def: [`tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py:89`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py#L89)
- doc: TPU-compatible indexer KV cache.
- signature: `class VllmDeepseekV4IndexerCache(DeepseekV4IndexerCache):`
- members:
  - `get_kv_cache_spec(self, vllm_config)` — [`L97`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py#L97)
- uses (calls/refs, reference-scoped): [`align_to`](deepseek_v4_indexer.md#align_to)
- used by: [`__init__`](deepseek_v4_indexer.md#VllmDeepseekV4Indexer.__init__)

## Functions
- `align_to(x, a)` — [`L52`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py#L52)
- `cdiv(a, b)` — [`L47`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py#L47)
- `fused_indexer_q_rope_quant(q: torch.Tensor, positions: torch.Tensor, rotary_emb: torch.nn.Module)` — [`L56`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py#L56) — Applies RoPE and dynamically quantizes the queries

## Module values
- `logger` — [`L44`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py#L44)

