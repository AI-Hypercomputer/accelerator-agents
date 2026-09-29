---
title: 'Module: tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.custom_ops.experimental.deepseek_v4.deepseek_v4_attention`/
symbols:
  VllmDeepseekV4MLAAttention.forward_mqa: VllmDeepseekV4MLAAttention#forward_mqa().
  VllmDeepseekV4MLAAttention.attention_impl: VllmDeepseekV4MLAAttention#attention_impl().
  VllmDeepseekV4MLAAttention.__init__: VllmDeepseekV4MLAAttention#__init__().
  VllmDeepseekV4MLAAttention.forward: VllmDeepseekV4MLAAttention#forward().
  patch_deepseek_v4_mla_cls: patch_deepseek_v4_mla_cls().
  VllmDeepseekV4MLAAttention._attention: VllmDeepseekV4MLAAttention#_attention().
  VllmDeepseekV4SWACache.get_kv_cache_spec: VllmDeepseekV4SWACache#get_kv_cache_spec().
  align_to: align_to().
  logger: logger.
  VllmDeepseekV4MLAAttention.get_kv_cache_spec: VllmDeepseekV4MLAAttention#get_kv_cache_spec().
  cdiv: cdiv().
  VllmDeepseekV4SWACache: VllmDeepseekV4SWACache#
  VllmDeepseekV4SWACache.block_size: VllmDeepseekV4SWACache#block_size.
  VllmDeepseekV4MLAAttention: VllmDeepseekV4MLAAttention#
  VllmDeepseekV4MLAAttention._o_proj: VllmDeepseekV4MLAAttention#_o_proj().
  VllmDeepseekV4MLAAttention.attn_gemm: VllmDeepseekV4MLAAttention#attn_gemm().
  VllmDeepseekV4MLAAttention.qnorm_rope: VllmDeepseekV4MLAAttention#qnorm_rope().
  VllmDeepseekV4MLAAttention.kv_rope: VllmDeepseekV4MLAAttention#kv_rope().
  VllmDeepseekV4SWACache.__init__: VllmDeepseekV4SWACache#__init__().
  VllmDeepseekV4MLAAttention.get_padded_num_q_heads: VllmDeepseekV4MLAAttention#get_padded_num_q_heads().
---
# Module: [`tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py)

## Classes
### `VllmDeepseekV4MLAAttention`
- def: [`tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py:104`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L104)
- signature: `class VllmDeepseekV4MLAAttention(DeepseekV4Attention):`
- members:
  - `_o_proj(self, o: torch.Tensor, positions: torch.Tensor)` — [`L157`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L157) — Inverse-RoPE + wo_a (per-group bmm) + wo_b output projection.
  - `attention_impl(self, hidden_states: torch.Tensor, qr: torch.Tensor, kv: torch.Tensor, kv_score: torch.Tensor, indexer_kv_score: torch.Tensor, indexer_weights: torch.Tensor, positions: torch.Tensor, out: torch.Tensor)` — [`L237`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L237)
  - `attn_gemm(self, hidden_states)` — [`L186`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L186)
  - `forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, llama_4_scaling: torch.Tensor | None = None)` — [`L435`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L435)
  - `forward_mqa(self, q: torch.Tensor, kv: torch.Tensor, positions: torch.Tensor, output: torch.Tensor, *, topk_indices: torch.Tensor)` — [`L273`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L273)
  - `get_kv_cache_spec(self, vllm_config: VllmConfig)` — [`L140`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L140)
  - `get_padded_num_q_heads(cls, num_heads: int)` — [`L137`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L137)
  - `kv_rope(self, kv: torch.Tensor, positions: torch.Tensor)` — [`L229`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L229)
  - `qnorm_rope(self, q: torch.Tensor, positions: torch.Tensor)` — [`L211`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L211) — Per-head RMSNorm (no weight) + GPT-J interleaved RoPE on q.
- protocol/private: `__init__`[`L106`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L106), `_attention`[`L360`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L360)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../../../common/sharding.md#ShardingAxisName), [`mla_sliding_window_ragged_paged_attention`](../../../../../kernels/experimental/deepseek_v4/mla_swa.md#mla_sliding_window_ragged_paged_attention), [`mla_ragged_paged_attention`](../../../../../kernels/experimental/deepseek_v4/mla.md#mla_ragged_paged_attention), [`get_vllm_model_wrapper_context`](../../../../../models/vllm/vllm_model_wrapper_context.md#get_vllm_model_wrapper_context), [`kv_caches`](../../../../../models/vllm/vllm_model_wrapper_context.md#VllmModelWrapperContext.kv_caches), [`layer_name_to_kvcache_index`](../../../../../models/vllm/vllm_model_wrapper_context.md#VllmModelWrapperContext.layer_name_to_kvcache_index), [`align_to`](deepseek_v4_attention.md#align_to), [`VllmDeepseekCompressor`](deepseek_v4_compressor.md#VllmDeepseekCompressor), [`mesh`](../../../../../models/vllm/vllm_model_wrapper_context.md#VllmModelWrapperContext.mesh), [`VllmDeepseekV4Indexer`](deepseek_v4_indexer.md#VllmDeepseekV4Indexer), [`VllmDeepseekV4SWACache`](deepseek_v4_attention.md#VllmDeepseekV4SWACache)
- used by: [`patch_deepseek_v4_mla_cls`](deepseek_v4_attention.md#patch_deepseek_v4_mla_cls)

### `VllmDeepseekV4SWACache`
- def: [`tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py:70`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L70)
- signature: `class VllmDeepseekV4SWACache(DeepseekV4SWACache):`
- members:
  - `get_kv_cache_spec(self, vllm_config: VllmConfig)` — [`L89`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L89)
  - `block_size` — [`L86`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L86)
- protocol/private: `__init__`[`L72`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L72)
- uses (calls/refs, reference-scoped): [`align_to`](deepseek_v4_attention.md#align_to)
- used by: [`__init__`](deepseek_v4_attention.md#VllmDeepseekV4MLAAttention.__init__)

## Functions
- `align_to(x, a)` — [`L66`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L66)
- `cdiv(a, b)` — [`L61`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L61)
- `patch_deepseek_v4_mla_cls()` — [`L461`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L461) — Rebind ``DeepseekV4ROCMAiterMLAAttention`` to the TPU subclass.

## Module values
- `logger` — [`L58`](../../../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_attention.py#L58)

