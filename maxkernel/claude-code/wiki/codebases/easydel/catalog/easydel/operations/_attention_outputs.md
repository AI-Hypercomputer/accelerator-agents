---
title: 'Module: easydel/operations/_attention_outputs.py'
type: catalog
provenance: extracted
module: easydel/operations/_attention_outputs.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations._attention_outputs`/AttentionOutput#
symbols:
  AttentionOutput: ''
  AttentionOutput.attention_outputs: attention_outputs.
  AttentionOutput.attention_weights: attention_weights.
  AttentionOutput.cache_view: cache_view.
---
# Module: [`easydel/operations/_attention_outputs.py`](../../../../../../raw/code/EasyDeL/easydel/operations/_attention_outputs.py)

## Classes
### `AttentionOutput`  ·  implements/extends OperationOutput
- def: [`easydel/operations/_attention_outputs.py:57`](../../../../../../raw/code/EasyDeL/easydel/operations/_attention_outputs.py#L57) — documented in [easydel-layers-attention-_unified](../../../concepts/easydel-layers-attention-_unified.md)
- doc: Container for attention computation results.
- signature: `class AttentionOutput(OperationOutput):`
- members:
  - `attention_outputs` — [`L110`](../../../../../../raw/code/EasyDeL/easydel/operations/_attention_outputs.py#L110) — documented in [easydel-layers-attention-_unified](../../../concepts/easydel-layers-attention-_unified.md)
  - `attention_weights` — [`L109`](../../../../../../raw/code/EasyDeL/easydel/operations/_attention_outputs.py#L109)
  - `cache_view` — [`L111`](../../../../../../raw/code/EasyDeL/easydel/operations/_attention_outputs.py#L111)
- uses (calls/refs, reference-scoped): [`RaggedPagesCacheView`](../caching/ragged_page/cache.md#RaggedPagesCacheView), [`TransformerCacheView`](../caching/transformer/cache.md#TransformerCacheView), [`UnifiedAttentionCacheView`](../caching/unified_attention/cache.md#UnifiedAttentionCacheView), [`GatedDeltaRuleOutput`](kernels/gated_delta_rule.md#GatedDeltaRuleOutput), [`KDAOutput`](kernels/kda.md#KDAOutput), [`SSM1Output`](kernels/ssm1.md#SSM1Output), [`SSM2Output`](kernels/ssm2.md#SSM2Output), [`OperationOutput`](_operation_impl.md#OperationOutput)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`__call__`](../modules/kimi_linear/modeling_kimi_linear.md#KimiDeltaAttention.__call__), [`__call__`](../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextLinearAttention.__call__), [`forward_mla`](../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.forward_mla), [`__call__`](../modules/falcon_h1/modeling_falcon_h1.md#FalconH1Mixer.__call__), [`forward_mla`](../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.forward_mla), [`forward`](../layers/attention/_flexible.md#FlexibleAttentionModule.forward), [`forward_mla`](../layers/attention/_unified.md#UnifiedAttention.forward_mla), [`__call__`](../modules/falcon_mamba/modeling_falcon_mamba.md#FalconMambaMixer.__call__), [`forward_alibi`](../layers/attention/_unified.md#UnifiedAttention.forward_alibi), [`forward_native`](kernels/paged_flash_attention.md#PagedFlashAttn.forward_native), [`__call__`](../modules/mamba2/modeling_mamba2.md#Mamba2Mixer.__call__), [`__call__`](../modules/mamba/modeling_mamba.md#MambaMixer.__call__), [`_forward_v2_turboquant`](kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v2_turboquant), [`_forward_v3_turboquant`](kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v3_turboquant), [`__call__`](../modules/gidd/modeling_gidd.md#GiddAttention.__call__), [`__call__`](../modules/stablelm/modeling_stablelm.md#StableLmAttention.__call__), [`forward`](../modules/dbrx/modeling_dbrx.md#DbrxAttention.forward), [`_forward_with_kv_capture`](../modules/gemma4/modeling_gemma4.md#Gemma4Attention._forward_with_kv_capture), [`forward_core`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttn.forward_core), [`forward_core`](kernels/multi_latent_ragged_page_attention.md#MultiLatentRaggedPageAttnV2.forward_core), [`__call__`](../modules/grok_1/modeling_grok_1.md#Grok1Attention.__call__), [`_forward_v3_standard`](kernels/ragged_page_attention.md#_RaggedPageAttn._forward_v3_standard), [`forward_native`](kernels/unified_attention.md#UnifiedAttn.forward_native), [`__call__`](../modules/phi3/modeling_phi3.md#Phi3Attention.__call__), [`__call__`](../modules/opt/modeling_opt.md#OPTAttention.__call__), [`__call__`](../modules/roberta/modeling_roberta.md#RobertaSelfAttention.__call__), [`forward_alibi`](../modules/mosaic_mpt/modeling_mosaic.md#MptAttention.forward_alibi), [`forward_native`](kernels/blocksparse_attention.md#BlockSparseAttn.forward_native), [`__call__`](../modules/gemma4/modeling_gemma4.md#Gemma4Attention.__call__), [`__call__`](../modules/gpt2/modeling_gpt2.md#GPT2Attention.__call__), [`__call__`](../modules/whisper/modeling_whisper.md#WhisperAttention.__call__), [`__call__`](../modules/gemma4/modeling_gemma4.md#Gemma4VisionAttention.__call__), [`forward_native`](kernels/decode_attention.md#AutoRegressiveDecodeAttn.forward_native), [`forward_native`](kernels/ring_attention.md#RingAttn.forward_native), [`forward_v2`](kernels/ragged_page_attention.md#_RaggedPageAttn.forward_v2), [`forward_native`](kernels/flash_attention.md#FlashAttn.forward_native), [`__call__`](../modules/pixtral/modeling_pixtral.md#PixtralAttention.__call__), [`output_infer`](kernels/ssm2.md#output_infer), [`forward_native`](kernels/scaled_dot_product_attention.md#ScaledDotProductAttn.forward_native)  (+96 more)

