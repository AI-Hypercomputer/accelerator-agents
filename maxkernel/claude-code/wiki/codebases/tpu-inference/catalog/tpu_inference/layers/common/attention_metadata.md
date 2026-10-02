---
title: 'Module: tpu_inference/layers/common/attention_metadata.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/attention_metadata.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.attention_metadata`/AttentionMetadata#
symbols:
  AttentionMetadata: ''
  AttentionMetadata.input_positions: input_positions.
  AttentionMetadata.seq_lens: seq_lens.
  AttentionMetadata.block_tables: block_tables.
  AttentionMetadata.query_start_loc: query_start_loc.
  AttentionMetadata.request_distribution: request_distribution.
  AttentionMetadata.mamba_state_indices: mamba_state_indices.
  AttentionMetadata.padded_num_reqs: padded_num_reqs.
---
# Module: [`tpu_inference/layers/common/attention_metadata.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_metadata.py)

## Classes
### `AttentionMetadata`
- def: [`tpu_inference/layers/common/attention_metadata.py:34`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_metadata.py#L34) — documented in [root](../../../../concepts/root.md)
- signature: `class AttentionMetadata:`
- members:
  - `block_tables` — [`L39`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_metadata.py#L39)
  - `input_positions` — [`L36`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_metadata.py#L36) — documented in [tpu_inference-layers-common-attention_metadata](../../../../concepts/tpu_inference-layers-common-attention_metadata.md)
  - `mamba_state_indices` — [`L53`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_metadata.py#L53)
  - `padded_num_reqs` — [`L60`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_metadata.py#L60)
  - `query_start_loc` — [`L43`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_metadata.py#L43) — documented in [tpu_inference-layers-common-attention_metadata](../../../../concepts/tpu_inference-layers-common-attention_metadata.md)
  - `request_distribution` — [`L45`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_metadata.py#L45)
  - `seq_lens` — [`L41`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/attention_metadata.py#L41) — documented in [tpu_inference-layers-common-attention_metadata](../../../../concepts/tpu_inference-layers-common-attention_metadata.md)
- used by: `_sample_from_logits`, [`_precompile_continue_decode`](../../runner/compilation_manager.md#CompilationManager._precompile_continue_decode), `_prepare_inputs`, [`__call__`](../jax/attention/llama4_attention.md#Llama4VisionAttention.__call__), [`_precompile_mtp_helpers`](../../runner/compilation_manager.md#CompilationManager._precompile_mtp_helpers), [`_precompile_eagle3_helpers`](../../runner/compilation_manager.md#CompilationManager._precompile_eagle3_helpers), [`__call__`](../jax/attention/llama4_attention.md#Llama4Attention.__call__), [`__call__`](../../models/jax/gemma4.md#Gemma4Attention.__call__), [`__call__`](../jax/attention/attention.md#Attention.__call__), [`_mla_ragged_paged_attention`](attention_interface.md#mla_attention._mla_ragged_paged_attention), [`__call__`](../jax/attention/gpt_oss_attention.md#GptOssAttention.__call__), [`compute_attention`](../../models/jax/deepseek_v3.md#DeepseekV3Attention.compute_attention), [`_precompile_backbone_helper`](../../runner/compilation_manager.md#CompilationManager._precompile_backbone_helper), [`propose_eagle3_draft_token_ids`](../../runner/speculative_decoding_manager.md#SpeculativeDecodingManager.propose_eagle3_draft_token_ids), [`__call__`](../../models/jax/qwen3.md#Qwen3Attention.__call__), [`continue_decode`](../../runner/decode_loop.md#continue_decode), [`__call__`](../../models/jax/gemma4_mtp.md#Gemma4MTPAttention.__call__), [`__call__`](../../models/jax/gemma4.md#Gemma4DecoderLayer.__call__), [`__call__`](../../models/jax/llama3.md#LlamaAttention.__call__), [`__call__`](../../models/jax/qwen2.md#Qwen2Attention.__call__), [`compute_attention`](../../models/jax/deepseek_v3.md#DeepseekV3MLA.compute_attention), [`propose_draft_token_ids`](../../runner/speculative_decoding_manager.md#SpeculativeDecodingManager.propose_draft_token_ids), [`qwix_quantize_nnx_model`](../../models/jax/utils/qwix/qwix_utils.md#qwix_quantize_nnx_model), [`__call__`](../../models/jax/deepseek_v3.md#DeepseekV3BaseAttention.__call__), [`_propose`](../../spec_decode/jax/eagle3.md#Eagle3Proposer._propose), [`attention`](../jax/attention/attention.md#Attention.attention), [`step_fun_impl`](../../models/vllm/vllm_model_wrapper.md#VllmModelWrapper.step_fun_impl), [`__call__`](../../models/jax/llama4.md#Llama4ForCausalLM.__call__), [`__call__`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.__call__), [`attention`](../jax/attention/gpt_oss_attention.md#GptOssAttention.attention), [`__call__`](../../models/jax/llama3.md#LlamaModel.__call__), [`attention`](attention_interface.md#attention), `build_attn`, [`build_attn`](../../runner/compilation_manager.md#CompilationManager.build_attn), [`_run_one_step`](../../runner/decode_loop.md#_decode_core._run_one_step), [`draft_step_fun_impl`](../../models/vllm/vllm_model_wrapper.md#VllmModelWrapper.draft_step_fun_impl), [`prepare_inputs`](../../spec_decode/jax/eagle3.md#Eagle3Proposer.prepare_inputs), [`__call__`](../../models/jax/gemma4_mtp.md#Gemma4MTPDecoderLayer.__call__), [`__call__`](../../models/jax/gemma4.md#Gemma4Model.__call__), [`__call__`](../../models/jax/llama_eagle3.md#Eagle3LlamaDecoderLayer.__call__)  (+31 more)

