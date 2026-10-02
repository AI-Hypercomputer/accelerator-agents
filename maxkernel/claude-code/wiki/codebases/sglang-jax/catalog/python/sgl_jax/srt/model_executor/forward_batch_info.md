---
title: 'Module: python/sgl_jax/srt/model_executor/forward_batch_info.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/model_executor/forward_batch_info.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.model_executor.forward_batch_info`/
symbols:
  ForwardMode: ForwardMode#
  ForwardBatch.init_new: ForwardBatch#init_new().
  ForwardBatch.tree_unflatten: ForwardBatch#tree_unflatten().
  ForwardBatch.tree_flatten: ForwardBatch#tree_flatten().
  CaptureHiddenMode: CaptureHiddenMode#
  ForwardBatch: ForwardBatch#
  ForwardMode.is_extend: ForwardMode#is_extend().
  ForwardMode.EXTEND: ForwardMode#EXTEND.
  ForwardBatch.forward_mode: ForwardBatch#forward_mode.
  ForwardMode.is_decode: ForwardMode#is_decode().
  ForwardMode.DECODE: ForwardMode#DECODE.
  CaptureHiddenMode.FULL: CaptureHiddenMode#FULL.
  ForwardBatch.spec_info: ForwardBatch#spec_info.
  logger: logger.
  ForwardMode.is_decode_or_idle: ForwardMode#is_decode_or_idle().
  ForwardBatch.input_ids: ForwardBatch#input_ids.
  ForwardMode.is_extend_or_draft_extend_or_mixed: ForwardMode#is_extend_or_draft_extend_or_mixed().
  ForwardMode.is_cuda_graph: ForwardMode#is_cuda_graph().
  CaptureHiddenMode.LAST: CaptureHiddenMode#LAST.
  CaptureHiddenMode.parse: CaptureHiddenMode#parse().
  ForwardBatch.out_cache_loc: ForwardBatch#out_cache_loc.
  ForwardMode.is_target_verify: ForwardMode#is_target_verify().
  ForwardBatch.seq_lens: ForwardBatch#seq_lens.
  ForwardBatch.attn_backend: ForwardBatch#attn_backend.
  ForwardMode.is_idle: ForwardMode#is_idle().
  ForwardMode.is_draft_extend: ForwardMode#is_draft_extend().
  CaptureHiddenMode.NULL: CaptureHiddenMode#NULL.
  ForwardBatch.bid: ForwardBatch#bid.
  ForwardMode.is_dummy_first: ForwardMode#is_dummy_first().
  CaptureHiddenMode.need_capture: CaptureHiddenMode#need_capture().
  CaptureHiddenMode.is_full: CaptureHiddenMode#is_full().
  CaptureHiddenMode.is_last: CaptureHiddenMode#is_last().
  ForwardBatch.expert_location_metadata: ForwardBatch#expert_location_metadata.
  ForwardBatch.spec_algorithm: ForwardBatch#spec_algorithm.
  ForwardBatch.capture_hidden_mode: ForwardBatch#capture_hidden_mode.
  ForwardMode.IDLE: ForwardMode#IDLE.
  ForwardMode.TARGET_VERIFY: ForwardMode#TARGET_VERIFY.
  ForwardMode.DRAFT_EXTEND: ForwardMode#DRAFT_EXTEND.
  ForwardMode.is_mixed: ForwardMode#is_mixed().
  ForwardBatch.cache_loc: ForwardBatch#cache_loc.
  ForwardBatch.__repr__: ForwardBatch#__repr__().
  ForwardBatch.get_token_valid_mask: ForwardBatch#get_token_valid_mask().
  ForwardBatch.batch_size: ForwardBatch#batch_size.
  ForwardBatch.positions: ForwardBatch#positions.
  ForwardBatch.extend_prefix_lens: ForwardBatch#extend_prefix_lens.
  ForwardBatch.extend_seq_lens: ForwardBatch#extend_seq_lens.
  ForwardBatch.lora_scalings: ForwardBatch#lora_scalings.
  ForwardBatch.lora_token_indices: ForwardBatch#lora_token_indices.
  ForwardMode.is_prefill: ForwardMode#is_prefill().
  ForwardMode.MIXED: ForwardMode#MIXED.
  ForwardBatch.req_pool_indices: ForwardBatch#req_pool_indices.
  ForwardBatch.lora_ranks: ForwardBatch#lora_ranks.
  ForwardBatch.attention_mask: ForwardBatch#attention_mask.
  ForwardBatch.input_embedding: ForwardBatch#input_embedding.
  ForwardBatch.mrope_positions: ForwardBatch#mrope_positions.
  ForwardBatch.apply_for_deepstack: ForwardBatch#apply_for_deepstack.
  ForwardBatch.deepstack_visual_embedding: ForwardBatch#deepstack_visual_embedding.
  ForwardBatch.recurrent_indices: ForwardBatch#recurrent_indices.
  ForwardMode.DUMMY_FIRST: ForwardMode#DUMMY_FIRST.
  ForwardBatch.deterministic: ForwardBatch#deterministic.
  ForwardBatch.recurrent_cow_src_indices: ForwardBatch#recurrent_cow_src_indices.
  ForwardBatch.lora_ids: ForwardBatch#lora_ids.
  ForwardBatch.trace_request_ids: ForwardBatch#trace_request_ids.
  ForwardBatch.trace_request_objects: ForwardBatch#trace_request_objects.
  CaptureHiddenMode.__lt__: CaptureHiddenMode#__lt__().
---
# Module: [`python/sgl_jax/srt/model_executor/forward_batch_info.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py)

## Classes
### `CaptureHiddenMode`  ·  implements/extends IntEnum
- def: [`python/sgl_jax/srt/model_executor/forward_batch_info.py:115`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L115)
- signature: `class CaptureHiddenMode(IntEnum):`
- members:
  - `is_full(self)` — [`L126`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L126)
  - `is_last(self)` — [`L129`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L129)
  - `need_capture(self)` — [`L123`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L123)
  - `parse(mode: int)` — [`L135`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L135)
  - `FULL` — [`L121`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L121)
  - `LAST` — [`L119`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L119)
  - `NULL` — [`L117`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L117)
- protocol/private: `__lt__`[`L132`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L132)
- used by: `get_model_worker_batch`, `_get_spec_decode_mwb_dp`, [`prepare_for_extend_after_verify`](../speculative/eagle_util.md#EagleDraftInput.prepare_for_extend_after_verify), [`_make_dummy_batch`](compilation_manager.md#CompilationManager._make_dummy_batch), [`padding_for_decode`](../speculative/eagle_draft_worker.md#EagleDraftWorker.padding_for_decode), [`__call__`](../layers/logits_processor.md#LogitsProcessor.__call__), [`draft_extend_for_prefill`](../speculative/multi_layer_draft_worker.md#MultiLayerDraftWorker.draft_extend_for_prefill), [`spec_prefill`](../speculative/draft_extend_fused.md#spec_prefill), [`draft_extend_for_prefill`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_prefill), [`draft`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft), [`precompile_spec_decode`](../speculative/eagle_worker.md#EAGLEWorker.precompile_spec_decode), [`_prepare_verify`](../speculative/draft_extend_fused.md#_prepare_verify), [`prepare_for_verify`](../speculative/eagle_util.md#EagleVerifyInput.prepare_for_verify), [`forward_target_extend`](../speculative/base_worker.md#BaseSpecWorker.forward_target_extend), `capture_hidden_mode`, [`prepare_for_draft_decode`](../speculative/eagle_util.md#EagleDraftInput.prepare_for_draft_decode), [`capture_hidden_mode`](../speculative/eagle_util.md#EagleDraftInput.capture_hidden_mode), [`prepare_forward_batch_for_prefill`](../speculative/draft_extend_fused.md#prepare_forward_batch_for_prefill), [`capture_hidden_mode`](../layers/logits_processor.md#LogitsMetadata.capture_hidden_mode), [`capture_hidden_mode`](../speculative/eagle_util.md#EagleVerifyInput.capture_hidden_mode), `speculative_num_draft_tokens`, [`capture_hidden_mode`](forward_batch_info.md#ForwardBatch.capture_hidden_mode)

### `ForwardBatch`
- def: [`python/sgl_jax/srt/model_executor/forward_batch_info.py:148`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L148)
- doc: Store all inputs of a forward pass.
- signature: `class ForwardBatch:`
- members:
  - `get_token_valid_mask(self, num_tokens: int, out_sharding: NamedSharding | None = None)` — [`L304`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L304) — Return a per-token validity mask for padded batches.
  - `init_new(cls, batch: ModelWorkerBatch, model_runner: ModelRunner)` — [`L335`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L335) — documented in [python-sgl_jax-srt-model_executor-forward_batch_info](../../../../../concepts/python-sgl_jax-srt-model_executor-forward_batch_info.md)
  - `tree_flatten(self)` — [`L209`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L209) — documented in [python-sgl_jax-srt-model_executor-forward_batch_info](../../../../../concepts/python-sgl_jax-srt-model_executor-forward_batch_info.md)
  - `tree_unflatten(cls, aux_data, children)` — [`L244`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L244) — documented in [python-sgl_jax-srt-model_executor-forward_batch_info](../../../../../concepts/python-sgl_jax-srt-model_executor-forward_batch_info.md)
  - `apply_for_deepstack` — [`L201`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L201)
  - `attention_mask` — [`L193`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L193)
  - `attn_backend` — [`L168`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L168)
  - `batch_size` — [`L156`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L156)
  - `bid` — [`L152`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L152)
  - `cache_loc` — [`L170`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L170)
  - `capture_hidden_mode` — [`L190`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L190)
  - `deepstack_visual_embedding` — [`L202`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L202)
  - `deterministic` — [`L194`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L194)
  - `expert_location_metadata` — [`L183`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L183)
  - `extend_prefix_lens` — [`L173`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L173)
  - `extend_seq_lens` — [`L174`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L174)
  - `forward_mode` — [`L154`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L154)
  - `input_embedding` — [`L196`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L196)
  - `input_ids` — [`L158`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L158)
  - `lora_ids` — [`L177`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L177)
  - `lora_ranks` — [`L180`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L180)
  - `lora_scalings` — [`L178`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L178)
  - `lora_token_indices` — [`L179`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L179)
  - `mrope_positions` — [`L198`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L198)
  - `out_cache_loc` — [`L164`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L164)
  - `positions` — [`L166`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L166)
  - `recurrent_cow_src_indices` — [`L207`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L207)
  - `recurrent_indices` — [`L205`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L205)
  - `req_pool_indices` — [`L160`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L160)
  - `seq_lens` — [`L162`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L162)
  - `spec_algorithm` — [`L189`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L189)
  - `spec_info` — [`L188`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L188)
  - `trace_request_ids` — [`L185`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L185)
  - `trace_request_objects` — [`L186`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L186)
- protocol/private: `__repr__`[`L279`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L279)
- uses (calls/refs, reference-scoped): [`hf_config`](../configs/model_config.md#ModelConfig.hf_config), [`ForwardMode`](forward_batch_info.md#ForwardMode), `ModelWorkerBatch`, [`model_config`](model_runner.md#ModelRunner.model_config), `seq_lens`, `spec_info_padded`, [`CaptureHiddenMode`](forward_batch_info.md#CaptureHiddenMode), `forward_mode`, [`device_array`](../utils/jax_utils.md#device_array), [`EagleDraftInput`](../speculative/eagle_util.md#EagleDraftInput), [`SpeculativeAlgorithm`](../speculative/spec_info.md#SpeculativeAlgorithm), [`ModelRunner`](model_runner.md#ModelRunner), [`mesh`](model_runner.md#ModelRunner.mesh), [`AttentionBackend`](../layers/attention/base_attn_backend.md#AttentionBackend), `extend_seq_lens`, [`attn_backend`](model_runner.md#ModelRunner.attn_backend), `cache_loc`, `capture_hidden_mode`, [`ExpertLocationMetadata`](../eplb/expert_location.md#ExpertLocationMetadata), `input_ids`, [`EagleVerifyInput`](../speculative/eagle_util.md#EagleVerifyInput), [`get_global_expert_location_metadata`](../eplb/expert_location.md#get_global_expert_location_metadata), `positions`, `spec_algorithm`, `bid`, `lora_ids`, `req_pool_indices`, `out_cache_loc`, `extend_prefix_lens`, `mrope_positions`, `recurrent_indices`, [`need_attention_mask`](../configs/model_config.md#need_attention_mask), `apply_for_deepstack`, `input_embedding`, `lora_scalings`, `recurrent_cow_src_indices`, `deepstack_visual_embedding`, `lora_ranks`, `lora_token_indices`
- used by: [`draft_extend_for_prefill`](../speculative/multi_layer_draft_worker.md#MultiLayerDraftWorker.draft_extend_for_prefill), [`forward_batch_generation`](../managers/tp_worker.md#ModelWorker.forward_batch_generation), [`draft_forward`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_forward), [`spec_prefill`](../speculative/draft_extend_fused.md#spec_prefill), [`__call__`](../layers/attention/linear/kda_backend.md#KDAAttnBackend.__call__), [`draft_extend_for_prefill`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_prefill), [`_make_forward_batch`](../speculative/draft_extend_fused.md#_make_forward_batch), [`spec_decode_verify`](../speculative/draft_extend_fused.md#spec_decode_verify), [`draft_extend_for_decode`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_decode), [`__call__`](../layers/attention/flashattention_backend.md#FlashAttention.__call__), [`__call__`](../layers/attention/native_backend.md#NativeAttention.__call__), [`launch_fused_draft_extend_for_decode`](../speculative/draft_extend_fused.md#launch_fused_draft_extend_for_decode), [`cur_sampling_info`](../managers/tp_worker_overlap_thread.md#ModelWorkerClient.cur_sampling_info), [`__call__`](../layers/attention/mla_backend.md#MLAAttentionBackend.__call__), [`_precompile_decode`](compilation_manager.md#CompilationManager._precompile_decode), [`_precompile_extend`](compilation_manager.md#CompilationManager._precompile_extend), [`__call__`](../layers/attention/linear/gdn_backend.md#GDNAttnBackend.__call__), [`__call__`](../layers/attention/linear/lightning_backend.md#LightningAttnBackend.__call__), [`forward`](model_runner.md#ModelRunner.forward), [`_forward_raw`](model_runner.md#ModelRunner._forward_raw), [`_get_and_update_kv_cache`](../layers/attention/native_backend.md#NativeAttention._get_and_update_kv_cache), [`_forward`](model_runner.md#ModelRunner._forward), [`__call__`](../lora/layers.md#LoRALinear.__call__), [`prepare_forward_batch_for_prefill`](../speculative/draft_extend_fused.md#prepare_forward_batch_for_prefill), [`update_forward_batch_info`](../speculative/eagle_draft_worker.md#update_forward_batch_info), [`__call__`](../layers/attention/hybrid_linear_attn_backend.md#HybridLinearAttnBackend.__call__), [`logger`](../layers/attention/linear/lightning_backend.md#logger), [`_get_fused_kv_cache`](../layers/attention/flashattention_backend.md#FlashAttention._get_fused_kv_cache), [`forward_idle`](model_runner.md#ModelRunner.forward_idle), [`__call__`](../layers/radix_attention.md#RadixAttention.__call__), [`get_batch`](../lora/context_manager.md#LoraBatchContext.get_batch), [`set_batch`](../lora/context_manager.md#LoraBatchContext.set_batch), [`__call__`](../layers/attention/base_attn_backend.md#AttentionBackend.__call__), [`__call__`](../layers/radix_lightning_attention.md#RadixLightningAttention.__call__), [`__call__`](../layers/radix_linear_attention.md#RadixLinearAttention.__call__), [`__call__`](../lora/layers.md#BaseLayerWithLoRA.__call__)

### `ForwardMode`  ·  implements/extends IntEnum
- def: [`python/sgl_jax/srt/model_executor/forward_batch_info.py:47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L47) — documented in [python-sgl_jax-srt-layers-logits_processor](../../../../../concepts/python-sgl_jax-srt-layers-logits_processor.md)
- signature: `class ForwardMode(IntEnum):`
- members:
  - `is_cuda_graph(self)` — [`L100`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L100)
  - `is_decode(self)` — [`L78`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L78)
  - `is_decode_or_idle(self)` — [`L110`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L110)
  - `is_draft_extend(self)` — [`L90`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L90)
  - `is_dummy_first(self)` — [`L107`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L107)
  - `is_extend(self)` — [`L70`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L70) — documented in [python-sgl_jax-srt-model_executor-forward_batch_info](../../../../../concepts/python-sgl_jax-srt-model_executor-forward_batch_info.md)
  - `is_extend_or_draft_extend_or_mixed(self)` — [`L93`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L93)
  - `is_idle(self)` — [`L84`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L84)
  - `is_mixed(self)` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L81)
  - `is_prefill(self)` — [`L67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L67)
  - `is_target_verify(self)` — [`L87`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L87)
  - `DECODE` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L52)
  - `DRAFT_EXTEND` — [`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L61)
  - `DUMMY_FIRST` — [`L65`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L65)
  - `EXTEND` — [`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L50)
  - `IDLE` — [`L56`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L56)
  - `MIXED` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L54)
  - `TARGET_VERIFY` — [`L59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L59)
- used by: `get_model_worker_batch`, `prepare_for_extend`, [`prepare_for_extend_after_verify`](../speculative/eagle_util.md#EagleDraftInput.prepare_for_extend_after_verify), [`_make_dummy_batch`](compilation_manager.md#CompilationManager._make_dummy_batch), [`__call__`](../layers/logits_processor.md#LogitsProcessor.__call__), [`forward_batch_speculative_generation`](../speculative/base_worker.md#BaseSpecWorker.forward_batch_speculative_generation), `prepare_for_decode`, [`run_batch`](../managers/scheduler.md#Scheduler.run_batch), [`draft_extend_for_prefill`](../speculative/multi_layer_draft_worker.md#MultiLayerDraftWorker.draft_extend_for_prefill), [`result_queue`](../managers/scheduler.md#Scheduler.result_queue), [`_run_speculative_batch`](../managers/scheduler.md#Scheduler._run_speculative_batch), [`__call__`](../layers/attention/linear/kda_backend.md#KDAAttnBackend.__call__), [`draft_extend_for_prefill`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_prefill), [`get_eagle_forward_metadata`](../layers/attention/flashattention_backend.md#FlashAttention.get_eagle_forward_metadata), [`get_eagle_multi_step_metadata`](../layers/attention/flashattention_backend.md#FlashAttention.get_eagle_multi_step_metadata), `mix_with_running`, `forward_mode`, [`__call__`](../layers/attention/flashattention_backend.md#FlashAttention.__call__), [`__call__`](../layers/attention/native_backend.md#NativeAttention.__call__), [`get_next_batch_to_run`](../managers/scheduler.md#Scheduler.get_next_batch_to_run), [`from_model_worker_batch`](../sampling/sampling_batch_info.md#SamplingMetadata.from_model_worker_batch), `forward_mode`, [`get_forward_metadata`](../layers/attention/flashattention_backend.md#FlashAttention.get_forward_metadata), [`get_forward_metadata`](../layers/attention/mla_backend.md#MLAAttentionBackend.get_forward_metadata), `maybe_evict_swa`, [`_precompile_decode`](compilation_manager.md#CompilationManager._precompile_decode), [`_precompile_extend`](compilation_manager.md#CompilationManager._precompile_extend), [`__call__`](../layers/attention/linear/gdn_backend.md#GDNAttnBackend.__call__), [`from_model_worker_batch`](../layers/logits_processor.md#LogitsMetadata.from_model_worker_batch), [`get_forward_metadata`](../layers/attention/hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.get_forward_metadata), [`prepare_for_verify`](../speculative/eagle_util.md#EagleVerifyInput.prepare_for_verify), `prepare_for_idle`, [`process_batch_result`](../managers/scheduler.md#Scheduler.process_batch_result), [`__call__`](../layers/attention/linear/lightning_backend.md#LightningAttnBackend.__call__), [`precompile_spec_extend`](../speculative/eagle_worker.md#EAGLEWorker.precompile_spec_extend), [`on_forward_end`](../layers/routed_experts_capturer.md#_RoutedExpertsCapturerReal.on_forward_end), `_generate_trace_info`, [`prepare_lora_batch`](../lora/backend/bgmv_backend.md#BgmvLoRABackend.prepare_lora_batch), [`forward_mode`](forward_batch_info.md#ForwardBatch.forward_mode), [`forward_batch_speculative_decode_overlap`](../speculative/base_worker.md#BaseSpecWorker.forward_batch_speculative_decode_overlap)  (+20 more)

## Module values
- `logger` — [`L38`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/forward_batch_info.py#L38)

