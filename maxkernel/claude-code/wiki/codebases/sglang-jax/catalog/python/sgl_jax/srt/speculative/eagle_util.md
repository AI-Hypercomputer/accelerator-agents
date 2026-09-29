---
title: 'Module: python/sgl_jax/srt/speculative/eagle_util.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/speculative/eagle_util.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.speculative.eagle_util`/
symbols:
  EagleDraftInput.prepare_for_extend_after_verify: EagleDraftInput#prepare_for_extend_after_verify().
  EagleDraftInput.prepare_for_decode: EagleDraftInput#prepare_for_decode().
  EagleDraftInput.allocate_lens: EagleDraftInput#allocate_lens.
  EagleVerifyInput.sample: EagleVerifyInput#sample().
  EagleDraftInput: EagleDraftInput#
  EagleVerifyInput.prepare_for_verify: EagleVerifyInput#prepare_for_verify().
  EagleDraftInput.tree_flatten: EagleDraftInput#tree_flatten().
  EagleDraftInput.hidden_states: EagleDraftInput#hidden_states.
  EagleDraftInput.tree_unflatten: EagleDraftInput#tree_unflatten().
  EagleVerifyInput.tree_flatten: EagleVerifyInput#tree_flatten().
  EagleDraftInput.verified_id: EagleDraftInput#verified_id.
  EagleVerifyInput.tree_unflatten: EagleVerifyInput#tree_unflatten().
  EagleDraftInput.merge_batch: EagleDraftInput#merge_batch().
  EagleDraftInput.filter_batch: EagleDraftInput#filter_batch().
  EagleDraftInput.new_seq_lens: EagleDraftInput#new_seq_lens.
  EagleDraftInput.topk_index: EagleDraftInput#topk_index.
  EagleDraftInput.topk_p: EagleDraftInput#topk_p.
  EagleDraftInput.prepare_for_extend_after_target_prefill: EagleDraftInput#prepare_for_extend_after_target_prefill().
  EagleDraftInput.prepare_for_draft_decode: EagleDraftInput#prepare_for_draft_decode().
  EagleDraftInput.capture_hidden_mode: EagleDraftInput#capture_hidden_mode.
  EagleDraftInput.future_indices: EagleDraftInput#future_indices.
  EagleDraftInput.accept_length_cpu: EagleDraftInput#accept_length_cpu.
  EagleVerifyInput: EagleVerifyInput#
  EagleVerifyInput.draft_token_num: EagleVerifyInput#draft_token_num.
  EagleDraftInput.accept_length: EagleDraftInput#accept_length.
  EagleDraftInput.num_tokens_per_batch: EagleDraftInput#num_tokens_per_batch.
  EagleDraftInput.num_tokens_for_logprob_per_batch: EagleDraftInput#num_tokens_for_logprob_per_batch.
  EagleDraftInput.trim_to_length: EagleDraftInput#trim_to_length().
  EagleDraftInput.ALLOC_LEN_PER_DECODE: EagleDraftInput#ALLOC_LEN_PER_DECODE.
  EagleVerifyInput.draft_token: EagleVerifyInput#draft_token.
  EagleVerifyInput.retrive_index: EagleVerifyInput#retrive_index.
  build_tree_kernel_efficient: build_tree_kernel_efficient().
  EagleVerifyInput.spec_steps: EagleVerifyInput#spec_steps.
  _as_int32_array: _as_int32_array().
  EagleDraftInput._to_device: EagleDraftInput#_to_device().
  EagleDraftInput.resolve_pending_draft_extend_result: EagleDraftInput#resolve_pending_draft_extend_result().
  EagleVerifyInput.capture_hidden_mode: EagleVerifyInput#capture_hidden_mode.
  EagleVerifyInput.retrive_next_token: EagleVerifyInput#retrive_next_token.
  EagleVerifyInput.retrive_next_sibling: EagleVerifyInput#retrive_next_sibling.
  EagleVerifyOutput.logits_output: EagleVerifyOutput#logits_output.
  EagleDraftInput.get_spec_adjust_token_coefficient: EagleDraftInput#get_spec_adjust_token_coefficient().
  EagleVerifyInput.custom_mask: EagleVerifyInput#custom_mask.
  EagleVerifyInput.positions: EagleVerifyInput#positions.
  build_tree_mask_for_draft_decode: build_tree_mask_for_draft_decode().
  EagleDraftInput._ensure_host: EagleDraftInput#_ensure_host().
  EagleVerifyInput.retrive_cum_len: EagleVerifyInput#retrive_cum_len.
  EagleVerifyInput.seq_lens_cpu: EagleVerifyInput#seq_lens_cpu.
  EagleVerifyInput.topk: EagleVerifyInput#topk.
  EagleVerifyInput.seq_lens_sum: EagleVerifyInput#seq_lens_sum.
  _generate_simulated_accept_index: _generate_simulated_accept_index().
  SIMULATE_ACC_METHOD: SIMULATE_ACC_METHOD.
  get_last_loc_large_page_size_top_k_1: get_last_loc_large_page_size_top_k_1().
  get_last_loc_large_page_size_large_top_k: get_last_loc_large_page_size_large_top_k().
  EagleDraftInput.get_logical_token_num: EagleDraftInput#get_logical_token_num().
  EagleDraftInput.get_allocated_token_num: EagleDraftInput#get_allocated_token_num().
  EagleDraftInput.new_tokens_required_next_decode: EagleDraftInput#new_tokens_required_next_decode().
  EagleVerifyOutput.draft_input: EagleVerifyOutput#draft_input.
  EagleVerifyInput.get_spec_adjust_token_coefficient: EagleVerifyInput#get_spec_adjust_token_coefficient().
  EagleVerifyInput.get_verify_token_num: EagleVerifyInput#get_verify_token_num().
  SIMULATE_ACC_LEN: SIMULATE_ACC_LEN.
  get_last_loc_jax_array: get_last_loc_jax_array().
  build_chain_verify_inputs: build_chain_verify_inputs().
  build_chain_verify_inputs_device: build_chain_verify_inputs_device().
  EagleDraftInput.kv_indptr: EagleDraftInput#kv_indptr.
  EagleDraftInput.kv_indices: EagleDraftInput#kv_indices.
  EagleDraftInput.seq_lens_for_draft_extend: EagleDraftInput#seq_lens_for_draft_extend.
  EagleDraftInput.req_pool_indices_for_draft_extend: EagleDraftInput#req_pool_indices_for_draft_extend.
  EagleDraftInput.pending_draft_extend_result: EagleDraftInput#pending_draft_extend_result.
  EagleVerifyOutput: EagleVerifyOutput#
  _is_jax_leaf: _is_jax_leaf().
  build_tree_kernel_efficient_preprocess: build_tree_kernel_efficient_preprocess().
  _extract_parent_branch_indices: _extract_parent_branch_indices().
  EagleVerifyOutput.verified_id: EagleVerifyOutput#verified_id.
  EagleVerifyOutput.accept_length_per_req_cpu: EagleVerifyOutput#accept_length_per_req_cpu.
  EagleVerifyOutput.accepted_indices: EagleVerifyOutput#accepted_indices.
  assign_req_to_token_pool: assign_req_to_token_pool().
  EagleDraftInput.is_draft_input: EagleDraftInput#is_draft_input().
  EagleDraftInput.is_verify_input: EagleDraftInput#is_verify_input().
  EagleDraftInput.get_verify_token_num: EagleDraftInput#get_verify_token_num().
  EagleVerifyInput.is_draft_input: EagleVerifyInput#is_draft_input().
  EagleVerifyInput.is_verify_input: EagleVerifyInput#is_verify_input().
  EagleVerifyInput.get_logical_token_num: EagleVerifyInput#get_logical_token_num().
  EagleVerifyInput.get_allocated_token_num: EagleVerifyInput#get_allocated_token_num().
  EagleVerifyInput.filter_batch: EagleVerifyInput#filter_batch().
  EagleVerifyInput.merge_batch: EagleVerifyInput#merge_batch().
---
# Module: [`python/sgl_jax/srt/speculative/eagle_util.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py)

## Classes
### `EagleDraftInput`
- def: [`python/sgl_jax/srt/speculative/eagle_util.py:441`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L441)
- doc: Next-round draft state — the only persistent cross-round spec state.
- signature: `class EagleDraftInput:`
- members:
  - `_ensure_host(self)` — [`L799`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L799) — Move device arrays to host (numpy) to avoid variable-shape device ops.
  - `filter_batch(self, new_indices: np.ndarray, has_been_filtered: bool = True)` — [`L818`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L818)
  - `get_allocated_token_num(self)` — [`L509`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L509)
  - `get_logical_token_num(self, bs: int)` — [`L504`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L504)
  - `get_spec_adjust_token_coefficient(self)` — [`L501`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L501)
  - `get_verify_token_num(self, bs: int)` — [`L512`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L512)
  - `is_draft_input(self)` — [`L495`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L495)
  - `is_verify_input(self)` — [`L498`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L498)
  - `merge_batch(self, spec_info: EagleDraftInput)` — [`L875`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L875)
  - `new_tokens_required_next_decode(self, requests, page_size: int)` — [`L515`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L515)
  - `prepare_for_decode(self, schedule_batch: ScheduleBatch)` — [`L720`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L720) — documented in [python-sgl_jax-srt-speculative-eagle_util](../../../../../concepts/python-sgl_jax-srt-speculative-eagle_util.md)
  - `prepare_for_draft_decode(self, model_worker_batch: ModelWorkerBatch, topk: int, num_steps: int)` — [`L783`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L783)
  - `prepare_for_extend_after_target_prefill(self, model_worker_batch: ModelWorkerBatch)` — [`L578`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L578)
  - `prepare_for_extend_after_verify(self, model_worker_batch: ModelWorkerBatch, draft_model_runner: Any, batch_output: GenerationBatchResult, speculative_num_draft_tokens: int)` — [`L614`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L614) — documented in [python-sgl_jax-srt-speculative-eagle_util](../../../../../concepts/python-sgl_jax-srt-speculative-eagle_util.md)
  - `resolve_pending_draft_extend_result(self)` — [`L792`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L792)
  - `tree_flatten(self)` — [`L526`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L526)
  - `tree_unflatten(cls, aux_data, children)` — [`L558`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L558)
  - `trim_to_length(self, n: int)` — [`L857`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L857)
  - `ALLOC_LEN_PER_DECODE` — [`L448`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L448)
  - `accept_length` — [`L466`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L466)
  - `accept_length_cpu` — [`L468`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L468)
  - `allocate_lens` — [`L486`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L486) — documented in [python-sgl_jax-srt-speculative-eagle_util](../../../../../concepts/python-sgl_jax-srt-speculative-eagle_util.md)
  - `capture_hidden_mode` — [`L460`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L460)
  - `future_indices` — [`L491`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L491)
  - `hidden_states` — [`L458`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L458)
  - `kv_indices` — [`L472`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L472)
  - `kv_indptr` — [`L471`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L471)
  - `new_seq_lens` — [`L489`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L489)
  - `num_tokens_for_logprob_per_batch` — [`L476`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L476)
  - `num_tokens_per_batch` — [`L475`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L475)
  - `pending_draft_extend_result` — [`L492`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L492)
  - `req_pool_indices_for_draft_extend` — [`L480`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L480)
  - `seq_lens_for_draft_extend` — [`L479`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L479)
  - `topk_index` — [`L454`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L454)
  - `topk_p` — [`L452`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L452)
  - `verified_id` — [`L464`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L464)
- protocol/private: `_to_device`[`L672`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L672)
- uses (calls/refs, reference-scoped): `reqs`, `reqs_info`, `seq_lens`, [`ForwardMode`](../model_executor/forward_batch_info.md#ForwardMode), `ModelWorkerBatch`, `seq_lens`, `spec_info_padded`, `tree_cache`, [`CaptureHiddenMode`](../model_executor/forward_batch_info.md#CaptureHiddenMode), `out_cache_loc`, `forward_mode`, `dp_size`, `ScheduleBatch`, [`device_array`](../utils/jax_utils.md#device_array), `token_to_kv_pool_allocator`, [`from_model_worker_batch`](../layers/logits_processor.md#LogitsMetadata.from_model_worker_batch), [`page_size`](../mem_cache/allocator.md#BaseTokenToKVPoolAllocator.page_size), `per_dp_bs_size`, `req_pool_indices`, `req_to_token_pool`, [`next_draft_input`](../managers/scheduler.md#GenerationBatchResult.next_draft_input), [`req_to_token`](../mem_cache/memory_pool.md#ReqToTokenPool.req_to_token), `seq_lens_sum`, `extend_seq_lens`, `spec_algorithm`, [`GenerationBatchResult`](../managers/scheduler.md#GenerationBatchResult), [`alloc_token_slots`](../mem_cache/common.md#alloc_token_slots), [`LogitsMetadata`](../layers/logits_processor.md#LogitsMetadata), [`FULL`](../model_executor/forward_batch_info.md#CaptureHiddenMode.FULL), [`use_legacy_eagle3_non_overlap`](overlap_utils.md#use_legacy_eagle3_non_overlap), `capture_hidden_mode`, `kv_committed_len`, `input_ids`, [`alloc_paged_token_slots_extend`](../mem_cache/common.md#alloc_paged_token_slots_extend), `logits_indices_selector`, `batch_size`, `real_bs`, `enable_overlap`, [`accept_lens`](../managers/scheduler.md#GenerationBatchResult.accept_lens), `positions`  (+28 more)
- used by: [`padding_for_decode`](eagle_draft_worker.md#EagleDraftWorker.padding_for_decode), [`forward_batch_speculative_generation`](base_worker.md#BaseSpecWorker.forward_batch_speculative_generation), `prepare_for_decode`, [`run_batch`](../managers/scheduler.md#Scheduler.run_batch), [`draft_extend_for_prefill`](multi_layer_draft_worker.md#MultiLayerDraftWorker.draft_extend_for_prefill), [`verify`](base_worker.md#BaseSpecWorker.verify), [`draft_forward`](eagle_draft_worker.md#EagleDraftWorker.draft_forward), [`spec_prefill`](draft_extend_fused.md#spec_prefill), [`draft_extend_for_prefill`](eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_prefill), [`draft`](eagle_draft_worker.md#EagleDraftWorker.draft), `filter_batch`, [`get_eagle_forward_metadata`](../layers/attention/flashattention_backend.md#FlashAttention.get_eagle_forward_metadata), [`spec_decode_verify`](draft_extend_fused.md#spec_decode_verify), [`precompile_spec_decode`](eagle_worker.md#EAGLEWorker.precompile_spec_decode), [`draft_extend_for_decode`](eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_decode), [`get_eagle_multi_step_metadata`](../layers/attention/flashattention_backend.md#FlashAttention.get_eagle_multi_step_metadata), `spec_info_padded`, `spec_info`, `merge_batch`, [`launch_fused_draft_extend_for_decode`](draft_extend_fused.md#launch_fused_draft_extend_for_decode), [`from_model_worker_batch`](../layers/logits_processor.md#LogitsMetadata.from_model_worker_batch), [`_worker`](eagle_draft_worker.md#EagleDraftWorker._worker), [`next_draft_input`](../managers/scheduler.md#GenerationBatchResult.next_draft_input), [`capture_for_decode`](eagle_draft_worker.md#EagleDraftWorker.capture_for_decode), [`forward_batch_speculative_decode_overlap`](base_worker.md#BaseSpecWorker.forward_batch_speculative_decode_overlap), [`_worker`](multi_layer_draft_worker.md#MultiLayerDraftWorker._worker), [`draft_forward`](multi_layer_draft_worker.md#MultiLayerDraftWorker.draft_forward), [`spec_decode_overlap`](draft_extend_fused.md#spec_decode_overlap), [`spec_info`](../model_executor/forward_batch_info.md#ForwardBatch.spec_info), [`update_forward_batch_info`](eagle_draft_worker.md#update_forward_batch_info), [`logger`](../model_executor/forward_batch_info.md#logger), [`draft_input`](eagle_util.md#EagleVerifyOutput.draft_input)

### `EagleVerifyInput`
- def: [`python/sgl_jax/srt/speculative/eagle_util.py:951`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L951)
- doc: Target-verify input. Implements `SpecInput`.
- signature: `class EagleVerifyInput:`
- members:
  - `filter_batch(self, new_indices: np.ndarray, has_been_filtered: bool = True)` — [`L1006`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L1006)
  - `get_allocated_token_num(self)` — [`L1000`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L1000)
  - `get_logical_token_num(self, bs: int)` — [`L997`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L997)
  - `get_spec_adjust_token_coefficient(self)` — [`L994`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L994)
  - `get_verify_token_num(self, bs: int)` — [`L1003`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L1003)
  - `is_draft_input(self)` — [`L988`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L988)
  - `is_verify_input(self)` — [`L991`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L991)
  - `merge_batch(self, other)` — [`L1009`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L1009)
  - `prepare_for_verify(self, model_worker_batch: ModelWorkerBatch, page_size: int, target_worker: ModelWorker)` — [`L1055`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L1055)
  - `sample(self, model_worker_batch: ModelWorkerBatch, logits_output: LogitsProcessorOutput, rng: nnx.Rngs, mesh: Mesh)` — [`L1073`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L1073) — Verify and find accepted tokens based on logits output and batch — documented in [python-sgl_jax-srt-speculative-eagle_util](../../../../../concepts/python-sgl_jax-srt-speculative-eagle_util.md)
  - `tree_flatten(self)` — [`L1012`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L1012)
  - `tree_unflatten(cls, aux_data, children)` — [`L1036`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L1036)
  - `capture_hidden_mode` — [`L985`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L985)
  - `custom_mask` — [`L965`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L965)
  - `draft_token` — [`L962`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L962)
  - `draft_token_num` — [`L983`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L983)
  - `positions` — [`L968`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L968)
  - `retrive_cum_len` — [`L975`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L975)
  - `retrive_index` — [`L970`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L970)
  - `retrive_next_sibling` — [`L974`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L974)
  - `retrive_next_token` — [`L972`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L972)
  - `seq_lens_cpu` — [`L977`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L977)
  - `seq_lens_sum` — [`L984`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L984)
  - `spec_steps` — [`L980`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L980)
  - `topk` — [`L981`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L981)
- uses (calls/refs, reference-scoped): [`ForwardMode`](../model_executor/forward_batch_info.md#ForwardMode), `ModelWorkerBatch`, `seq_lens`, `spec_info_padded`, [`CaptureHiddenMode`](../model_executor/forward_batch_info.md#CaptureHiddenMode), `forward_mode`, `sampling_info`, [`LogitsProcessorOutput`](../layers/logits_processor.md#LogitsProcessorOutput), [`next_token_logits`](../layers/logits_processor.md#LogitsProcessorOutput.next_token_logits), `extend_seq_lens`, [`ModelWorker`](../managers/tp_worker.md#ModelWorker), `global_server_args_dict`, [`FULL`](../model_executor/forward_batch_info.md#CaptureHiddenMode.FULL), `capture_hidden_mode`, `input_ids`, `temperatures`, `logits_indices_selector`, `positions`, `filter_batch`, [`top_p_renorm_prob`](../kernels/speculative/kernel.md#top_p_renorm_prob), `top_ps`, `top_ks`, [`_as_int32_array`](eagle_util.md#_as_int32_array), [`top_k_renorm_prob`](../kernels/speculative/kernel.md#top_k_renorm_prob), `is_all_greedy`, [`TARGET_VERIFY`](../model_executor/forward_batch_info.md#ForwardMode.TARGET_VERIFY), [`tree_speculative_sampling_target_only`](../kernels/speculative/kernel.md#tree_speculative_sampling_target_only), [`verify_tree_greedy`](../kernels/speculative/verify_tree_greedy_kernel.md#verify_tree_greedy), [`_generate_simulated_accept_index`](eagle_util.md#_generate_simulated_accept_index), [`SIMULATE_ACC_LEN`](eagle_util.md#SIMULATE_ACC_LEN)
- used by: [`verify`](base_worker.md#BaseSpecWorker.verify), [`draft`](eagle_draft_worker.md#EagleDraftWorker.draft), [`get_eagle_forward_metadata`](../layers/attention/flashattention_backend.md#FlashAttention.get_eagle_forward_metadata), `spec_info_padded`, `spec_info`, [`add_logprob_values`](eagle_worker.md#EAGLEWorker.add_logprob_values), [`_prepare_verify`](draft_extend_fused.md#_prepare_verify), [`spec_info`](../model_executor/forward_batch_info.md#ForwardBatch.spec_info), [`logger`](../model_executor/forward_batch_info.md#logger)

### `EagleVerifyOutput`
- def: [`python/sgl_jax/srt/speculative/eagle_util.py:936`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L936)
- signature: `class EagleVerifyOutput:`
- members:
  - `accept_length_per_req_cpu` — [`L944`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L944)
  - `accepted_indices` — [`L946`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L946)
  - `draft_input` — [`L938`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L938)
  - `logits_output` — [`L940`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L940)
  - `verified_id` — [`L942`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L942)
- uses (calls/refs, reference-scoped): [`EagleDraftInput`](eagle_util.md#EagleDraftInput), [`LogitsProcessorOutput`](../layers/logits_processor.md#LogitsProcessorOutput)
- used by: [`add_logprob_values`](eagle_worker.md#EAGLEWorker.add_logprob_values)

## Functions
- `_as_int32_array(value: Any, *, fallback: int = -1)` — [`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L57) — Convert scalar-like metadata into int32 arrays without forcing device work.
- `_extract_parent_branch_indices(parents_entry: np.ndarray, step_index: int, topk: int)` — [`L224`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L224)
- `_generate_simulated_accept_index(accept_index: jax.Array, predict, accept_length, simulate_acc_len, bs, spec_steps, rng: nnx.Rngs)` — [`L1200`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L1200)
- `_is_jax_leaf(value: Any)` — [`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L51) — Detect sentinel nodes generated by jax.tree_util when shaping pytrees.
- `assign_req_to_token_pool(req_pool_indices, req_to_token_pool, start_offsets, end_offsets, out_cache_loc)` — [`L1251`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L1251)
- `build_chain_verify_inputs(verified_id: np.ndarray, token_list: np.ndarray, seq_lens: np.ndarray, num_verify_tokens: int, batch_size: int)` — [`L296`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L296) — Build verify inputs for topk=1 (linear chain) without tree mask.
- `build_chain_verify_inputs_device(verified_id: jax.Array, token_list: jax.Array, seq_lens: jax.Array, num_verify_tokens: int, batch_size: int)` — [`L342`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L342) — Build verify inputs for topk=1 linear chains on device.
- `build_tree_kernel_efficient(verified_id: jax.Array, score_list: jax.Array, token_list: jax.Array, parents_list: jax.Array, seq_lens: jax.Array, seq_lens_sum: jax.Array, topk: int, num_verify_tokens: int, max_seq_len_per_req: int, batch_size: int, speculative_num_steps: int, mesh: Mesh)` — [`L370`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L370) — JAX implementation of build_tree_kernel_efficient.
- `build_tree_kernel_efficient_preprocess(verified_id: jax.Array, scores: jax.Array, tokens: jax.Array, parents: jax.Array, num_verify_tokens: int, batch_size: int, speculative_num_steps: int)` — [`L184`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L184)
- `build_tree_mask_for_draft_decode(seq_lens: jax.Array | np.ndarray, topk: int, speculative_step_id: int, parents_list: Sequence[jax.Array])` — [`L236`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L236) — Build flattened custom mask for draft decode that respects branch ancestry.
- `get_last_loc_jax_array(req_to_token: jax.Array, req_pool_indices: jax.Array, prefix_lens: jax.Array)` — [`L79`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L79) — JAX version of get_last_loc that operates on JAX arrays.
- `get_last_loc_large_page_size_large_top_k(req_to_token: jax.Array, req_pool_indices: jax.Array, seq_lens: jax.Array, speculative_num_steps: int, topk: int, page_size: int)` — [`L134`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L134) — JAX implementation of get_last_loc_large_page_size_large_top_k.
- `get_last_loc_large_page_size_top_k_1(req_to_token: jax.Array, req_pool_indices: jax.Array, seq_lens: jax.Array, speculative_num_steps: int)` — [`L101`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L101) — JAX implementation of get_last_loc_large_page_size_top_k_1.

## Module values
- `SIMULATE_ACC_LEN` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L47)
- `SIMULATE_ACC_METHOD` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_util.py#L48)

