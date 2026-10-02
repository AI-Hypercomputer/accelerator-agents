---
title: 'Module: python/sgl_jax/srt/speculative/base_worker.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/speculative/base_worker.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.speculative.base_worker`/
symbols:
  BaseSpecWorker.forward_batch_speculative_generation: BaseSpecWorker#forward_batch_speculative_generation().
  BaseSpecWorker.verify: BaseSpecWorker#verify().
  BaseSpecWorker.forward_batch_speculative_decode_overlap: BaseSpecWorker#forward_batch_speculative_decode_overlap().
  BaseSpecWorker.init_spec_relay_buffers: BaseSpecWorker#init_spec_relay_buffers().
  BaseSpecWorker.forward_batch_speculative_prefill_overlap: BaseSpecWorker#forward_batch_speculative_prefill_overlap().
  BaseSpecWorker.forward_target_extend: BaseSpecWorker#forward_target_extend().
  BaseSpecWorker._can_use_fused_spec_decode: BaseSpecWorker#_can_use_fused_spec_decode.
  BaseSpecWorker.draft_worker: BaseSpecWorker#draft_worker().
  BaseSpecWorker.target_worker: BaseSpecWorker#target_worker().
  BaseSpecWorker._can_use_fused_spec_prefill: BaseSpecWorker#_can_use_fused_spec_prefill().
  BaseSpecWorker._prepare_overlap_sampling_info: BaseSpecWorker#_prepare_overlap_sampling_info().
  BaseSpecWorker.mesh: BaseSpecWorker#mesh.
  replicate_to_mesh: replicate_to_mesh().
  BaseSpecWorker.speculative_num_draft_tokens: BaseSpecWorker#speculative_num_draft_tokens.
  BaseSpecWorker.server_args: BaseSpecWorker#server_args.
  BaseSpecWorker.speculative_algorithm: BaseSpecWorker#speculative_algorithm.
  BaseSpecWorker.spec_relay_buffers: BaseSpecWorker#spec_relay_buffers.
  BaseSpecWorker.speculative_num_steps: BaseSpecWorker#speculative_num_steps.
  BaseDraftWorker: BaseDraftWorker#
  BaseSpecWorker.__init__: BaseSpecWorker#__init__().
  BaseSpecWorker.topk: BaseSpecWorker#topk.
  BaseSpecWorker: BaseSpecWorker#
  BaseSpecWorker.page_size: BaseSpecWorker#page_size.
  BaseDraftWorker.draft_model_runner: BaseDraftWorker#draft_model_runner().
  BaseDraftWorker.draft: BaseDraftWorker#draft().
  BaseDraftWorker.draft_extend_for_prefill: BaseDraftWorker#draft_extend_for_prefill().
  BaseDraftWorker.draft_extend_for_decode: BaseDraftWorker#draft_extend_for_decode().
  BaseSpecWorker.precompile_cache_loc_paddings: BaseSpecWorker#precompile_cache_loc_paddings.
  BaseSpecWorker.precompile_bs_paddings: BaseSpecWorker#precompile_bs_paddings.
  BaseSpecWorker.token_to_kv_pool_allocator: BaseSpecWorker#token_to_kv_pool_allocator.
  BaseSpecWorker.cur_sampling_info: BaseSpecWorker#cur_sampling_info.
  BaseSpecWorker.precompile_token_paddings: BaseSpecWorker#precompile_token_paddings.
  BaseSpecWorker._target_worker: BaseSpecWorker#_target_worker.
  BaseSpecWorker._draft_worker: BaseSpecWorker#_draft_worker.
  BaseSpecWorker.req_to_token_pool: BaseSpecWorker#req_to_token_pool.
---
# Module: [`python/sgl_jax/srt/speculative/base_worker.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py)

## Classes
### `BaseDraftWorker`  ·  implements/extends ABC
- def: [`python/sgl_jax/srt/speculative/base_worker.py:34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L34)
- doc: Draft model worker interface for speculative decoding.
- signature: `class BaseDraftWorker(ABC):`
- members:
  - `draft(self, model_worker_batch)` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L49)
  - `draft_extend_for_decode(self, model_worker_batch, batch_output)` — [`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L57)
  - `draft_extend_for_prefill(self, model_worker_batch, hidden_states, next_token_ids)` — [`L53`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L53)
  - `draft_model_runner(self)` — [`L45`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L45) — Primary model runner (multi-runner workers return a designated one).
- uses (calls/refs, reference-scoped): [`draft_extend_for_prefill`](eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_prefill), [`draft`](eagle_draft_worker.md#EagleDraftWorker.draft), [`draft_extend_for_decode`](eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_decode), [`draft_model_runner`](eagle_draft_worker.md#EagleDraftWorker.draft_model_runner), [`EagleDraftWorker`](eagle_draft_worker.md#EagleDraftWorker)
- used by: [`forward_batch_speculative_generation`](base_worker.md#BaseSpecWorker.forward_batch_speculative_generation), [`verify`](base_worker.md#BaseSpecWorker.verify), [`draft_worker`](base_worker.md#BaseSpecWorker.draft_worker), [`EagleDraftWorker`](eagle_draft_worker.md#EagleDraftWorker), [`__init__`](base_worker.md#BaseSpecWorker.__init__)

### `BaseSpecWorker`
- def: [`python/sgl_jax/srt/speculative/base_worker.py:61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L61)
- doc: Speculative decode orchestrator.
- signature: `class BaseSpecWorker:`
- members:
  - `draft_worker(self)` — [`L108`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L108)
  - `forward_batch_speculative_decode_overlap(self, model_worker_batch: ModelWorkerBatch)` — [`L153`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L153)
  - `forward_batch_speculative_generation(self, model_worker_batch: ModelWorkerBatch, launch_done=None)` — [`L197`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L197) — documented in [python-sgl_jax-srt-speculative-eagle_util](../../../../../concepts/python-sgl_jax-srt-speculative-eagle_util.md)
  - `forward_batch_speculative_prefill_overlap(self, model_worker_batch: ModelWorkerBatch)` — [`L175`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L175)
  - `forward_target_extend(self, model_worker_batch: ModelWorkerBatch, sampling_metadata, *, skip_sample: bool = False)` — [`L283`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L283)
  - `init_spec_relay_buffers(self)` — [`L111`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L111)
  - `target_worker(self)` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L104)
  - `verify(self, model_worker_batch: ModelWorkerBatch, cur_allocate_lens: jax.Array)` — [`L307`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L307) — documented in [python-sgl_jax-srt-speculative-eagle_util](../../../../../concepts/python-sgl_jax-srt-speculative-eagle_util.md)
  - `cur_sampling_info` — [`L147`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L147)
  - `mesh` — [`L80`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L80)
  - `page_size` — [`L79`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L79)
  - `precompile_bs_paddings` — [`L98`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L98)
  - `precompile_cache_loc_paddings` — [`L99`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L99)
  - `precompile_token_paddings` — [`L97`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L97)
  - `req_to_token_pool` — [`L94`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L94)
  - `server_args` — [`L72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L72)
  - `spec_relay_buffers` — [`L101`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L101)
  - `speculative_algorithm` — [`L84`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L84)
  - `speculative_num_draft_tokens` — [`L78`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L78)
  - `speculative_num_steps` — [`L77`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L77)
  - `token_to_kv_pool_allocator` — [`L94`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L94)
  - `topk` — [`L76`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L76)
- protocol/private: `__init__`[`L71`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L71), `_can_use_fused_spec_decode`[`L87`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L87), `_can_use_fused_spec_prefill`[`L126`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L126), `_draft_worker`[`L74`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L74), `_prepare_overlap_sampling_info`[`L144`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L144), `_target_worker`[`L73`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L73)
- uses (calls/refs, reference-scoped): [`forward_batch_generation`](../managers/tp_worker.md#ModelWorker.forward_batch_generation), [`spec_prefill`](draft_extend_fused.md#spec_prefill), `ModelWorkerBatch`, [`model_runner`](../managers/tp_worker.md#ModelWorker.model_runner), `seq_lens`, [`from_model_worker_batch`](../sampling/sampling_batch_info.md#SamplingMetadata.from_model_worker_batch), `spec_info_padded`, [`allocate_lens`](eagle_util.md#EagleDraftInput.allocate_lens), [`CaptureHiddenMode`](../model_executor/forward_batch_info.md#CaptureHiddenMode), [`sample`](eagle_util.md#EagleVerifyInput.sample), `forward_mode`, `dp_size`, [`is_extend`](../model_executor/forward_batch_info.md#ForwardMode.is_extend), `sampling_info`, [`EagleDraftInput`](eagle_util.md#EagleDraftInput), [`prepare_for_verify`](eagle_util.md#EagleVerifyInput.prepare_for_verify), [`SpeculativeAlgorithm`](spec_info.md#SpeculativeAlgorithm), [`hidden_states`](eagle_util.md#EagleDraftInput.hidden_states), [`verified_id`](eagle_util.md#EagleDraftInput.verified_id), [`next_draft_input`](../managers/scheduler.md#GenerationBatchResult.next_draft_input), [`model_config`](../managers/tp_worker.md#ModelWorker.model_config), [`next_token_logits`](../layers/logits_processor.md#LogitsProcessorOutput.next_token_logits), [`new_seq_lens`](eagle_util.md#EagleDraftInput.new_seq_lens), [`ModelWorker`](../managers/tp_worker.md#ModelWorker), [`hidden_states`](../layers/logits_processor.md#LogitsProcessorOutput.hidden_states), [`is_decode`](../model_executor/forward_batch_info.md#ForwardMode.is_decode), [`spec_decode_overlap`](draft_extend_fused.md#spec_decode_overlap), [`GenerationBatchResult`](../managers/scheduler.md#GenerationBatchResult), [`from_string`](spec_info.md#SpeculativeAlgorithm.from_string), [`attn_backend`](../model_executor/model_runner.md#ModelRunner.attn_backend), [`create_spec_relay_buffers`](relay_buffer.md#create_spec_relay_buffers), [`SamplingMetadata`](../sampling/sampling_batch_info.md#SamplingMetadata), [`FULL`](../model_executor/forward_batch_info.md#CaptureHiddenMode.FULL), [`use_legacy_eagle3_non_overlap`](overlap_utils.md#use_legacy_eagle3_non_overlap), `capture_hidden_mode`, `temperatures`, `logits_indices_selector`, [`prepare_forward_batch_for_prefill`](draft_extend_fused.md#prepare_forward_batch_for_prefill), `real_bs`, [`EagleVerifyInput`](eagle_util.md#EagleVerifyInput)  (+26 more)
- used by: [`process_batch_result_decode`](../managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_decode), [`result_queue`](../managers/scheduler.md#Scheduler.result_queue), [`_run_speculative_batch`](../managers/scheduler.md#Scheduler._run_speculative_batch), [`precompile_spec_decode`](eagle_worker.md#EAGLEWorker.precompile_spec_decode), [`precompile_spec_extend`](eagle_worker.md#EAGLEWorker.precompile_spec_extend), [`draft_worker`](../managers/scheduler.md#Scheduler.draft_worker), [`_make_decode_batch`](eagle_worker.md#EAGLEWorker._make_decode_batch), [`run_spec_decode_precompile`](eagle_worker.md#EAGLEWorker.run_spec_decode_precompile), [`__init__`](eagle_worker.md#EAGLEWorker.__init__), [`EAGLEWorker`](eagle_worker.md#EAGLEWorker)

## Functions
- `replicate_to_mesh(mesh: jax.sharding.Mesh, *arrs: jax.Array)` — [`L22`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/base_worker.py#L22) — Replicate arrays across a mesh under explicit sharding.

