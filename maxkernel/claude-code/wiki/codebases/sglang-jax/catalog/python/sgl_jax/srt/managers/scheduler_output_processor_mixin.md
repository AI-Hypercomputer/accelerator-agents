---
title: 'Module: python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.managers.scheduler_output_processor_mixin`/
symbols:
  SchedulerOutputProcessorMixin.process_batch_result_decode: SchedulerOutputProcessorMixin#process_batch_result_decode().
  SchedulerOutputProcessorMixin.process_batch_result_prefill: SchedulerOutputProcessorMixin#process_batch_result_prefill().
  SchedulerOutputProcessorMixin.stream_output_generation: SchedulerOutputProcessorMixin#stream_output_generation().
  SchedulerOutputProcessorMixin.add_input_logprob_return_values: SchedulerOutputProcessorMixin#add_input_logprob_return_values().
  SchedulerOutputProcessorMixin.add_logprob_return_values: SchedulerOutputProcessorMixin#add_logprob_return_values().
  SchedulerOutputProcessorMixin.maybe_collect_routed_experts: SchedulerOutputProcessorMixin#maybe_collect_routed_experts().
  _input_logprob_lens_per_dp: _input_logprob_lens_per_dp().
  SchedulerOutputProcessorMixin.stream_output: SchedulerOutputProcessorMixin#stream_output().
  logger: logger.
  SchedulerOutputProcessorMixin: SchedulerOutputProcessorMixin#
  DEFAULT_FORCE_STREAM_INTERVAL: DEFAULT_FORCE_STREAM_INTERVAL.
  _materialize_input_token_logprobs: _materialize_input_token_logprobs().
---
# Module: [`python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py)

## Classes
### `SchedulerOutputProcessorMixin`
- def: [`python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py:75`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py#L75)
- doc: This class implements the output processing logic for Scheduler.
- signature: `class SchedulerOutputProcessorMixin:`
- members:
  - `add_input_logprob_return_values(self: Scheduler, i: int, req: Req, output: LogitsProcessorOutput, logprob_pt: int, num_input_logprobs: int, last_prefill_chunk: bool)` — [`L563`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py#L563) — Incrementally add input logprobs to `req`.
  - `add_logprob_return_values(self: Scheduler, i: int, req: Req, pt: int, next_token_ids: list[int], num_input_logprobs: int, output: LogitsProcessorOutput, local_idx: int | None = None)` — [`L688`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py#L688) — Attach logprobs to the return values.
  - `maybe_collect_routed_experts(self: Scheduler, req: Req)` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py#L81) — Collect routed experts for a finished request.
  - `process_batch_result_decode(self: Scheduler, batch: ScheduleBatch, result: GenerationBatchResult, launch_done: threading.Event | None = None)` — [`L325`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py#L325) — documented in [root](../../../../../concepts/root.md)
  - `process_batch_result_prefill(self: Scheduler, batch: ScheduleBatch, result: GenerationBatchResult, launch_done: threading.Event | None = None)` — [`L94`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py#L94) — documented in [root](../../../../../concepts/root.md)
  - `stream_output(self: Scheduler, reqs: list[Req], return_logprob: bool, return_output_logprob_only: bool, skip_reqs: set | Req | None = None, cache_miss_count: int = None)` — [`L724`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py#L724) — Stream the output to detokenizer.
  - `stream_output_generation(self: Scheduler, reqs: list[Req], return_logprob: bool, return_output_logprob_only: bool, skip_reqs: set | Req | None = None, cache_miss_count: int = None)` — [`L743`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py#L743)
- uses (calls/refs, reference-scoped): `reqs`, `reqs_info`, `Req`, `dp_size`, [`abort_request`](scheduler.md#Scheduler.abort_request), `sampling_params`, [`Scheduler`](scheduler.md#Scheduler), `output_ids`, [`log_decode_stats`](scheduler_metrics_mixin.md#SchedulerMetricsMixin.log_decode_stats), [`tree_cache`](scheduler.md#Scheduler.tree_cache), `spec_info`, [`spec_algorithm`](scheduler.md#Scheduler.spec_algorithm), [`precision_tracer`](../precision_tracer.md#precision_tracer), `origin_input_ids`, `rid`, `ScheduleBatch`, [`enable_overlap`](scheduler.md#Scheduler.enable_overlap), `return_logprob`, [`token_to_kv_pool_allocator`](scheduler.md#Scheduler.token_to_kv_pool_allocator), `req_pool_idx`, [`LogitsProcessorOutput`](../layers/logits_processor.md#LogitsProcessorOutput), `check_finished`, [`next_draft_input`](scheduler.md#GenerationBatchResult.next_draft_input), [`_comm_backend`](scheduler.md#Scheduler._comm_backend), `BaseFinishReason`, [`stop_trace`](../precision_tracer.md#PrecisionTracer.stop_trace), `grammar`, [`draft_worker`](scheduler.md#Scheduler.draft_worker), [`hidden_states`](../layers/logits_processor.md#LogitsProcessorOutput.hidden_states), [`is_none`](../speculative/spec_info.md#SpeculativeAlgorithm.is_none), [`GenerationBatchResult`](scheduler.md#GenerationBatchResult), [`req_to_token_pool`](scheduler.md#Scheduler.req_to_token_pool), `output_token_ids_logprobs_idx`, `return_output_logprob_only`, `finished_reason`, [`is_eagle`](../speculative/spec_info.md#SpeculativeAlgorithm.is_eagle), [`release_kv_cache`](../mem_cache/common.md#release_kv_cache), [`rid`](io_struct.md#BaseReq.rid), [`server_args`](scheduler.md#Scheduler.server_args), [`use_legacy_eagle3_non_overlap`](../speculative/overlap_utils.md#use_legacy_eagle3_non_overlap)  (+106 more)
- used by: [`Scheduler`](scheduler.md#Scheduler), [`process_batch_result`](scheduler.md#Scheduler.process_batch_result), [`_stream_prefill_req`](../disaggregation/prefill.md#SchedulerDisaggregationPrefillMixin._stream_prefill_req)

## Functions
- `_input_logprob_lens_per_dp(batch: ScheduleBatch)` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py#L34)
- `_materialize_input_token_logprobs(input_token_logprobs, lens_per_dp: list[int] | None = None)` — [`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py#L51)

## Module values
- `DEFAULT_FORCE_STREAM_INTERVAL` — [`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py#L31)
- `logger` — [`L29`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py#L29)

