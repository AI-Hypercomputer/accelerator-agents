---
title: 'Module: python/sgl_jax/srt/speculative/spec_info.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/speculative/spec_info.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.speculative.spec_info`/
symbols:
  SpeculativeAlgorithm: SpeculativeAlgorithm#
  SpeculativeAlgorithm.is_none: SpeculativeAlgorithm#is_none().
  SpeculativeAlgorithm.from_string: SpeculativeAlgorithm#from_string().
  SpeculativeAlgorithm.is_eagle: SpeculativeAlgorithm#is_eagle().
  detect_nan: detect_nan().
  SpeculativeAlgorithm.is_eagle3: SpeculativeAlgorithm#is_eagle3().
  SpeculativeAlgorithm.is_nextn: SpeculativeAlgorithm#is_nextn().
  SpeculativeAlgorithm.is_standalone: SpeculativeAlgorithm#is_standalone().
  SpeculativeAlgorithm.NONE: SpeculativeAlgorithm#NONE.
  SpeculativeAlgorithm.EAGLE3: SpeculativeAlgorithm#EAGLE3.
  SpeculativeAlgorithm.NEXTN: SpeculativeAlgorithm#NEXTN.
  SpecInput.merge_batch: SpecInput#merge_batch().
  SpeculativeAlgorithm.EAGLE: SpeculativeAlgorithm#EAGLE.
  SpeculativeAlgorithm.STANDALONE: SpeculativeAlgorithm#STANDALONE.
  logger: logger.
  SpecInput: SpecInput#
  SpecInput.is_draft_input: SpecInput#is_draft_input().
  SpecInput.is_verify_input: SpecInput#is_verify_input().
  SpecInput.get_spec_adjust_token_coefficient: SpecInput#get_spec_adjust_token_coefficient().
  SpecInput.get_logical_token_num: SpecInput#get_logical_token_num().
  SpecInput.get_allocated_token_num: SpecInput#get_allocated_token_num().
  SpecInput.get_verify_token_num: SpecInput#get_verify_token_num().
  SpecInput.filter_batch: SpecInput#filter_batch().
---
# Module: [`python/sgl_jax/srt/speculative/spec_info.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py)

## Classes
### `SpecInput`  ·  implements/extends Protocol
- def: [`python/sgl_jax/srt/speculative/spec_info.py:16`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L16)
- doc: Common interface for speculative-decode state passed through
- signature: `class SpecInput(Protocol):`
- members:
  - `filter_batch(self, new_indices: np.ndarray, has_been_filtered: bool = True)` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L54)
  - `get_allocated_token_num(self)` — [`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L51)
  - `get_logical_token_num(self, bs: int)` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L47) — Per-request host int32 ``(bs,)``; callers sum for batch totals.
  - `get_spec_adjust_token_coefficient(self)` — [`L43`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L43) — Multiplier for scheduler new-token budgeting (e.g. draft_token_num).
  - `get_verify_token_num(self, bs: int)` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L52)
  - `is_draft_input(self)` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L40)
  - `is_verify_input(self)` — [`L41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L41)
  - `merge_batch(self, other: SpecInput)` — [`L55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L55)

### `SpeculativeAlgorithm`  ·  implements/extends IntEnum
- def: [`python/sgl_jax/srt/speculative/spec_info.py:58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L58)
- signature: `class SpeculativeAlgorithm(IntEnum):`
- members:
  - `from_string(name: str)` — [`L85`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L85)
  - `is_eagle(self)` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L68)
  - `is_eagle3(self)` — [`L75`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L75)
  - `is_nextn(self)` — [`L78`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L78)
  - `is_none(self)` — [`L65`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L65)
  - `is_standalone(self)` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L81)
  - `EAGLE` — [`L60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L60)
  - `EAGLE3` — [`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L61)
  - `NEXTN` — [`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L62)
  - `NONE` — [`L59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L59)
  - `STANDALONE` — [`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L63)
- used by: [`process_batch_result_decode`](../managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_decode), `get_model_worker_batch`, [`process_batch_result_prefill`](../managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_prefill), [`stream_output_generation`](../managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.stream_output_generation), [`prepare_for_extend_after_verify`](eagle_util.md#EagleDraftInput.prepare_for_extend_after_verify), [`_make_dummy_batch`](../model_executor/compilation_manager.md#CompilationManager._make_dummy_batch), `prepare_for_decode`, [`run_batch`](../managers/scheduler.md#Scheduler.run_batch), [`get_eagle_multi_step_metadata`](../layers/attention/flashattention_backend.md#FlashAttention.get_eagle_multi_step_metadata), [`build_kv_cache`](../mem_cache/kv_cache_builder.md#build_kv_cache), [`_request_dispatcher`](../managers/scheduler.md#Scheduler._request_dispatcher), [`log_decode_stats`](../managers/scheduler_metrics_mixin.md#SchedulerMetricsMixin.log_decode_stats), `init_new`, [`spec_algorithm`](../managers/scheduler.md#Scheduler.spec_algorithm), [`init_memory_pool`](../model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin.init_memory_pool), [`_resolve_max_num_reqs`](../model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._resolve_max_num_reqs), [`_spec_multi_layer`](../managers/scheduler.md#Scheduler._spec_multi_layer), [`_share_embed_head`](eagle_draft_worker.md#EagleDraftWorker._share_embed_head), `spec_algorithm`, [`_can_use_fused_spec_decode`](base_worker.md#BaseSpecWorker._can_use_fused_spec_decode), [`_current_sampling_info_owner`](../managers/scheduler.md#Scheduler._current_sampling_info_owner), [`dp_size`](../managers/tp_worker.md#ModelWorker.dp_size), `spec_algorithm`, [`spec_algorithm`](../model_executor/model_runner.md#ModelRunner.spec_algorithm), [`mesh`](base_worker.md#BaseSpecWorker.mesh), [`_is_spec_decode_enabled`](../managers/scheduler.md#Scheduler._is_spec_decode_enabled), [`speculative_algorithm`](base_worker.md#BaseSpecWorker.speculative_algorithm), [`speculative_algorithm`](../managers/tp_worker.md#ModelWorker.speculative_algorithm), [`speculative_algorithm`](eagle_draft_worker.md#EagleDraftWorker.speculative_algorithm), [`spec_algorithm`](../model_executor/forward_batch_info.md#ForwardBatch.spec_algorithm), [`speculative_algorithm`](multi_layer_draft_worker.md#MultiLayerDraftWorker.speculative_algorithm), [`page_size`](multi_layer_draft_worker.md#MultiLayerDraftWorker.page_size)

## Functions
- `detect_nan(logits_output: LogitsProcessorOutput)` — [`L98`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L98)

## Module values
- `logger` — [`L12`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/spec_info.py#L12)

