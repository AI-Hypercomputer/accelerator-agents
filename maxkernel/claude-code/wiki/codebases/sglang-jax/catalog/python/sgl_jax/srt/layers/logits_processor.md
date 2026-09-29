---
title: 'Module: python/sgl_jax/srt/layers/logits_processor.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/logits_processor.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.logits_processor`/Logits
symbols:
  LogitsProcessor.__call__: Processor#__call__().
  LogitsMetadata.tree_flatten: Metadata#tree_flatten().
  LogitsMetadata.tree_unflatten: Metadata#tree_unflatten().
  LogitsMetadata.from_model_worker_batch: Metadata#from_model_worker_batch().
  LogitsProcessorOutput: ProcessorOutput#
  LogitsProcessorOutput.tree_flatten: ProcessorOutput#tree_flatten().
  LogitsProcessorOutput.tree_unflatten: ProcessorOutput#tree_unflatten().
  LogitsProcessorOutput.next_token_logits: ProcessorOutput#next_token_logits.
  LogitsProcessorOutput.hidden_states: ProcessorOutput#hidden_states.
  LogitsMetadata: Metadata#
  LogitsProcessor.compute_temp_top_p_normalized_logprobs: Processor#compute_temp_top_p_normalized_logprobs().
  LogitsProcessor._get_logits: Processor#_get_logits().
  LogitsProcessorOutput.next_token_logprobs: ProcessorOutput#next_token_logprobs.
  LogitsMetadata.forward_mode: Metadata#forward_mode.
  LogitsProcessorOutput.next_token_top_logprobs_val: ProcessorOutput#next_token_top_logprobs_val.
  LogitsMetadata.capture_hidden_mode: Metadata#capture_hidden_mode.
  LogitsProcessorOutput.next_token_top_logprobs_idx: ProcessorOutput#next_token_top_logprobs_idx.
  LogitsProcessorOutput.next_token_token_ids_logprobs_val: ProcessorOutput#next_token_token_ids_logprobs_val.
  LogitsProcessorOutput.input_token_logprobs: ProcessorOutput#input_token_logprobs.
  LogitsProcessor._select_hidden_states: Processor#_select_hidden_states().
  LogitsProcessorOutput.input_top_logprobs_val: ProcessorOutput#input_top_logprobs_val.
  LogitsProcessorOutput.input_token_ids_logprobs_val: ProcessorOutput#input_token_ids_logprobs_val.
  LogitsProcessorOutput.next_token_token_ids_logprobs_idx: ProcessorOutput#next_token_token_ids_logprobs_idx.
  LogitsProcessorOutput.input_top_logprobs_idx: ProcessorOutput#input_top_logprobs_idx.
  LogitsProcessor.mesh: Processor#mesh.
  LogitsProcessor._select_logits: Processor#_select_logits().
  LogitsProcessorOutput.input_token_ids_logprobs_idx: ProcessorOutput#input_token_ids_logprobs_idx.
  LogitsMetadata.extend_return_logprob: Metadata#extend_return_logprob.
  LogitsMetadata.accept_lens: Metadata#accept_lens.
  LogitsMetadata.logits_indices: Metadata#logits_indices.
  LogitsMetadata.top_p: Metadata#top_p.
  LogitsMetadata.extend_return_top_logprob: Metadata#extend_return_top_logprob.
  LogitsMetadata.extend_token_ids_logprob: Metadata#extend_token_ids_logprob.
  LogitsMetadata.extend_seq_lens: Metadata#extend_seq_lens.
  LogitsMetadata.extend_seq_lens_cpu: Metadata#extend_seq_lens_cpu.
  LogitsMetadata.extend_logprob_start_lens_cpu: Metadata#extend_logprob_start_lens_cpu.
  LogitsMetadata.extend_logprob_pruned_lens_cpu: Metadata#extend_logprob_pruned_lens_cpu.
  LogitsMetadata.top_logprobs_nums: Metadata#top_logprobs_nums.
  LogitsMetadata.extend_input_logprob_token_ids_device: Metadata#extend_input_logprob_token_ids_device.
  LogitsMetadata.input_logprob_indices_device: Metadata#input_logprob_indices_device.
  LogitsMetadata.temp_scaled_logprobs: Metadata#temp_scaled_logprobs.
  LogitsMetadata.temperature: Metadata#temperature.
  LogitsMetadata.top_p_normalized_logprobs: Metadata#top_p_normalized_logprobs.
  LogitsMetadata.token_ids_logprobs: Metadata#token_ids_logprobs.
  LogitsProcessor._select_input_token_logprobs: Processor#_select_input_token_logprobs().
  LogitsProcessor.soft_cap: Processor#soft_cap.
  LogitsProcessor.vocab_size: Processor#vocab_size.
  LogitsProcessor.select_local_fn: Processor#select_local_fn().
  LogitsProcessor: Processor#
  LogitsProcessor.__init__: Processor#__init__().
---
# Module: [`python/sgl_jax/srt/layers/logits_processor.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py)

## Classes
### `LogitsMetadata`
- def: [`python/sgl_jax/srt/layers/logits_processor.py:97`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L97)
- signature: `class LogitsMetadata:`
- members:
  - `from_model_worker_batch(cls, batch: ModelWorkerBatch, mesh: Mesh = None)` — [`L184`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L184) — documented in [python-sgl_jax-srt-layers-logits_processor](../../../../../concepts/python-sgl_jax-srt-layers-logits_processor.md)
  - `tree_flatten(self)` — [`L121`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L121) — documented in [python-sgl_jax-srt-layers-logits_processor](../../../../../concepts/python-sgl_jax-srt-layers-logits_processor.md)
  - `tree_unflatten(cls, aux_data, children)` — [`L157`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L157)
  - `accept_lens` — [`L105`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L105)
  - `capture_hidden_mode` — [`L99`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L99)
  - `extend_input_logprob_token_ids_device` — [`L111`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L111)
  - `extend_logprob_pruned_lens_cpu` — [`L109`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L109)
  - `extend_logprob_start_lens_cpu` — [`L108`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L108)
  - `extend_return_logprob` — [`L101`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L101)
  - `extend_return_top_logprob` — [`L102`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L102)
  - `extend_seq_lens` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L104)
  - `extend_seq_lens_cpu` — [`L107`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L107)
  - `extend_token_ids_logprob` — [`L103`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L103)
  - `forward_mode` — [`L98`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L98)
  - `input_logprob_indices_device` — [`L112`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L112)
  - `logits_indices` — [`L106`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L106)
  - `temp_scaled_logprobs` — [`L116`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L116)
  - `temperature` — [`L117`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L117)
  - `token_ids_logprobs` — [`L113`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L113)
  - `top_logprobs_nums` — [`L110`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L110)
  - `top_p` — [`L119`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L119)
  - `top_p_normalized_logprobs` — [`L118`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L118)
- uses (calls/refs, reference-scoped): [`ForwardMode`](../model_executor/forward_batch_info.md#ForwardMode), `ModelWorkerBatch`, `spec_info_padded`, [`CaptureHiddenMode`](../model_executor/forward_batch_info.md#CaptureHiddenMode), `forward_mode`, [`is_extend`](../model_executor/forward_batch_info.md#ForwardMode.is_extend), [`device_array`](../utils/jax_utils.md#device_array), `extend_seq_lens`, `capture_hidden_mode`, `logits_indices`, `token_ids_logprobs`, `top_logprobs_nums`, [`accept_length`](../speculative/eagle_util.md#EagleDraftInput.accept_length), `return_logprob`, `extend_logprob_start_lens`, `extend_input_logprob_token_ids`, [`is_draft_extend`](../model_executor/forward_batch_info.md#ForwardMode.is_draft_extend), `input_logprob_indices`
- used by: [`prepare_for_extend_after_verify`](../speculative/eagle_util.md#EagleDraftInput.prepare_for_extend_after_verify), [`__call__`](logits_processor.md#LogitsProcessor.__call__), [`draft_extend_for_prefill`](../speculative/multi_layer_draft_worker.md#MultiLayerDraftWorker.draft_extend_for_prefill), [`forward_batch_generation`](../managers/tp_worker.md#ModelWorker.forward_batch_generation), [`draft_forward`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_forward), [`draft_extend_for_prefill`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_prefill), [`_prepare_logits_metadata`](../speculative/draft_extend_fused.md#_prepare_logits_metadata), [`forward`](../model_executor/model_runner.md#ModelRunner.forward), [`_forward_raw`](../model_executor/model_runner.md#ModelRunner._forward_raw), [`compute_temp_top_p_normalized_logprobs`](logits_processor.md#LogitsProcessor.compute_temp_top_p_normalized_logprobs), [`_forward`](../model_executor/model_runner.md#ModelRunner._forward), [`forward_idle`](../model_executor/model_runner.md#ModelRunner.forward_idle)

### `LogitsProcessor`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/layers/logits_processor.py:249`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L249)
- doc: Logits processor for the model.
- signature: `class LogitsProcessor(nnx.Module):`
- members:
  - `_get_logits(self, hidden_states: jax.Array, lm_head: Embed)` — [`L516`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L516) — Get logits from hidden_states.
  - `compute_temp_top_p_normalized_logprobs(self, last_logits: jax.Array, logits_metadata: LogitsMetadata)` — [`L491`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L491) — compute logprobs for the output token from the given logits.
  - `select_local_fn(local_states, local_indices)` — [`L258`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L258)
  - `mesh` — [`L255`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L255)
  - `soft_cap` — [`L254`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L254)
  - `vocab_size` — [`L253`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L253)
- protocol/private: `__call__`[`L309`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L309), `__init__`[`L252`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L252), `_select_hidden_states`[`L257`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L257), `_select_input_token_logprobs`[`L286`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L286), `_select_logits`[`L268`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L268)
- uses (calls/refs, reference-scoped): [`is_extend`](../model_executor/forward_batch_info.md#ForwardMode.is_extend), [`device_array`](../utils/jax_utils.md#device_array), [`named_scope`](../utils/profiling_utils.md#named_scope), [`LogitsProcessorOutput`](logits_processor.md#LogitsProcessorOutput), [`next_token_logits`](logits_processor.md#LogitsProcessorOutput.next_token_logits), [`hidden_states`](logits_processor.md#LogitsProcessorOutput.hidden_states), [`LogitsMetadata`](logits_processor.md#LogitsMetadata), [`is_decode_or_idle`](../model_executor/forward_batch_info.md#ForwardMode.is_decode_or_idle), [`forward_mode`](logits_processor.md#LogitsMetadata.forward_mode), [`capture_hidden_mode`](logits_processor.md#LogitsMetadata.capture_hidden_mode), [`input_token_logprobs`](logits_processor.md#LogitsProcessorOutput.input_token_logprobs), [`is_target_verify`](../model_executor/forward_batch_info.md#ForwardMode.is_target_verify), [`input_token_ids_logprobs_val`](logits_processor.md#LogitsProcessorOutput.input_token_ids_logprobs_val), [`input_top_logprobs_val`](logits_processor.md#LogitsProcessorOutput.input_top_logprobs_val), [`Embed`](embeddings.md#Embed), [`input_top_logprobs_idx`](logits_processor.md#LogitsProcessorOutput.input_top_logprobs_idx), [`is_draft_extend`](../model_executor/forward_batch_info.md#ForwardMode.is_draft_extend), [`accept_lens`](logits_processor.md#LogitsMetadata.accept_lens), [`extend_return_logprob`](logits_processor.md#LogitsMetadata.extend_return_logprob), [`input_token_ids_logprobs_idx`](logits_processor.md#LogitsProcessorOutput.input_token_ids_logprobs_idx), [`is_full`](../model_executor/forward_batch_info.md#CaptureHiddenMode.is_full), [`is_last`](../model_executor/forward_batch_info.md#CaptureHiddenMode.is_last), [`logits_indices`](logits_processor.md#LogitsMetadata.logits_indices), [`need_capture`](../model_executor/forward_batch_info.md#CaptureHiddenMode.need_capture), [`top_p`](logits_processor.md#LogitsMetadata.top_p), [`dtype`](embeddings.md#Embed.dtype), [`embedding`](embeddings.md#Embed.embedding), [`extend_input_logprob_token_ids_device`](logits_processor.md#LogitsMetadata.extend_input_logprob_token_ids_device), [`extend_logprob_pruned_lens_cpu`](logits_processor.md#LogitsMetadata.extend_logprob_pruned_lens_cpu), [`extend_logprob_start_lens_cpu`](logits_processor.md#LogitsMetadata.extend_logprob_start_lens_cpu), [`extend_return_top_logprob`](logits_processor.md#LogitsMetadata.extend_return_top_logprob), [`extend_seq_lens`](logits_processor.md#LogitsMetadata.extend_seq_lens), [`extend_seq_lens_cpu`](logits_processor.md#LogitsMetadata.extend_seq_lens_cpu), [`extend_token_ids_logprob`](logits_processor.md#LogitsMetadata.extend_token_ids_logprob), [`get_top_logprobs`](sampler.md#get_top_logprobs), [`input_logprob_indices_device`](logits_processor.md#LogitsMetadata.input_logprob_indices_device), [`temp_scaled_logprobs`](logits_processor.md#LogitsMetadata.temp_scaled_logprobs), [`temperature`](logits_processor.md#LogitsMetadata.temperature), [`top_logprobs_nums`](logits_processor.md#LogitsMetadata.top_logprobs_nums), [`top_p_normalized_logprobs`](logits_processor.md#LogitsMetadata.top_p_normalized_logprobs)  (+2 more)

### `LogitsProcessorOutput`
- def: [`python/sgl_jax/srt/layers/logits_processor.py:27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L27) — documented in [python-sgl_jax-srt-layers-logits_processor](../../../../../concepts/python-sgl_jax-srt-layers-logits_processor.md)
- signature: `class LogitsProcessorOutput:`
- members:
  - `tree_flatten(self)` — [`L55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L55)
  - `tree_unflatten(cls, aux_data, children)` — [`L75`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L75)
  - `hidden_states` — [`L33`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L33) — documented in [python-sgl_jax-srt-layers-logits_processor](../../../../../concepts/python-sgl_jax-srt-layers-logits_processor.md)
  - `input_token_ids_logprobs_idx` — [`L53`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L53)
  - `input_token_ids_logprobs_val` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L52)
  - `input_token_logprobs` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L47)
  - `input_top_logprobs_idx` — [`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L50)
  - `input_top_logprobs_val` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L49)
  - `next_token_logits` — [`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L30) — documented in [python-sgl_jax-srt-layers-logits_processor](../../../../../concepts/python-sgl_jax-srt-layers-logits_processor.md)
  - `next_token_logprobs` — [`L37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L37)
  - `next_token_token_ids_logprobs_idx` — [`L43`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L43)
  - `next_token_token_ids_logprobs_val` — [`L42`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L42)
  - `next_token_top_logprobs_idx` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L40)
  - `next_token_top_logprobs_val` — [`L39`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/logits_processor.py#L39)
- used by: [`process_batch_result_decode`](../managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_decode), [`process_batch_result_prefill`](../managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_prefill), [`__call__`](logits_processor.md#LogitsProcessor.__call__), [`forward_batch_speculative_generation`](../speculative/base_worker.md#BaseSpecWorker.forward_batch_speculative_generation), [`draft_extend_for_prefill`](../speculative/multi_layer_draft_worker.md#MultiLayerDraftWorker.draft_extend_for_prefill), [`forward_batch_generation`](../managers/tp_worker.md#ModelWorker.forward_batch_generation), [`verify`](../speculative/base_worker.md#BaseSpecWorker.verify), [`draft_forward`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_forward), [`draft_extend_for_prefill`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_prefill), [`spec_decode_verify`](../speculative/draft_extend_fused.md#spec_decode_verify), [`draft_extend_for_decode`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_decode), [`add_input_logprob_return_values`](../managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.add_input_logprob_return_values), [`sample`](../speculative/eagle_util.md#EagleVerifyInput.sample), [`add_logprob_values`](../speculative/eagle_worker.md#EAGLEWorker.add_logprob_values), [`add_logprob_return_values`](../managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.add_logprob_return_values), [`_materialize_logprobs_to_host`](../managers/tp_worker.md#ModelWorker._materialize_logprobs_to_host), [`_process_logprob_results`](sampler.md#Sampler._process_logprob_results), [`__call__`](sampler.md#Sampler.__call__), [`capture_for_decode`](../speculative/eagle_draft_worker.md#EagleDraftWorker.capture_for_decode), [`forward`](../model_executor/model_runner.md#ModelRunner.forward), [`_forward_raw`](../model_executor/model_runner.md#ModelRunner._forward_raw), [`logits_output`](../managers/scheduler.md#GenerationBatchResult.logits_output), [`sample`](../model_executor/model_runner.md#ModelRunner.sample), [`forward_batch_generation`](../managers/tp_worker.md#MockModelWorker.forward_batch_generation), [`forward_idle`](../model_executor/model_runner.md#ModelRunner.forward_idle), [`detect_nan`](../speculative/spec_info.md#detect_nan), [`logits_output`](../speculative/eagle_util.md#EagleVerifyOutput.logits_output)

