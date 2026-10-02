---
title: 'Module: python/sgl_jax/srt/speculative/eagle_worker.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/speculative/eagle_worker.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.speculative.eagle_worker`/
symbols:
  EAGLEWorker.precompile_spec_decode: EAGLEWorker#precompile_spec_decode().
  EAGLEWorker.add_logprob_values: EAGLEWorker#add_logprob_values().
  EAGLEWorker.precompile_spec_extend: EAGLEWorker#precompile_spec_extend().
  EAGLEWorker._make_decode_batch: EAGLEWorker#_make_decode_batch().
  EAGLEWorker.run_spec_decode_precompile: EAGLEWorker#run_spec_decode_precompile().
  EAGLEWorker.__init__: EAGLEWorker#__init__().
  logger: logger.
  EAGLEWorker: EAGLEWorker#
  RETURN_ORIGINAL_LOGPROB: RETURN_ORIGINAL_LOGPROB.
---
# Module: [`python/sgl_jax/srt/speculative/eagle_worker.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_worker.py)

## Classes
### `EAGLEWorker`  ·  implements/extends BaseSpecWorker
- def: [`python/sgl_jax/srt/speculative/eagle_worker.py:26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_worker.py#L26)
- doc: Standard EAGLE speculative decode orchestrator.
- signature: `class EAGLEWorker(BaseSpecWorker):`
- members:
  - `add_logprob_values(self, batch: ScheduleBatch, res: EagleVerifyOutput, logits_output: LogitsProcessorOutput)` — [`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_worker.py#L46)
  - `precompile_spec_decode(self)` — [`L172`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_worker.py#L172)
  - `precompile_spec_extend(self)` — [`L123`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_worker.py#L123)
  - `run_spec_decode_precompile(self)` — [`L116`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_worker.py#L116)
- protocol/private: `__init__`[`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_worker.py#L34), `_make_decode_batch`[`L198`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_worker.py#L198)
- uses (calls/refs, reference-scoped): `reqs_info`, [`forward_batch_speculative_generation`](base_worker.md#BaseSpecWorker.forward_batch_speculative_generation), [`ForwardMode`](../model_executor/forward_batch_info.md#ForwardMode), `spec_info`, [`allocate_lens`](eagle_util.md#EagleDraftInput.allocate_lens), [`CaptureHiddenMode`](../model_executor/forward_batch_info.md#CaptureHiddenMode), `ScheduleBatch`, [`EagleDraftInput`](eagle_util.md#EagleDraftInput), [`hidden_states`](eagle_util.md#EagleDraftInput.hidden_states), [`verified_id`](eagle_util.md#EagleDraftInput.verified_id), [`LogitsProcessorOutput`](../layers/logits_processor.md#LogitsProcessorOutput), [`EXTEND`](../model_executor/forward_batch_info.md#ForwardMode.EXTEND), [`forward_batch_speculative_decode_overlap`](base_worker.md#BaseSpecWorker.forward_batch_speculative_decode_overlap), [`next_token_logits`](../layers/logits_processor.md#LogitsProcessorOutput.next_token_logits), [`init_spec_relay_buffers`](base_worker.md#BaseSpecWorker.init_spec_relay_buffers), [`ModelWorker`](../managers/tp_worker.md#ModelWorker), [`forward_batch_speculative_prefill_overlap`](base_worker.md#BaseSpecWorker.forward_batch_speculative_prefill_overlap), [`topk_index`](eagle_util.md#EagleDraftInput.topk_index), [`topk_p`](eagle_util.md#EagleDraftInput.topk_p), [`DECODE`](../model_executor/forward_batch_info.md#ForwardMode.DECODE), [`FULL`](../model_executor/forward_batch_info.md#CaptureHiddenMode.FULL), [`next_token_logprobs`](../layers/logits_processor.md#LogitsProcessorOutput.next_token_logprobs), [`capture_hidden_mode`](eagle_util.md#EagleDraftInput.capture_hidden_mode), [`draft_worker`](base_worker.md#BaseSpecWorker.draft_worker), [`future_indices`](eagle_util.md#EagleDraftInput.future_indices), [`draft_token_num`](eagle_util.md#EagleVerifyInput.draft_token_num), [`_can_use_fused_spec_prefill`](base_worker.md#BaseSpecWorker._can_use_fused_spec_prefill), [`LAST`](../model_executor/forward_batch_info.md#CaptureHiddenMode.LAST), [`mesh`](base_worker.md#BaseSpecWorker.mesh), [`next_token_top_logprobs_val`](../layers/logits_processor.md#LogitsProcessorOutput.next_token_top_logprobs_val), [`EagleDraftWorker`](eagle_draft_worker.md#EagleDraftWorker), [`next_token_token_ids_logprobs_val`](../layers/logits_processor.md#LogitsProcessorOutput.next_token_token_ids_logprobs_val), [`next_token_top_logprobs_idx`](../layers/logits_processor.md#LogitsProcessorOutput.next_token_top_logprobs_idx), [`num_tokens_for_logprob_per_batch`](eagle_util.md#EagleDraftInput.num_tokens_for_logprob_per_batch), [`num_tokens_per_batch`](eagle_util.md#EagleDraftInput.num_tokens_per_batch), [`ALLOC_LEN_PER_DECODE`](eagle_util.md#EagleDraftInput.ALLOC_LEN_PER_DECODE), [`logger`](eagle_worker.md#logger), [`server_args`](base_worker.md#BaseSpecWorker.server_args), [`spec_relay_buffers`](base_worker.md#BaseSpecWorker.spec_relay_buffers), [`speculative_algorithm`](base_worker.md#BaseSpecWorker.speculative_algorithm)  (+20 more)
- used by: [`_request_dispatcher`](../managers/scheduler.md#Scheduler._request_dispatcher), [`_spec_multi_layer`](../managers/scheduler.md#Scheduler._spec_multi_layer), [`__init__`](multi_layer_eagle_worker.md#MultiLayerEAGLEWorker.__init__), [`BaseSpecWorker`](base_worker.md#BaseSpecWorker), [`MultiLayerEAGLEWorker`](multi_layer_eagle_worker.md#MultiLayerEAGLEWorker)

## Module values
- `RETURN_ORIGINAL_LOGPROB` — [`L23`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_worker.py#L23)
- `logger` — [`L22`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_worker.py#L22)

