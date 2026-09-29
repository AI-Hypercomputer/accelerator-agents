---
title: 'Module: python/sgl_jax/srt/speculative/multi_layer_draft_worker.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/speculative/multi_layer_draft_worker.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.speculative.multi_layer_draft_worker`/
symbols:
  MultiLayerDraftWorker.draft_extend_for_prefill: MultiLayerDraftWorker#draft_extend_for_prefill().
  MultiLayerDraftWorker._worker: MultiLayerDraftWorker#_worker.
  MultiLayerDraftWorker.draft_forward: MultiLayerDraftWorker#draft_forward().
  MultiLayerDraftWorker._workers: MultiLayerDraftWorker#_workers.
  MultiLayerDraftWorker.num_mtp_layers: MultiLayerDraftWorker#num_mtp_layers.
  MultiLayerDraftWorker.topk: MultiLayerDraftWorker#topk.
  MultiLayerDraftWorker._rotate_ids: MultiLayerDraftWorker#_rotate_ids().
  MultiLayerDraftWorker._share_embed_head_one: MultiLayerDraftWorker#_share_embed_head_one().
  MultiLayerDraftWorker.draft_extend_for_decode: MultiLayerDraftWorker#draft_extend_for_decode().
  MultiLayerDraftWorker: MultiLayerDraftWorker#
  MultiLayerDraftWorker.speculative_num_steps: MultiLayerDraftWorker#speculative_num_steps.
  MultiLayerDraftWorker.speculative_algorithm: MultiLayerDraftWorker#speculative_algorithm.
  MultiLayerDraftWorker.draft_model_runner: MultiLayerDraftWorker#draft_model_runner().
  MultiLayerDraftWorker.runner: MultiLayerDraftWorker#runner().
  MultiLayerDraftWorker.__init__: MultiLayerDraftWorker#__init__().
  MultiLayerDraftWorker.page_size: MultiLayerDraftWorker#page_size.
  MultiLayerDraftWorker.precompile_cache_loc_paddings: MultiLayerDraftWorker#precompile_cache_loc_paddings.
  _server_args_with_mtp_layer: _server_args_with_mtp_layer().
  MultiLayerDraftWorker.speculative_num_draft_tokens: MultiLayerDraftWorker#speculative_num_draft_tokens.
  logger: logger.
  MultiLayerDraftWorker.server_args: MultiLayerDraftWorker#server_args.
  MultiLayerDraftWorker.target_worker_ref: MultiLayerDraftWorker#target_worker_ref.
  MultiLayerDraftWorker.hot_token_ids: MultiLayerDraftWorker#hot_token_ids.
  MultiLayerDraftWorker.precompile_token_paddings: MultiLayerDraftWorker#precompile_token_paddings.
  MultiLayerDraftWorker.precompile_bs_paddings: MultiLayerDraftWorker#precompile_bs_paddings.
---
# Module: [`python/sgl_jax/srt/speculative/multi_layer_draft_worker.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py)

## Classes
### `MultiLayerDraftWorker`  ·  implements/extends EagleDraftWorker
- def: [`python/sgl_jax/srt/speculative/multi_layer_draft_worker.py:53`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L53)
- doc: N independent draft model runners, one per MTP layer.
- signature: `class MultiLayerDraftWorker(EagleDraftWorker):`
- members:
  - `_rotate_ids(mwb: ModelWorkerBatch, last_tok: np.ndarray, sel_pos: np.ndarray)` — [`L186`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L186) — In-place left-shift each req's input_ids by 1, then write last_tok[slot]
  - `draft_extend_for_decode(self, model_worker_batch: ModelWorkerBatch, batch_output: GenerationBatchResult)` — [`L298`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L298) — Decode-extend across all MTP layers.
  - `draft_extend_for_prefill(self, model_worker_batch: ModelWorkerBatch, hidden_states: jax.Array, next_token_ids: jax.Array)` — [`L205`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L205) — Prefill-extend across all MTP layers. — documented in [python-sgl_jax-srt-model_executor-forward_batch_info](../../../../../concepts/python-sgl_jax-srt-model_executor-forward_batch_info.md)
  - `draft_forward(self, model_worker_batch: ModelWorkerBatch)` — [`L149`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L149)
  - `draft_model_runner(self)` — [`L141`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L141)
  - `runner(self, step: int)` — [`L144`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L144)
  - `hot_token_ids` — [`L74`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L74)
  - `num_mtp_layers` — [`L76`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L76)
  - `page_size` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L68)
  - `precompile_bs_paddings` — [`L128`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L128)
  - `precompile_cache_loc_paddings` — [`L129`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L129)
  - `precompile_token_paddings` — [`L127`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L127)
  - `server_args` — [`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L63)
  - `speculative_algorithm` — [`L71`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L71)
  - `speculative_num_draft_tokens` — [`L67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L67)
  - `speculative_num_steps` — [`L66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L66)
  - `target_worker_ref` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L64)
  - `topk` — [`L65`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L65)
- protocol/private: `__init__`[`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L62), `_share_embed_head_one`[`L132`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L132), `_worker`[`L101`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L101), `_workers`[`L90`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L90)
- uses (calls/refs, reference-scoped): [`hf_config`](../configs/model_config.md#ModelConfig.hf_config), [`ForwardMode`](../model_executor/forward_batch_info.md#ForwardMode), [`init_new`](../model_executor/forward_batch_info.md#ForwardBatch.init_new), `ModelWorkerBatch`, [`model_runner`](../managers/tp_worker.md#ModelWorker.model_runner), [`model`](../model_executor/model_runner.md#ModelRunner.model), `seq_lens`, `spec_info_padded`, [`allocate_lens`](eagle_util.md#EagleDraftInput.allocate_lens), [`CaptureHiddenMode`](../model_executor/forward_batch_info.md#CaptureHiddenMode), [`ForwardBatch`](../model_executor/forward_batch_info.md#ForwardBatch), `dp_size`, [`from_model_worker_batch`](../layers/logits_processor.md#LogitsMetadata.from_model_worker_batch), [`EagleDraftInput`](eagle_util.md#EagleDraftInput), [`SpeculativeAlgorithm`](spec_info.md#SpeculativeAlgorithm), `per_dp_bs_size`, [`hidden_states`](eagle_util.md#EagleDraftInput.hidden_states), [`verified_id`](eagle_util.md#EagleDraftInput.verified_id), [`EXTEND`](../model_executor/forward_batch_info.md#ForwardMode.EXTEND), [`forward_mode`](../model_executor/forward_batch_info.md#ForwardBatch.forward_mode), [`model_config`](../managers/tp_worker.md#ModelWorker.model_config), [`forward`](../model_executor/model_runner.md#ModelRunner.forward), [`next_token_logits`](../layers/logits_processor.md#LogitsProcessorOutput.next_token_logits), `extend_seq_lens`, [`ModelWorker`](../managers/tp_worker.md#ModelWorker), [`hidden_states`](../layers/logits_processor.md#LogitsProcessorOutput.hidden_states), [`topk_index`](eagle_util.md#EagleDraftInput.topk_index), [`GenerationBatchResult`](../managers/scheduler.md#GenerationBatchResult), [`from_string`](spec_info.md#SpeculativeAlgorithm.from_string), [`topk_p`](eagle_util.md#EagleDraftInput.topk_p), [`attn_backend`](../model_executor/model_runner.md#ModelRunner.attn_backend), [`prepare_for_extend_after_target_prefill`](eagle_util.md#EagleDraftInput.prepare_for_extend_after_target_prefill), [`LogitsMetadata`](../layers/logits_processor.md#LogitsMetadata), [`FULL`](../model_executor/forward_batch_info.md#CaptureHiddenMode.FULL), `capture_hidden_mode`, [`mesh`](eagle_draft_worker.md#EagleDraftWorker.mesh), `input_ids`, `logits_indices_selector`, [`capture_hidden_mode`](eagle_util.md#EagleDraftInput.capture_hidden_mode), [`get_precompile_paddings`](../managers/tp_worker.md#ModelWorker.get_precompile_paddings)  (+16 more)
- used by: [`draft_forward`](eagle_draft_worker.md#EagleDraftWorker.draft_forward), [`draft_extend_for_prefill`](eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_prefill), [`precompile_spec_decode`](eagle_worker.md#EAGLEWorker.precompile_spec_decode), [`draft_extend_for_decode`](eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_decode), [`draft_model_runner`](eagle_draft_worker.md#EagleDraftWorker.draft_model_runner), [`EagleDraftWorker`](eagle_draft_worker.md#EagleDraftWorker), [`__init__`](multi_layer_eagle_worker.md#MultiLayerEAGLEWorker.__init__)

## Functions
- `_server_args_with_mtp_layer(server_args, layer_idx: int)` — [`L45`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L45)

## Module values
- `logger` — [`L42`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/multi_layer_draft_worker.py#L42)

