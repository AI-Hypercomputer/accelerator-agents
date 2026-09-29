---
title: 'Module: python/sgl_jax/srt/speculative/eagle_draft_worker.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/speculative/eagle_draft_worker.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.speculative.eagle_draft_worker`/
symbols:
  EagleDraftWorker.padding_for_decode: EagleDraftWorker#padding_for_decode().
  EagleDraftWorker.draft_forward: EagleDraftWorker#draft_forward().
  EagleDraftWorker.draft_extend_for_prefill: EagleDraftWorker#draft_extend_for_prefill().
  EagleDraftWorker.draft: EagleDraftWorker#draft().
  EagleDraftWorker.draft_extend_for_decode: EagleDraftWorker#draft_extend_for_decode().
  EagleDraftWorker._worker: EagleDraftWorker#_worker.
  EagleDraftWorker.draft_model_runner: EagleDraftWorker#draft_model_runner().
  EagleDraftWorker.capture_for_decode: EagleDraftWorker#capture_for_decode().
  EagleDraftWorker._share_embed_head: EagleDraftWorker#_share_embed_head().
  EagleDraftWorker.topk: EagleDraftWorker#topk.
  EagleDraftWorker.mesh: EagleDraftWorker#mesh().
  update_forward_batch_info: update_forward_batch_info().
  EagleDraftWorker.hot_token_ids: EagleDraftWorker#hot_token_ids.
  EagleDraftWorker: EagleDraftWorker#
  EagleDraftWorker.speculative_num_steps: EagleDraftWorker#speculative_num_steps.
  select_top_k_tokens: select_top_k_tokens().
  EagleDraftWorker.speculative_algorithm: EagleDraftWorker#speculative_algorithm.
  EagleDraftWorker.speculative_num_draft_tokens: EagleDraftWorker#speculative_num_draft_tokens.
  EagleDraftWorker._dp_segment_pad: EagleDraftWorker#_dp_segment_pad().
  EagleDraftWorker.model_config: EagleDraftWorker#model_config().
  EagleDraftWorker.compilation_manager: EagleDraftWorker#compilation_manager().
  EagleDraftWorker.max_req_len: EagleDraftWorker#max_req_len().
  EagleDraftWorker.get_max_padded_size: EagleDraftWorker#get_max_padded_size().
  topk_probs_from_logits: topk_probs_from_logits().
  EagleDraftWorker._map_hot_token_ids: EagleDraftWorker#_map_hot_token_ids().
  EagleDraftWorker._pick_context_len: EagleDraftWorker#_pick_context_len().
  EagleDraftWorker.copy_model_worker_batch_to_cpu: EagleDraftWorker#copy_model_worker_batch_to_cpu().
  EagleDraftWorker.get_padding_bs: EagleDraftWorker#get_padding_bs().
  select_top_k_tokens_step_greater_0: select_top_k_tokens_step_greater_0().
  EagleDraftWorker.precompile_bs_paddings: EagleDraftWorker#precompile_bs_paddings.
  update_eagle_lists: update_eagle_lists().
  RETURN_ORIGINAL_LOGPROB: RETURN_ORIGINAL_LOGPROB.
  EagleDraftWorker.__init__: EagleDraftWorker#__init__().
  EagleDraftWorker.precompile_token_paddings: EagleDraftWorker#precompile_token_paddings.
  EagleDraftWorker.server_args: EagleDraftWorker#server_args.
  EagleDraftWorker.target_worker_ref: EagleDraftWorker#target_worker_ref.
  EagleDraftWorker.page_size: EagleDraftWorker#page_size.
  EagleDraftWorker.precompile_cache_loc_paddings: EagleDraftWorker#precompile_cache_loc_paddings.
  fast_topk: fast_topk().
  select_top_k_tokens_step_0: select_top_k_tokens_step_0().
  logger: logger.
---
# Module: [`python/sgl_jax/srt/speculative/eagle_draft_worker.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py)

## Classes
### `EagleDraftWorker`  ·  implements/extends BaseDraftWorker
- def: [`python/sgl_jax/srt/speculative/eagle_draft_worker.py:37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L37)
- doc: EAGLE draft model worker.
- signature: `class EagleDraftWorker(BaseDraftWorker):`
- members:
  - `_dp_segment_pad(arr, target_bs)` — [`L433`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L433) — DP-segmented pad: pad each rank's section separately to per_dp_padded.
  - `capture_for_decode(self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput, sel=None)` — [`L332`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L332)
  - `compilation_manager(self)` — [`L126`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L126)
  - `copy_model_worker_batch_to_cpu(self, model_worker_batch: ModelWorkerBatch)` — [`L539`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L539)
  - `draft(self, model_worker_batch: ModelWorkerBatch)` — [`L138`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L138)
  - `draft_extend_for_decode(self, model_worker_batch: ModelWorkerBatch, batch_output: GenerationBatchResult)` — [`L275`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L275)
  - `draft_extend_for_prefill(self, model_worker_batch: ModelWorkerBatch, hidden_states: jax.Array, next_token_ids: jax.Array)` — [`L219`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L219)
  - `draft_forward(self, model_worker_batch: ModelWorkerBatch)` — [`L468`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L468) — documented in [python-sgl_jax-srt-model_executor-forward_batch_info](../../../../../concepts/python-sgl_jax-srt-model_executor-forward_batch_info.md)
  - `draft_model_runner(self)` — [`L114`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L114)
  - `get_max_padded_size(self)` — [`L133`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L133)
  - `get_padding_bs(self, real_bs: int)` — [`L568`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L568)
  - `max_req_len(self)` — [`L130`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L130)
  - `mesh(self)` — [`L118`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L118)
  - `model_config(self)` — [`L122`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L122)
  - `padding_for_decode(self, model_worker_batch: ModelWorkerBatch)` — [`L348`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L348)
  - `hot_token_ids` — [`L55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L55)
  - `page_size` — [`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L51)
  - `precompile_bs_paddings` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L81)
  - `precompile_cache_loc_paddings` — [`L82`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L82)
  - `precompile_token_paddings` — [`L83`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L83)
  - `server_args` — [`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L46)
  - `speculative_algorithm` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L52)
  - `speculative_num_draft_tokens` — [`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L50)
  - `speculative_num_steps` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L49)
  - `target_worker_ref` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L47)
  - `topk` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L48)
- protocol/private: `__init__`[`L45`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L45), `_map_hot_token_ids`[`L529`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L529), `_pick_context_len`[`L533`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L533), `_share_embed_head`[`L85`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L85), `_worker`[`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L61)
- uses (calls/refs, reference-scoped): [`prepare_for_extend_after_verify`](eagle_util.md#EagleDraftInput.prepare_for_extend_after_verify), [`ForwardMode`](../model_executor/forward_batch_info.md#ForwardMode), [`draft_extend_for_prefill`](multi_layer_draft_worker.md#MultiLayerDraftWorker.draft_extend_for_prefill), [`init_new`](../model_executor/forward_batch_info.md#ForwardBatch.init_new), `ModelWorkerBatch`, [`model_runner`](../managers/tp_worker.md#ModelWorker.model_runner), [`model`](../model_executor/model_runner.md#ModelRunner.model), `seq_lens`, `spec_info_padded`, [`allocate_lens`](eagle_util.md#EagleDraftInput.allocate_lens), [`CaptureHiddenMode`](../model_executor/forward_batch_info.md#CaptureHiddenMode), [`ForwardBatch`](../model_executor/forward_batch_info.md#ForwardBatch), `dp_size`, [`compilation_manager`](../managers/tp_worker.md#ModelWorker.compilation_manager), [`device_array`](../utils/jax_utils.md#device_array), [`from_model_worker_batch`](../layers/logits_processor.md#LogitsMetadata.from_model_worker_batch), [`EagleDraftInput`](eagle_util.md#EagleDraftInput), [`SpeculativeAlgorithm`](spec_info.md#SpeculativeAlgorithm), `per_dp_bs_size`, [`hidden_states`](eagle_util.md#EagleDraftInput.hidden_states), [`verified_id`](eagle_util.md#EagleDraftInput.verified_id), [`LogitsProcessorOutput`](../layers/logits_processor.md#LogitsProcessorOutput), [`next_draft_input`](../managers/scheduler.md#GenerationBatchResult.next_draft_input), [`EXTEND`](../model_executor/forward_batch_info.md#ForwardMode.EXTEND), [`forward_mode`](../model_executor/forward_batch_info.md#ForwardBatch.forward_mode), [`model_config`](../managers/tp_worker.md#ModelWorker.model_config), [`forward`](../model_executor/model_runner.md#ModelRunner.forward), [`mesh`](../model_executor/model_runner.md#ModelRunner.mesh), [`next_token_logits`](../layers/logits_processor.md#LogitsProcessorOutput.next_token_logits), [`draft_forward`](multi_layer_draft_worker.md#MultiLayerDraftWorker.draft_forward), [`req_to_token`](../mem_cache/memory_pool.md#ReqToTokenPool.req_to_token), [`ModelWorker`](../managers/tp_worker.md#ModelWorker), [`hidden_states`](../layers/logits_processor.md#LogitsProcessorOutput.hidden_states), [`topk_index`](eagle_util.md#EagleDraftInput.topk_index), [`GenerationBatchResult`](../managers/scheduler.md#GenerationBatchResult), [`from_string`](spec_info.md#SpeculativeAlgorithm.from_string), [`topk_p`](eagle_util.md#EagleDraftInput.topk_p), [`attn_backend`](../model_executor/model_runner.md#ModelRunner.attn_backend), [`prepare_for_extend_after_target_prefill`](eagle_util.md#EagleDraftInput.prepare_for_extend_after_target_prefill), [`LogitsMetadata`](../layers/logits_processor.md#LogitsMetadata)  (+61 more)
- used by: [`draft_extend_for_prefill`](multi_layer_draft_worker.md#MultiLayerDraftWorker.draft_extend_for_prefill), [`__init__`](eagle_worker.md#EAGLEWorker.__init__), [`BaseDraftWorker`](base_worker.md#BaseDraftWorker), [`MultiLayerDraftWorker`](multi_layer_draft_worker.md#MultiLayerDraftWorker), [`draft`](base_worker.md#BaseDraftWorker.draft), [`draft_extend_for_decode`](base_worker.md#BaseDraftWorker.draft_extend_for_decode), [`draft_extend_for_prefill`](base_worker.md#BaseDraftWorker.draft_extend_for_prefill), [`draft_model_runner`](base_worker.md#BaseDraftWorker.draft_model_runner)

## Functions
- `fast_topk(values, topk, axis=-1)` — [`L609`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L609)
- `select_top_k_tokens(i: int, topk_p: jax.Array, topk_index: jax.Array, hidden_states: jax.Array, scores: jax.Array, topk: int)` — [`L673`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L673)
- `select_top_k_tokens_step_0(topk_p: jax.Array, topk_index: jax.Array, hidden_states: jax.Array, scores: jax.Array, topk: int)` — [`L690`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L690)
- `select_top_k_tokens_step_greater_0(i: jax.Array, topk_p: jax.Array, topk_index: jax.Array, hidden_states: jax.Array, scores: jax.Array, topk: int)` — [`L712`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L712)
- `topk_probs_from_logits(logits: jax.Array, topk: int, axis: int = -1)` — [`L588`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L588) — Return top-k probabilities without materializing the full softmax tensor.
- `update_eagle_lists(i: int, score_list: jax.Array, token_list: jax.Array, parents_list: jax.Array, tree_info: tuple[jax.Array, jax.Array, jax.Array], topk: int)` — [`L621`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L621)
- `update_forward_batch_info(forward_batch: ForwardBatch, i: int, input_ids: jax.Array, hidden_states: jax.Array, positions_base: jax.Array)` — [`L659`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L659)

## Module values
- `RETURN_ORIGINAL_LOGPROB` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L34)
- `logger` — [`L33`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/eagle_draft_worker.py#L33)

