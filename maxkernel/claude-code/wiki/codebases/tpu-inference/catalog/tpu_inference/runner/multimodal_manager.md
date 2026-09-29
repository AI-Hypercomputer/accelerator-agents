---
title: 'Module: tpu_inference/runner/multimodal_manager.py'
type: catalog
provenance: extracted
module: tpu_inference/runner/multimodal_manager.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.runner.multimodal_manager`/
symbols:
  MultiModalManager.calc_mrope_positions: MultiModalManager#calc_mrope_positions().
  MultiModalManager.execute_mm_encoder: MultiModalManager#execute_mm_encoder().
  MultiModalManager.runner: MultiModalManager#runner.
  MultiModalManager.gather_mm_embeddings: MultiModalManager#gather_mm_embeddings().
  logger: logger.
  MultiModalManager: MultiModalManager#
  MultiModalManager.__init__: MultiModalManager#__init__().
---
# Module: [`tpu_inference/runner/multimodal_manager.py`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/multimodal_manager.py)

## Classes
### `MultiModalManager`
- def: [`tpu_inference/runner/multimodal_manager.py:35`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/multimodal_manager.py#L35)
- signature: `class MultiModalManager:`
- members:
  - `calc_mrope_positions(self, scheduler_output: VllmSchedulerOutput, req_ids_dp: dict[int, list[str]], padded_num_scheduled_tokens_per_dp_rank: int)` — [`L40`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/multimodal_manager.py#L40) — Calculate and update the mrope_positions for the scheduled tokens in
  - `execute_mm_encoder(self, scheduler_output: VllmSchedulerOutput)` — [`L113`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/multimodal_manager.py#L113)
  - `gather_mm_embeddings(self, scheduler_output: VllmSchedulerOutput, target_pad_len: int, req_ids_dp: dict[int, list[str]], padded_num_scheduled_tokens_per_dp_rank: int)` — [`L166`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/multimodal_manager.py#L166) — Gather multimodal_embeddings from the encoder cache with is_multimodal.
  - `runner` — [`L38`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/multimodal_manager.py#L38)
- protocol/private: `__init__`[`L37`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/multimodal_manager.py#L37)
- uses (calls/refs, reference-scoped): `input_batch`, [`req_id_to_index`](input_batch.md#InputBatch.req_id_to_index), `state_leaves`, `requests`, [`num_computed_tokens_cpu`](input_batch.md#InputBatch.num_computed_tokens_cpu), `encoder_cache`, `TPUModelRunner`, `embed_multimodal_fn`, `mrope_positions_cpu`, [`sanity_check_mm_encoder_outputs`](../models/jax/utils/multi_modal_utils.md#sanity_check_mm_encoder_outputs), [`mrope_positions`](input_batch.md#CachedRequestState.mrope_positions), [`mrope_position_delta`](input_batch.md#CachedRequestState.mrope_position_delta)
- used by: `_execute_model`, `_prepare_inputs`, `mm_manager`

## Module values
- `logger` — [`L29`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/multimodal_manager.py#L29)

