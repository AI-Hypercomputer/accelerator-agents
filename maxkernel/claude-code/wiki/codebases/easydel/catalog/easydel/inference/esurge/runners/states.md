---
title: 'Module: easydel/inference/esurge/runners/states.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/runners/states.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.runners.states`/CachedRequestState#
symbols:
  CachedRequestState.clear_vision_data: clear_vision_data().
  CachedRequestState: ''
  CachedRequestState.num_prompt_tokens: num_prompt_tokens.
  CachedRequestState.has_vision: has_vision().
  CachedRequestState.get_token_id: get_token_id().
  CachedRequestState.output_token_ids: output_token_ids.
  CachedRequestState.prefill_inputs_embeds: prefill_inputs_embeds.
  CachedRequestState.pixel_values: pixel_values.
  CachedRequestState.pixel_values_videos: pixel_values_videos.
  CachedRequestState.prefill_position_ids: prefill_position_ids.
  CachedRequestState.prefill_rope_deltas: prefill_rope_deltas.
  CachedRequestState.prefill_visual_pos_masks: prefill_visual_pos_masks.
  CachedRequestState.__post_init__: __post_init__().
  CachedRequestState.num_tokens: num_tokens().
  CachedRequestState.vision_processed: vision_processed().
  CachedRequestState.prefill_deepstack_visual_embeds: prefill_deepstack_visual_embeds.
  CachedRequestState.req_id: req_id.
  CachedRequestState.prompt_token_ids: prompt_token_ids.
  CachedRequestState.page_ids: page_ids.
  CachedRequestState.num_computed_tokens: num_computed_tokens.
  CachedRequestState.image_grid_thw: image_grid_thw.
  CachedRequestState.video_grid_thw: video_grid_thw.
  CachedRequestState._vision_processed: _vision_processed.
  CachedRequestState.sampling_params: sampling_params.
  CachedRequestState.mm_features: mm_features.
  CachedRequestState.generator: generator.
---
# Module: [`easydel/inference/esurge/runners/states.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py)

## Classes
### `CachedRequestState`
- def: [`easydel/inference/esurge/runners/states.py:54`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L54)
- doc: Represents the state of a single request, compatible with JAX PyTree.
- signature: `class CachedRequestState:`
- members:
  - `__post_init__(self)` — [`L141`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L141) — Initialize computed fields after instance creation.
  - `clear_vision_data(self)` — [`L177`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L177) — Clear raw vision data after prefill to free memory.
  - `get_token_id(self, idx: int)` — [`L201`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L201) — Get token ID at a specific position.
  - `has_vision(self)` — [`L158`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L158) — Check if request has vision data (images or videos).
  - `num_tokens(self)` — [`L149`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L149) — Get total number of tokens (prompt + generated).
  - `vision_processed(self)` — [`L168`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L168) — Check if vision data has been processed (prefill complete).
  - `generator` — [`L121`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L121)
  - `image_grid_thw` — [`L128`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L128)
  - `mm_features` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L132)
  - `num_computed_tokens` — [`L123`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L123)
  - `num_prompt_tokens` — [`L125`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L125)
  - `output_token_ids` — [`L124`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L124)
  - `page_ids` — [`L122`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L122)
  - `pixel_values` — [`L127`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L127)
  - `pixel_values_videos` — [`L129`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L129)
  - `prefill_deepstack_visual_embeds` — [`L139`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L139)
  - `prefill_inputs_embeds` — [`L135`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L135)
  - `prefill_position_ids` — [`L136`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L136)
  - `prefill_rope_deltas` — [`L137`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L137)
  - `prefill_visual_pos_masks` — [`L138`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L138)
  - `prompt_token_ids` — [`L119`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L119)
  - `req_id` — [`L118`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L118)
  - `sampling_params` — [`L120`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L120)
  - `video_grid_thw` — [`L130`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L130)
- protocol/private: `_vision_processed`[`L133`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/states.py#L133)
- uses (calls/refs, reference-scoped): [`SamplingParams`](../../sampling_params.md#SamplingParams)
- used by: [`_execute_model_impl`](model_runner.md#eSurgeRunner._execute_model_impl), [`_update_states`](model_runner.md#eSurgeRunner._update_states), [`_precompute_vlm_prefill`](model_runner.md#eSurgeRunner._precompute_vlm_prefill), [`requests`](model_runner.md#eSurgeRunner.requests), [`_modify_prev_results`](model_runner.md#eSurgeRunner._modify_prev_results), [`_update_placeholder`](model_runner.md#eSurgeRunner._update_placeholder), [`_finalize_sync_runner_state`](model_runner.md#eSurgeRunner._finalize_sync_runner_state), [`AsyncPreResults`](async_types.md#AsyncPreResults)

