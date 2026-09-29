---
title: 'Module: tpu_inference/runner/input_batch.py'
type: catalog
provenance: extracted
module: tpu_inference/runner/input_batch.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.runner.input_batch`/
symbols:
  InputBatch.add_request: InputBatch#add_request().
  InputBatch.remove_request: InputBatch#remove_request().
  InputBatch.condense: InputBatch#condense().
  InputBatch.swap_states: InputBatch#swap_states().
  InputBatch.collect_sampling: InputBatch#collect_sampling().
  InputBatch.num_reqs: InputBatch#num_reqs().
  InputBatch.req_id_to_index: InputBatch#req_id_to_index.
  InputBatch.req_ids: InputBatch#req_ids().
  InputBatch.num_tokens_no_spec: InputBatch#num_tokens_no_spec.
  InputBatch.num_tokens: InputBatch#num_tokens.
  InputBatch.token_ids_cpu: InputBatch#token_ids_cpu.
  CachedRequestState: CachedRequestState#
  CachedRequestState.output_token_ids: CachedRequestState#output_token_ids.
  InputBatch.assert_mamba_state_invariants: InputBatch#assert_mamba_state_invariants().
  InputBatch._req_ids: InputBatch#_req_ids.
  InputBatch.num_computed_tokens_cpu: InputBatch#num_computed_tokens_cpu.
  InputBatch.mamba_state_indices_cpu: InputBatch#mamba_state_indices_cpu.
  CachedRequestState.mamba_state_slot: CachedRequestState#mamba_state_slot.
  InputBatch.get_pooling_metadata: InputBatch#get_pooling_metadata().
  InputBatch.num_prompt_logprobs: InputBatch#num_prompt_logprobs.
  InputBatch: InputBatch#
  InputBatch.allowed_token_ids_mask_cpu: InputBatch#allowed_token_ids_mask_cpu.
  InputBatch.request_lora_mapping: InputBatch#request_lora_mapping.
  InputBatch.num_prompt_tokens: InputBatch#num_prompt_tokens.
  InputBatch._free_mamba_slots_per_rank: InputBatch#_free_mamba_slots_per_rank.
  InputBatch.block_table: InputBatch#block_table.
  InputBatch._mamba_local_slots: InputBatch#_mamba_local_slots.
  InputBatch.req_output_token_ids: InputBatch#req_output_token_ids.
  CachedRequestState.num_tokens: CachedRequestState#num_tokens().
  InputBatch.init_mamba_pools: InputBatch#init_mamba_pools().
  InputBatch.temperature_cpu: InputBatch#temperature_cpu.
  CachedRequestState.in_progress_prompt_logprobs_cpu: CachedRequestState#in_progress_prompt_logprobs_cpu.
  InputBatch.make_lora_inputs: InputBatch#make_lora_inputs().
  InputBatch.top_k_cpu: InputBatch#top_k_cpu.
  InputBatch.top_p_cpu: InputBatch#top_p_cpu.
  InputBatch.logit_bias: InputBatch#logit_bias.
  InputBatch.release_mamba_slot: InputBatch#release_mamba_slot().
  InputBatch.generators: InputBatch#generators.
  InputBatch.vocab_size: InputBatch#vocab_size.
  CachedRequestState.get_token_id: CachedRequestState#get_token_id().
  InputBatch.lora_id_to_request_ids: InputBatch#lora_id_to_request_ids.
  InputBatch.get_pooling_params: InputBatch#get_pooling_params().
  InputBatch.get_pooling_states: InputBatch#get_pooling_states().
  InputBatch.max_num_reqs: InputBatch#max_num_reqs.
  InputBatch.min_tokens: InputBatch#min_tokens.
  InputBatch.bad_words_token_ids: InputBatch#bad_words_token_ids.
  InputBatch.all_greedy: InputBatch#all_greedy().
  InputBatch.max_num_logprobs: InputBatch#max_num_logprobs().
  InputBatch.request_distribution: InputBatch#request_distribution.
  CachedRequestState.num_prompt_tokens: CachedRequestState#num_prompt_tokens.
  CachedRequestState.generator: CachedRequestState#generator.
  InputBatch.num_logprobs: InputBatch#num_logprobs.
  InputBatch.has_mamba_layers: InputBatch#has_mamba_layers.
  CachedRequestState.mrope_positions: CachedRequestState#mrope_positions.
  InputBatch.random_reqs: InputBatch#random_reqs.
  InputBatch.lora_id_to_lora_request: InputBatch#lora_id_to_lora_request.
  InputBatch.pooling_params: InputBatch#pooling_params.
  InputBatch.pooling_states: InputBatch#pooling_states.
  InputBatch.max_num_prompt_logprobs: InputBatch#max_num_prompt_logprobs().
  CachedRequestState.mrope_position_delta: CachedRequestState#mrope_position_delta.
  InputBatch.dp_size: InputBatch#dp_size.
  InputBatch.greedy_reqs: InputBatch#greedy_reqs.
  InputBatch.has_allowed_token_ids: InputBatch#has_allowed_token_ids.
  CachedRequestState.pooling_states: CachedRequestState#pooling_states.
  InputBatch.spec_decode_unsupported_reqs: InputBatch#spec_decode_unsupported_reqs.
  InputBatch.allowed_token_ids_mask: InputBatch#allowed_token_ids_mask.
  _SAMPLING_EPS: _SAMPLING_EPS.
  CachedRequestState.__post_init__: CachedRequestState#__post_init__().
  InputBatch.__init__: InputBatch#__init__().
  InputBatch.is_spec_decode: InputBatch#is_spec_decode.
  InputBatch.max_model_len: InputBatch#max_model_len.
  InputBatch.max_num_batched_tokens: InputBatch#max_num_batched_tokens.
  InputBatch.pin_memory: InputBatch#pin_memory.
---
# Module: [`tpu_inference/runner/input_batch.py`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py)

## Classes
### `CachedRequestState`
- def: [`tpu_inference/runner/input_batch.py:26`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L26)
- signature: `class CachedRequestState(NewRequestData):`
- members:
  - `get_token_id(self, idx: int)` — [`L46`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L46)
  - `num_tokens(self)` — [`L43`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L43)
  - `generator` — [`L29`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L29)
  - `in_progress_prompt_logprobs_cpu` — [`L36`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L36)
  - `mamba_state_slot` — [`L37`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L37)
  - `mrope_position_delta` — [`L31`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L31)
  - `mrope_positions` — [`L30`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L30)
  - `num_prompt_tokens` — [`L40`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L40)
  - `output_token_ids` — [`L28`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L28)
  - `pooling_states` — [`L32`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L32)
- protocol/private: `__post_init__`[`L39`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L39)
- used by: `_sample_from_logits`, `_execute_continue_decode`, [`update_states`](persistent_batch_manager.md#PersistentBatchManager.update_states), [`add_request`](input_batch.md#InputBatch.add_request), [`propose_eagle3_draft_token_ids`](speculative_decoding_manager.md#SpeculativeDecodingManager.propose_eagle3_draft_token_ids), `_modify_prev_results`, [`compute_prompt_logprobs`](../layers/jax/sample/sampling.md#compute_prompt_logprobs), [`collect_sampling`](input_batch.md#InputBatch.collect_sampling), `_get_prompt_logprobs_dict`, [`insert_request_with_kv_cache`](kv_cache_manager.md#KVCacheManager.insert_request_with_kv_cache), [`calc_mrope_positions`](multimodal_manager.md#MultiModalManager.calc_mrope_positions), [`assert_mamba_state_invariants`](input_batch.md#InputBatch.assert_mamba_state_invariants), `requests`, [`req_state`](../layers/jax/sample/sampling.md#PromptLogprobsReqSnap.req_state), `_reconstruct_slots_for_request`, `request_seq_lens`, [`__init__`](persistent_batch_manager.md#PersistentBatchManager.__init__)

### `InputBatch`
- def: [`tpu_inference/runner/input_batch.py:53`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L53)
- signature: `class InputBatch:`
- members:
  - `add_request(self, request: CachedRequestState, req_index: Optional[int] = None, dp_rank: int = 0)` — [`L316`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L316)
  - `all_greedy(self)` — [`L653`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L653)
  - `assert_mamba_state_invariants(self, requests: Optional[dict[str, CachedRequestState]] = None, assigned_dp_rank: Optional[dict[str, int]] = None)` — [`L206`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L206)
  - `collect_sampling(sampling_params: SamplingParams)` — [`L375`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L375)
  - `condense(self, empty_req_indices: list[int])` — [`L563`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L563)
  - `get_pooling_metadata(self)` — [`L292`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L292)
  - `get_pooling_params(self)` — [`L284`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L284)
  - `get_pooling_states(self)` — [`L288`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L288)
  - `init_mamba_pools(self, mamba_num_blocks: int)` — [`L181`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L181) — Reinitialize mamba slot pools with the actual device block count.
  - `make_lora_inputs(self, num_scheduled_tokens: np.ndarray)` — [`L665`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L665) — Given the num_scheduled_tokens for each request in the batch, return
  - `max_num_logprobs(self)` — [`L657`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L657)
  - `max_num_prompt_logprobs(self)` — [`L661`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L661)
  - `num_reqs(self)` — [`L649`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L649)
  - `release_mamba_slot(self, slot: Optional[int])` — [`L195`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L195)
  - `remove_request(self, req_id: str, *, free_mamba_slot: bool = True)` — [`L450`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L450) — This method must always be followed by a call to condense().
  - `req_ids(self)` — [`L279`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L279)
  - `swap_states(self, i1: int, i2: int)` — [`L512`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L512)
  - `allowed_token_ids_mask` — [`L138`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L138)
  - `allowed_token_ids_mask_cpu` — [`L139`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L139)
  - `bad_words_token_ids` — [`L142`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L142)
  - `block_table` — [`L95`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L95)
  - `dp_size` — [`L69`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L69)
  - `generators` — [`L128`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L128)
  - `greedy_reqs` — [`L106`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L106)
  - `has_allowed_token_ids` — [`L135`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L135)
  - `has_mamba_layers` — [`L179`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L179)
  - `is_spec_decode` — [`L67`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L67)
  - `logit_bias` — [`L133`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L133)
  - `lora_id_to_lora_request` — [`L123`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L123)
  - `lora_id_to_request_ids` — [`L122`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L122)
  - `mamba_state_indices_cpu` — [`L163`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L163)
  - `max_model_len` — [`L70`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L70)
  - `max_num_batched_tokens` — [`L71`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L71)
  - `max_num_reqs` — [`L68`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L68)
  - `min_tokens` — [`L117`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L117)
  - `num_computed_tokens_cpu` — [`L89`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L89)
  - `num_logprobs` — [`L130`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L130)
  - `num_prompt_logprobs` — [`L131`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L131)
  - `num_prompt_tokens` — [`L88`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L88)
  - `num_tokens` — [`L86`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L86)
  - `num_tokens_no_spec` — [`L87`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L87)
  - `pin_memory` — [`L72`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L72)
  - `pooling_params` — [`L177`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L177)
  - `pooling_states` — [`L178`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L178)
  - `random_reqs` — [`L107`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L107)
  - `req_id_to_index` — [`L76`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L76)
  - `req_output_token_ids` — [`L144`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L144)
  - `request_distribution` — [`L146`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L146)
  - `request_lora_mapping` — [`L120`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L120)
  - `spec_decode_unsupported_reqs` — [`L114`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L114)
  - `temperature_cpu` — [`L105`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L105)
  - `token_ids_cpu` — [`L82`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L82)
  - `top_k_cpu` — [`L111`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L111)
  - `top_p_cpu` — [`L109`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L109)
  - `vocab_size` — [`L73`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L73)
- protocol/private: `__init__`[`L55`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L55), `_free_mamba_slots_per_rank`[`L170`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L170), `_mamba_local_slots`[`L169`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L169), `_req_ids`[`L75`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L75)
- uses (calls/refs, reference-scoped): [`CachedRequestState`](input_batch.md#CachedRequestState), [`output_token_ids`](input_batch.md#CachedRequestState.output_token_ids), [`mamba_state_slot`](input_batch.md#CachedRequestState.mamba_state_slot), [`num_tokens`](input_batch.md#CachedRequestState.num_tokens), [`in_progress_prompt_logprobs_cpu`](input_batch.md#CachedRequestState.in_progress_prompt_logprobs_cpu), [`add_row`](block_table.md#MultiGroupBlockTable.add_row), [`move_row`](block_table.md#MultiGroupBlockTable.move_row), [`swap_row`](block_table.md#MultiGroupBlockTable.swap_row), [`generator`](input_batch.md#CachedRequestState.generator), [`MultiGroupBlockTable`](block_table.md#MultiGroupBlockTable), [`pooling_states`](input_batch.md#CachedRequestState.pooling_states)
- used by: `input_batch`, `_sample_from_logits`, `_execute_model`, `_prepare_inputs`, `_execute_continue_decode`, [`_precompile_mtp_helpers`](compilation_manager.md#CompilationManager._precompile_mtp_helpers), [`update_states`](persistent_batch_manager.md#PersistentBatchManager.update_states), [`_precompile_eagle3_helpers`](compilation_manager.md#CompilationManager._precompile_eagle3_helpers), [`propose_eagle3_draft_token_ids`](speculative_decoding_manager.md#SpeculativeDecodingManager.propose_eagle3_draft_token_ids), `_modify_prev_results`, [`propose_draft_token_ids`](speculative_decoding_manager.md#SpeculativeDecodingManager.propose_draft_token_ids), `_get_prompt_logprobs_dict`, [`insert_request_with_kv_cache`](kv_cache_manager.md#KVCacheManager.insert_request_with_kv_cache), [`maybe_reinitialize_input_batch`](kv_cache_manager.md#KVCacheManager.maybe_reinitialize_input_batch), `use_hybrid_kvcache`, `_prepare_input_metadata`, [`from_input_batch`](../layers/jax/sample/sampling_metadata.md#TPUSupportedSamplingMetadata.from_input_batch), `_update_placeholder`, [`calc_mrope_positions`](multimodal_manager.md#MultiModalManager.calc_mrope_positions), `_prepare_async_token_substitution_indices`, [`prepare_structured_decoding_input`](structured_decoding_manager.md#StructuredDecodingManager.prepare_structured_decoding_input), [`set_active_loras`](lora_utils.md#LoraUtils.set_active_loras), [`take_draft_token_ids`](speculative_decoding_manager.md#SpeculativeDecodingManager.take_draft_token_ids), [`build_block_table`](compilation_manager.md#CompilationManager.build_block_table), `build_block_table_host`, `execute_model`, [`extract_request_ids_for_tracing`](../utils.md#extract_request_ids_for_tracing), [`_reorder_batch`](persistent_batch_manager.md#PersistentBatchManager._reorder_batch), `_get_min_remaining_slots`, [`get_batch_composition_stats`](../utils.md#get_batch_composition_stats), [`__init__`](persistent_batch_manager.md#PersistentBatchManager.__init__)

## Module values
- `_SAMPLING_EPS` — [`L20`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/input_batch.py#L20)

