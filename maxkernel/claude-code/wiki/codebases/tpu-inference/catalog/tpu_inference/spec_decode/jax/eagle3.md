---
title: 'Module: tpu_inference/spec_decode/jax/eagle3.py'
type: catalog
provenance: extracted
module: tpu_inference/spec_decode/jax/eagle3.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.spec_decode.jax.eagle3`/
symbols:
  Eagle3Proposer._propose: Eagle3Proposer#_propose().
  Eagle3Proposer.model: Eagle3Proposer#model.
  Eagle3Proposer.prepare_inputs: Eagle3Proposer#prepare_inputs().
  Eagle3Proposer._prepare_inputs: Eagle3Proposer#_prepare_inputs().
  Eagle3Proposer.propose: Eagle3Proposer#propose().
  Eagle3Proposer._filter_token_and_prepare_initial_inputs: Eagle3Proposer#_filter_token_and_prepare_initial_inputs().
  Eagle3Proposer.load_model: Eagle3Proposer#load_model().
  Eagle3Proposer._select_draft_token_ids: Eagle3Proposer#_select_draft_token_ids().
  Eagle3Proposer._select_inputs_for_loop_speculation: Eagle3Proposer#_select_inputs_for_loop_speculation().
  Eagle3Proposer.mesh: Eagle3Proposer#mesh.
  logger: logger.
  Eagle3Proposer._get_draft_token_ids: Eagle3Proposer#_get_draft_token_ids().
  Eagle3Proposer.speculative_config: Eagle3Proposer#speculative_config.
  Eagle3Proposer._prepare_input_ids: Eagle3Proposer#_prepare_input_ids().
  Eagle3Proposer._update_inputs_for_loop_speculation: Eagle3Proposer#_update_inputs_for_loop_speculation().
  Eagle3Proposer._get_loop_query_start_loc: Eagle3Proposer#_get_loop_query_start_loc().
  Eagle3Proposer._prepare_hidden_states_and_input_ids: Eagle3Proposer#_prepare_hidden_states_and_input_ids().
  Eagle3Proposer.state: Eagle3Proposer#state.
  Eagle3Proposer.state_leaves: Eagle3Proposer#state_leaves.
  Eagle3Proposer: Eagle3Proposer#
  Eagle3Proposer.runner: Eagle3Proposer#runner.
  Eagle3Proposer.method: Eagle3Proposer#method.
  Eagle3Proposer.model_fn: Eagle3Proposer#model_fn.
  Eagle3Proposer._select: Eagle3Proposer#_select().
  Eagle3Proposer.vllm_config: Eagle3Proposer#vllm_config.
  Eagle3Proposer.num_speculative_tokens: Eagle3Proposer#num_speculative_tokens.
  Eagle3Proposer.rng_key: Eagle3Proposer#rng_key.
  Eagle3Proposer.constant_draft_positions: Eagle3Proposer#constant_draft_positions.
  Eagle3Proposer.compute_logits_fn: Eagle3Proposer#compute_logits_fn.
  Eagle3Proposer.combine_hidden_states_fn: Eagle3Proposer#combine_hidden_states_fn.
  Eagle3Proposer._sharded_update_inputs_for_loop_speculation: Eagle3Proposer#_sharded_update_inputs_for_loop_speculation().
  Eagle3Proposer.draft_model_config: Eagle3Proposer#draft_model_config.
  Eagle3Proposer.token_arange: Eagle3Proposer#token_arange.
  Eagle3Proposer.pooler_fn: Eagle3Proposer#pooler_fn.
  Eagle3Proposer.max_num_tokens: Eagle3Proposer#max_num_tokens.
  Eagle3Proposer._sharded_prepare_input_ids: Eagle3Proposer#_sharded_prepare_input_ids().
  Eagle3Proposer._sharded_get: Eagle3Proposer#_sharded_get().
  Eagle3Proposer._stack_draft_token_ids: Eagle3Proposer#_stack_draft_token_ids().
  Eagle3Proposer._compute_token_indices: Eagle3Proposer#_compute_token_indices().
  Eagle3Proposer._sharded_select_target_tokens_and_hidden_states: Eagle3Proposer#_sharded_select_target_tokens_and_hidden_states().
  Eagle3Proposer.__init__: Eagle3Proposer#__init__().
  Eagle3Proposer.block_size: Eagle3Proposer#block_size.
---
# Module: [`tpu_inference/spec_decode/jax/eagle3.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py)

## Classes
### `Eagle3Proposer`
- def: [`tpu_inference/spec_decode/jax/eagle3.py:37`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L37)
- doc: A proposer for speculative decoding using the Eagle3 method.
- signature: `class Eagle3Proposer:`
- members:
  - `__init__(self, vllm_config: VllmConfig, runner: Any)` — [`L44`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L44) — Initializes the Eagle3Proposer.
  - `_get_loop_query_start_loc(self, positions: jax.Array)` — [`L268`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L268) — JIT-compiled helper for generating query_start_loc inside speculation loop.
  - `_prepare_input_ids(self, query_start_loc: jax.Array, target_token_ids: jax.Array, next_token_ids: jax.Array, num_reqs: jax.Array)` — [`L176`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L176) — JIT-compiled helper for preparing the input IDs for the draft model.
  - `_prepare_inputs(self, state_leaves: Any, num_reqs: jax.Array, block_tables: jax.Array, attn_metadata: AttentionMetadata, input_ids: jax.Array, aux_hidden_states: tuple[jax.Array, ...], last_sampled_token_id: jax.Array, next_prompt_token_id: jax.Array, is_in_prefill: jax.Array, num_rejected_tokens: jax.Array)` — [`L360`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L360) — Prepare drafter inputs based on target forward outputs.
  - `_propose(self, state_leaves: Any, kv_caches: list[jax.Array], input_ids: jax.Array, attn_metadata: AttentionMetadata, last_token_indices, target_hidden_states, num_speculative_tokens: int, layer_name_to_kvcache_index: tuple)` — [`L615`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L615) — Proposes draft tokens using the draft model.
  - `_stack_draft_token_ids(self, draft_token_ids_list: list[jax.Array])` — [`L283`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L283) — JIT-compiled helper for stacking draft token IDs.
  - `_update_inputs_for_loop_speculation(self, positions: jax.Array, seq_lens: jax.Array, block_tables: jax.Array)` — [`L213`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L213) — JIT-compiled helper for preparing inputs in the loop of prediction.
  - `load_model(self, target_model: Any)` — [`L72`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L72) — Loads the draft model.
  - `prepare_inputs(self, attn_metadata: AttentionMetadata, input_ids: jax.Array, aux_hidden_states: tuple[jax.Array, ...], last_sampled_token_id: jax.Array, next_prompt_token_id: jax.Array, is_in_prefill: jax.Array, num_rejected_tokens: jax.Array, num_reqs_dp: jax.Array)` — [`L310`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L310) — Prepare drafter inputs based on target forward outputs.
  - `propose(self, kv_caches: list[jax.Array], input_ids: jax.Array, attn_metadata: AttentionMetadata, last_token_indices, target_hidden_states)` — [`L578`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L578)
  - `block_size` — [`L65`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L65)
  - `combine_hidden_states_fn` — [`L94`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L94)
  - `compute_logits_fn` — [`L92`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L92)
  - `constant_draft_positions` — [`L69`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L69)
  - `draft_model_config` — [`L58`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L58)
  - `max_num_tokens` — [`L67`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L67)
  - `mesh` — [`L62`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L62)
  - `method` — [`L59`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L59)
  - `model` — [`L97`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L97)
  - `model_fn` — [`L91`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L91)
  - `num_speculative_tokens` — [`L63`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L63)
  - `pooler_fn` — [`L93`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L93)
  - `rng_key` — [`L66`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L66)
  - `runner` — [`L61`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L61)
  - `speculative_config` — [`L56`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L56)
  - `state` — [`L95`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L95)
  - `state_leaves` — [`L96`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L96)
  - `token_arange` — [`L68`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L68)
  - `vllm_config` — [`L55`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L55)
- protocol/private: `_compute_token_indices`[`L388`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L388), `_filter_token_and_prepare_initial_inputs`[`L457`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L457), `_get_draft_token_ids`[`L529`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L529), `_prepare_hidden_states_and_input_ids`[`L288`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L288), `_select`[`L517`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L517), `_select_draft_token_ids`[`L510`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L510), `_select_inputs_for_loop_speculation`[`L540`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L540), `_sharded_get`[`L271`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L271), `_sharded_prepare_input_ids`[`L182`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L182), `_sharded_select_target_tokens_and_hidden_states`[`L476`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L476), `_sharded_update_inputs_for_loop_speculation`[`L219`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L219)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../layers/common/sharding.md#ShardingAxisName), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`input_positions`](../../layers/common/attention_metadata.md#AttentionMetadata.input_positions), [`get_model`](../../models/common/model_loader.md#get_model), [`seq_lens`](../../layers/common/attention_metadata.md#AttentionMetadata.seq_lens), [`block_tables`](../../layers/common/attention_metadata.md#AttentionMetadata.block_tables), [`query_start_loc`](../../layers/common/attention_metadata.md#AttentionMetadata.query_start_loc), [`get_param`](../../models/jax/utils/weight_utils.md#get_param), [`resolve_model_architecture`](../../models/common/model_loader.md#resolve_model_architecture), [`logger`](eagle3.md#logger), [`combine_hidden_states_fn`](../../models/common/interface.md#ModelInterface.combine_hidden_states_fn), [`compute_logits_fn`](../../models/common/interface.md#ModelInterface.compute_logits_fn), [`model`](../../models/common/interface.md#ModelInterface.model), [`model_fn`](../../models/common/interface.md#ModelInterface.model_fn), [`pooler_fn`](../../models/common/interface.md#ModelInterface.pooler_fn), [`state`](../../models/common/interface.md#ModelInterface.state), [`state_leaves`](../../models/common/interface.md#ModelInterface.state_leaves), [`DRAFT_MODEL_IMPL_TYPE`](../../envs.md#DRAFT_MODEL_IMPL_TYPE), [`MODEL_IMPL_TYPE`](../../envs.md#MODEL_IMPL_TYPE)
- used by: [`_precompile_mtp_helpers`](../../runner/compilation_manager.md#CompilationManager._precompile_mtp_helpers), [`_precompile_eagle3_helpers`](../../runner/compilation_manager.md#CompilationManager._precompile_eagle3_helpers), [`propose_eagle3_draft_token_ids`](../../runner/speculative_decoding_manager.md#SpeculativeDecodingManager.propose_eagle3_draft_token_ids), [`propose_draft_token_ids`](../../runner/speculative_decoding_manager.md#SpeculativeDecodingManager.propose_draft_token_ids), `state`, `drafter`, [`drafter_propose_warmup`](../../runner/compilation_manager.md#CompilationManager.drafter_propose_warmup)

## Module values
- `logger` — [`L34`](../../../../../../../raw/code/tpu-inference/tpu_inference/spec_decode/jax/eagle3.py#L34)

