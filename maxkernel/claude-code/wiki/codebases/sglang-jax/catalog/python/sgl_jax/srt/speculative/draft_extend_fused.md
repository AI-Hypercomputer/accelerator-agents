---
title: 'Module: python/sgl_jax/srt/speculative/draft_extend_fused.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/speculative/draft_extend_fused.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.speculative.draft_extend_fused`/
symbols:
  spec_prefill: spec_prefill().
  _make_forward_batch: _make_forward_batch().
  spec_decode_verify: spec_decode_verify().
  _verify_rejection_sampling: _verify_rejection_sampling().
  launch_fused_draft_extend_for_decode: launch_fused_draft_extend_for_decode().
  _verify_greedy: _verify_greedy().
  _build_verify.fused_verify: _build_verify().fused_verify().
  _prepare_logits_metadata: _prepare_logits_metadata().
  _prepare_verify: _prepare_verify().
  _prepare_device_array: _prepare_device_array().
  _make_target_verify_metadata: _make_target_verify_metadata().
  _make_draft_extend_metadata: _make_draft_extend_metadata().
  _prepare_draft_inputs: _prepare_draft_inputs().
  spec_decode_overlap: spec_decode_overlap().
  prepare_forward_batch_for_prefill: prepare_forward_batch_for_prefill().
  _build_draft_extend.fused_draft_extend: _build_draft_extend().fused_draft_extend().
  _build_prefill.fused_prefill: _build_prefill().fused_prefill().
  _verify_rejection_sampling._rep: _verify_rejection_sampling()._rep().
  draft_extend_for_decode: draft_extend_for_decode().
  spec_decode: spec_decode().
  _prepare_rejection_sampling: _prepare_rejection_sampling().
  GreedySampleAndPrepareOutput.verified_id: GreedySampleAndPrepareOutput#verified_id.
  _rotate_prefill_input_ids: _rotate_prefill_input_ids().
  spec_prefill_overlap: spec_prefill_overlap().
  GreedySampleAndPrepareOutput.positions: GreedySampleAndPrepareOutput#positions.
  GreedySampleAndPrepareOutput.accept_lens: GreedySampleAndPrepareOutput#accept_lens.
  GreedySampleAndPrepareOutput.sel_pos: GreedySampleAndPrepareOutput#sel_pos.
  _repack_page_indices: _repack_page_indices().
  _build_draft_extend: _build_draft_extend().
  _build_verify: _build_verify().
  _build_prefill: _build_prefill().
  GreedyDraftInputs.hidden_states: GreedyDraftInputs#hidden_states.
  GreedyDraftInputs.positions: GreedyDraftInputs#positions.
  GreedyDraftInputs.new_seq_lens: GreedyDraftInputs#new_seq_lens.
  GreedyDraftInputs.select_index: GreedyDraftInputs#select_index.
  GreedyDraftInputs.verified_id: GreedyDraftInputs#verified_id.
  GreedyDraftInputs.accept_lens: GreedyDraftInputs#accept_lens.
  GreedyDraftInputs.sel_pos: GreedyDraftInputs#sel_pos.
  GreedySampleAndPrepareOutput.hidden_states: GreedySampleAndPrepareOutput#hidden_states.
  GreedySampleAndPrepareOutput.new_seq_lens: GreedySampleAndPrepareOutput#new_seq_lens.
  GreedySampleAndPrepareOutput.select_index: GreedySampleAndPrepareOutput#select_index.
  GreedySampleAndPrepareOutput.safe_index: GreedySampleAndPrepareOutput#safe_index.
  GreedySampleAndPrepareOutput.predict: GreedySampleAndPrepareOutput#predict.
  _take_with_index_sharding: _take_with_index_sharding().
  _gather_rows_preserve_sharding: _gather_rows_preserve_sharding().
  _per_dp_cumsum_device: _per_dp_cumsum_device().
  GreedySampleAndPrepareOutput: GreedySampleAndPrepareOutput#
  FusedDraftExtendPendingResult.updated_relay_buffers: FusedDraftExtendPendingResult#updated_relay_buffers.
  _count_pjit_cpp_cache_miss: _count_pjit_cpp_cache_miss().
  _reshard_values: _reshard_values().
  _topk1_index_from_logits: _topk1_index_from_logits().
  GreedyDraftInputs: GreedyDraftInputs#
  FusedDraftExtendPendingResult: FusedDraftExtendPendingResult#
  FusedDraftExtendPendingResult.batch_output: FusedDraftExtendPendingResult#batch_output.
  FusedDraftExtendPendingResult.selected_layer0_hidden: FusedDraftExtendPendingResult#selected_layer0_hidden.
  FusedDraftExtendPendingResult.topk_index_stacked: FusedDraftExtendPendingResult#topk_index_stacked.
  FusedDraftExtendPendingResult.next_verified_id: FusedDraftExtendPendingResult#next_verified_id.
  FusedDraftExtendPendingResult.accept_lens: FusedDraftExtendPendingResult#accept_lens.
  FusedDraftExtendPendingResult.sel: FusedDraftExtendPendingResult#sel.
  _active_dp_slot_mask: _active_dp_slot_mask().
  _prepare_spec_prefill_output_token_ids: _prepare_spec_prefill_output_token_ids().
  _build_chain_verify_arrays: _build_chain_verify_arrays().
  _rotate_input_ids: _rotate_input_ids().
  _rotate_prefill_input_ids.rotate_rank: _rotate_prefill_input_ids().rotate_rank().
  restore_draft_extend_result: restore_draft_extend_result().
  spec_decode_draft_extend: spec_decode_draft_extend().
---
# Module: [`python/sgl_jax/srt/speculative/draft_extend_fused.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py)

## Classes
### `FusedDraftExtendPendingResult`  ·  implements/extends NamedTuple
- def: [`python/sgl_jax/srt/speculative/draft_extend_fused.py:47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L47)
- signature: `class FusedDraftExtendPendingResult(NamedTuple):`
- members:
  - `accept_lens` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L52)
  - `batch_output` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L48)
  - `next_verified_id` — [`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L51)
  - `sel` — [`L53`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L53)
  - `selected_layer0_hidden` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L49)
  - `topk_index_stacked` — [`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L50)
  - `updated_relay_buffers` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L54)
- used by: [`launch_fused_draft_extend_for_decode`](draft_extend_fused.md#launch_fused_draft_extend_for_decode), [`spec_decode_overlap`](draft_extend_fused.md#spec_decode_overlap)

### `GreedyDraftInputs`  ·  implements/extends NamedTuple
- def: [`python/sgl_jax/srt/speculative/draft_extend_fused.py:25`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L25)
- signature: `class GreedyDraftInputs(NamedTuple):`
- members:
  - `accept_lens` — [`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L31)
  - `hidden_states` — [`L26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L26)
  - `new_seq_lens` — [`L28`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L28)
  - `positions` — [`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L27)
  - `sel_pos` — [`L32`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L32)
  - `select_index` — [`L29`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L29)
  - `verified_id` — [`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L30)
- used by: [`_verify_rejection_sampling`](draft_extend_fused.md#_verify_rejection_sampling), [`_verify_greedy`](draft_extend_fused.md#_verify_greedy), [`_prepare_draft_inputs`](draft_extend_fused.md#_prepare_draft_inputs)

### `GreedySampleAndPrepareOutput`  ·  implements/extends NamedTuple
- def: [`python/sgl_jax/srt/speculative/draft_extend_fused.py:35`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L35)
- signature: `class GreedySampleAndPrepareOutput(NamedTuple):`
- members:
  - `accept_lens` — [`L42`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L42)
  - `hidden_states` — [`L36`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L36)
  - `new_seq_lens` — [`L38`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L38)
  - `positions` — [`L37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L37)
  - `predict` — [`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L44)
  - `safe_index` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L40)
  - `sel_pos` — [`L43`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L43)
  - `select_index` — [`L39`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L39)
  - `verified_id` — [`L41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L41)
- used by: [`_verify_rejection_sampling`](draft_extend_fused.md#_verify_rejection_sampling), [`_verify_greedy`](draft_extend_fused.md#_verify_greedy), [`fused_verify`](draft_extend_fused.md#_build_verify.fused_verify)

## Functions
- `_active_dp_slot_mask(batch, total_bs: int)` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L68)
- `_build_chain_verify_arrays(*, verified_id, token_list, seq_lens, num_verify_tokens, batch_size)` — [`L370`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L370) — Build topk=1 linear-chain verify inputs in-JIT without stacking shardings.
- `_build_draft_extend(num_layers: int, topk: int)` — [`L458`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L458) — Build the fused JIT. Called once, result cached on draft_worker.
- `_build_prefill(num_layers: int, topk: int)` — [`L1015`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1015) — Build prefill JIT: target extend + all MTP draft-extend layers.
- `_build_verify(topk: int)` — [`L754`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L754) — Build target verify JIT for greedy NEXTN decode.
- `_count_pjit_cpp_cache_miss()` — [`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L58)
- `_gather_rows_preserve_sharding(values, index)` — [`L442`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L442)
- `_make_draft_extend_metadata(old_metadata, draft_seq_lens, allocated_lens, *, page_size: int, dp_size: int)` — [`L700`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L700)
- `_make_forward_batch(batch, model_runner)` — [`L1268`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1268)
- `_make_target_verify_metadata(old_metadata, verify_seq_lens, allocated_lens, *, speculative_num_draft_tokens: int, page_size: int, dp_size: int)` — [`L642`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L642)
- `_per_dp_cumsum_device(lens, dp_size: int)` — [`L582`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L582)
- `_prepare_device_array(value, sharding, name: str | None = None)` — [`L1220`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1220)
- `_prepare_draft_inputs(hidden_states, positions, seq_lens, accept_index, accept_length, verified_id, *, speculative_num_steps, speculative_num_draft_tokens)` — [`L131`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L131)
- `_prepare_logits_metadata(batch, mesh, *, include_accept_lens: bool = True)` — [`L1232`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1232)
- `_prepare_rejection_sampling(sampling_info, batch, total_bs: int, vocab_size: int)` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L81)
- `_prepare_spec_prefill_output_token_ids(draft_worker, next_token_ids)` — [`L112`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L112)
- `_prepare_verify(draft_worker, model_worker_batch)` — [`L1152`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1152) — Prepare fixed-shape verify placeholders while keeping chain build inside JIT.
- `_rep(x)` — [`L275`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L275)
- `_repack_page_indices(page_indices, allocated_lens, metadata_seq_lens, *, page_size: int, dp_size: int)` — [`L591`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L591)
- `_reshard_values(sharding, *values)` — [`L449`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L449)
- `_rotate_input_ids(input_ids, ext_lens, sel_pos, new_tokens)` — [`L402`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L402) — Mirror MultiLayerDraftWorker._rotate_ids on device for topk=1.
- `_rotate_prefill_input_ids(input_ids, extend_seq_lens, verified_id, dp_size, per_dp_bs)` — [`L417`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L417)
- `_take_with_index_sharding(values, index)` — [`L124`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L124)
- `_topk1_index_from_logits(logits)` — [`L453`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L453)
- `_verify_greedy(*, target_hidden, positions, seq_lens, draft_tokens, target_predict, speculative_num_steps, speculative_num_draft_tokens)` — [`L174`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L174)
- `_verify_rejection_sampling(*, target_hidden, positions, seq_lens, draft_tokens, target_logits, temperatures, top_ks, top_ps, coins, coin_f, threshold_single, threshold_acc, enable_top_k, enable_top_p, speculative_num_steps, speculative_num_draft_tokens)` — [`L238`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L238) — Non-greedy counterpart of the greedy chain verify.
- `draft_extend_for_decode(draft_worker, model_worker_batch, batch_output)` — [`L1563`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1563) — Drop-in replacement for MultiLayerDraftWorker.draft_extend_for_decode.
- `fused_draft_extend(model_def, model_state_def, all_leaves, forward_batch, all_memory_pools, logits_metadata, target_hidden, sel_pos, draft_logits_indices, relay_buffers, relay_future_indices, relay_valid_mask, relay_verified_id, relay_new_seq_lens, draft_verify_seq_lens, draft_allocate_lens, *, num_layers, update_relay, dp_size)` — [`L467`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L467)
- `fused_prefill(target_model_def, target_model_state_def, target_leaves, target_forward_batch, target_memory_pools, target_logits_metadata, draft_model_def, draft_model_state_def, draft_all_leaves, draft_forward_batch, draft_logits_indices, all_memory_pools, draft_logits_metadata, relay_buffers, relay_future_indices, relay_valid_mask, *, num_layers, dp_size, per_dp_bs, update_relay)` — [`L1031`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1031)
- `fused_verify(target_model_def, target_model_state_def, target_leaves, target_forward_batch, target_memory_pools, target_logits_metadata, previous_verified_id, previous_token_list, relay_buffers, relay_future_indices, verify_allocate_lens, sampling_base_rng, sampling_step, temperatures, top_ks, top_ps, *, speculative_num_steps, speculative_num_draft_tokens, return_target_logits, use_relay_state, dp_size, is_greedy=True, threshold_single=1, threshold_acc=1, enable_top_k=False, enable_top_p=False)` — [`L775`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L775)
- `launch_fused_draft_extend_for_decode(draft_worker, model_worker_batch, batch_output, *, relay_buffers=None, relay_future_indices=None, relay_valid_mask=None)` — [`L1377`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1377) — Launch fused MTP draft extend and return deferred host restore state.
- `prepare_forward_batch_for_prefill(spec_worker, model_worker_batch)` — [`L1363`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1363) — Prepare the target ForwardBatch before speculative prefill is queued.
- `restore_draft_extend_result(draft_worker, model_worker_batch, pending_result)` — [`L1533`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1533)
- `rotate_rank(ids_rank, ext_rank, verified_rank)` — [`L424`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L424)
- `spec_decode(spec_worker, model_worker_batch, cur_allocate_lens)` — [`L1939`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1939) — Run speculative decode as verify JIT followed by draft-extend JIT.
- `spec_decode_draft_extend(spec_worker, model_worker_batch, batch_output)` — [`L1933`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1933) — Run MTP draft extend as the second speculative decode JIT.
- `spec_decode_overlap(spec_worker, model_worker_batch, cur_allocate_lens)` — [`L1945`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1945) — Launch decode verify and draft-extend without restoring draft results inline.
- `spec_decode_verify(spec_worker, model_worker_batch, cur_allocate_lens)` — [`L1761`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1761) — Run target verify as the first speculative decode JIT.
- `spec_prefill(spec_worker, model_worker_batch, launch_done=None, *, update_relay=False)` — [`L1574`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1574) — Run greedy prefill target forward and MTP draft-extend in one JIT. — documented in [python-sgl_jax-srt-model_executor-forward_batch_info](../../../../../concepts/python-sgl_jax-srt-model_executor-forward_batch_info.md)
- `spec_prefill_overlap(spec_worker, model_worker_batch)` — [`L1757`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/speculative/draft_extend_fused.py#L1757)

