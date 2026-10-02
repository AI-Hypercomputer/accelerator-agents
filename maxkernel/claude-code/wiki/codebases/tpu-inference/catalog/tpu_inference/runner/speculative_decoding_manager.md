---
title: 'Module: tpu_inference/runner/speculative_decoding_manager.py'
type: catalog
provenance: extracted
module: tpu_inference/runner/speculative_decoding_manager.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.runner.speculative_decoding_manager`/SpeculativeDecodingManager#
symbols:
  SpeculativeDecodingManager.propose_eagle3_draft_token_ids: propose_eagle3_draft_token_ids().
  SpeculativeDecodingManager.propose_draft_token_ids: propose_draft_token_ids().
  SpeculativeDecodingManager.get_spec_decode_metadata: get_spec_decode_metadata().
  SpeculativeDecodingManager.runner: runner.
  SpeculativeDecodingManager.take_draft_token_ids: take_draft_token_ids().
  SpeculativeDecodingManager._draft_token_ids: _draft_token_ids.
  SpeculativeDecodingManager._req_indices_dp: _req_indices_dp.
  SpeculativeDecodingManager: ''
  SpeculativeDecodingManager._pad: _pad().
  SpeculativeDecodingManager.__init__: __init__().
---
# Module: [`tpu_inference/runner/speculative_decoding_manager.py`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/speculative_decoding_manager.py)

## Classes
### `SpeculativeDecodingManager`
- def: [`tpu_inference/runner/speculative_decoding_manager.py:38`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/speculative_decoding_manager.py#L38)
- signature: `class SpeculativeDecodingManager:`
- members:
  - `get_spec_decode_metadata(self, num_draft_tokens_dp: np.ndarray, dp_size: int, req_indices_dp: dict, req_ids_dp: dict, query_start_loc: np.ndarray, padded_num_reqs_per_dp_rank: int, padded_logits_length_dp_rank: int, max_num_reqs_per_dp_rank: int)` — [`L216`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/speculative_decoding_manager.py#L216)
  - `propose_draft_token_ids(self, sampled_output: jnp.ndarray, logits_indices_selector: np.ndarray, last_sampled_token_id: jnp.ndarray, num_rejected_tokens: jnp.ndarray, discard_sampled_tokens_req_indices: list, aux_hidden_states: Optional[tuple[jnp.ndarray, ...]], attn_metadata: AttentionMetadata, async_scheduling: bool, spec_decode_metadata: SpecDecodeMetadata, scheduler_output: Optional[VllmSchedulerOutput] = None, input_ids: Optional[jnp.ndarray] = None, hidden_states: Optional[jnp.ndarray] = None)` — [`L69`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/speculative_decoding_manager.py#L69) — documented in [root](../../../concepts/root.md)
  - `propose_eagle3_draft_token_ids(self, spec_decode_metadata: SpecDecodeMetadata, last_sampled_token_id: jnp.ndarray, num_rejected_tokens: jnp.ndarray, discard_sampled_tokens_req_indices: list[int], aux_hidden_states: Optional[tuple[jnp.ndarray, ...]], attn_metadata: AttentionMetadata | dict[str, AttentionMetadata], scheduler_output: VllmSchedulerOutput, input_ids: jnp.ndarray, async_scheduling: bool, hidden_states: jnp.ndarray)` — [`L121`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/speculative_decoding_manager.py#L121)
  - `take_draft_token_ids(self)` — [`L46`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/speculative_decoding_manager.py#L46)
  - `runner` — [`L41`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/speculative_decoding_manager.py#L41)
- protocol/private: `__init__`[`L40`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/speculative_decoding_manager.py#L40), `_draft_token_ids`[`L43`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/speculative_decoding_manager.py#L43), `_pad`[`L244`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/speculative_decoding_manager.py#L244), `_req_indices_dp`[`L44`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/speculative_decoding_manager.py#L44)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../layers/common/sharding.md#ShardingAxisName), `mesh`, `input_batch`, [`AttentionMetadata`](../layers/common/attention_metadata.md#AttentionMetadata), `max_num_reqs`, `dp_size`, `speculative_config`, `kv_caches`, [`num_reqs`](input_batch.md#InputBatch.num_reqs), [`req_ids`](input_batch.md#InputBatch.req_ids), `drafter`, [`SpecDecodeMetadata`](../utils.md#SpecDecodeMetadata), [`num_tokens_no_spec`](input_batch.md#InputBatch.num_tokens_no_spec), [`seq_lens`](../layers/common/attention_metadata.md#AttentionMetadata.seq_lens), [`prepare_inputs`](../spec_decode/jax/eagle3.md#Eagle3Proposer.prepare_inputs), [`token_ids_cpu`](input_batch.md#InputBatch.token_ids_cpu), [`propose`](../spec_decode/jax/eagle3.md#Eagle3Proposer.propose), `requests`, [`host_extract_sampled_tokens`](../utils.md#host_extract_sampled_tokens), `TPUModelRunner`, `arange_cpu`, [`final_logits_indices`](../utils.md#SpecDecodeMetadata.final_logits_indices), [`draft_lengths`](../utils.md#SpecDecodeMetadata.draft_lengths), [`target_logits_indices`](../utils.md#SpecDecodeMetadata.target_logits_indices), [`Eagle3Proposer`](../spec_decode/jax/eagle3.md#Eagle3Proposer), [`get_token_id`](input_batch.md#CachedRequestState.get_token_id), [`bonus_logits_indices`](../utils.md#SpecDecodeMetadata.bonus_logits_indices), [`draft_lengths_cpu`](../utils.md#SpecDecodeMetadata.draft_lengths_cpu), [`req_indices_dp`](../utils.md#SpecDecodeMetadata.req_indices_dp), [`req_ids_dp`](../utils.md#SpecDecodeMetadata.req_ids_dp)
- used by: `_sample_from_logits`, `_prepare_inputs`, `speculative_decoding_manager`, `take_draft_token_ids`

