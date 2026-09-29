---
title: 'Module: tpu_inference/runner/persistent_batch_manager.py'
type: catalog
provenance: extracted
module: tpu_inference/runner/persistent_batch_manager.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.runner.persistent_batch_manager`/
symbols:
  PersistentBatchManager.update_states: PersistentBatchManager#update_states().
  PersistentBatchManager.input_batch: PersistentBatchManager#input_batch.
  PersistentBatchManager._reorder_batch: PersistentBatchManager#_reorder_batch().
  PersistentBatchManager.requests: PersistentBatchManager#requests.
  PersistentBatchManager.__init__: PersistentBatchManager#__init__().
  PersistentBatchManager: PersistentBatchManager#
  logger: logger.
  PersistentBatchManager.is_last_rank: PersistentBatchManager#is_last_rank.
  PersistentBatchManager.encoder_cache: PersistentBatchManager#encoder_cache.
  PersistentBatchManager.uses_mrope: PersistentBatchManager#uses_mrope.
  PersistentBatchManager.model_config: PersistentBatchManager#model_config.
---
# Module: [`tpu_inference/runner/persistent_batch_manager.py`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/persistent_batch_manager.py)

## Classes
### `PersistentBatchManager`
- def: [`tpu_inference/runner/persistent_batch_manager.py:26`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/persistent_batch_manager.py#L26)
- signature: `class PersistentBatchManager:`
- members:
  - `_reorder_batch(self, scheduler_output: VllmSchedulerOutput)` — [`L39`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/persistent_batch_manager.py#L39) — Reorder the sheduled requests to RPA kernel friendly distribution
  - `update_states(self, scheduler_output: VllmSchedulerOutput, get_mrope_input_positions_fn)` — [`L88`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/persistent_batch_manager.py#L88) — Update the cached states and the persistent batch with the scheduler
  - `encoder_cache` — [`L34`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/persistent_batch_manager.py#L34)
  - `input_batch` — [`L33`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/persistent_batch_manager.py#L33)
  - `is_last_rank` — [`L37`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/persistent_batch_manager.py#L37)
  - `model_config` — [`L36`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/persistent_batch_manager.py#L36)
  - `requests` — [`L32`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/persistent_batch_manager.py#L32)
  - `uses_mrope` — [`L35`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/persistent_batch_manager.py#L35)
- protocol/private: `__init__`[`L28`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/persistent_batch_manager.py#L28)
- uses (calls/refs, reference-scoped): [`add_request`](input_batch.md#InputBatch.add_request), [`remove_request`](input_batch.md#InputBatch.remove_request), [`condense`](input_batch.md#InputBatch.condense), [`swap_states`](input_batch.md#InputBatch.swap_states), [`num_reqs`](input_batch.md#InputBatch.num_reqs), [`req_id_to_index`](input_batch.md#InputBatch.req_id_to_index), [`req_ids`](input_batch.md#InputBatch.req_ids), [`num_tokens_no_spec`](input_batch.md#InputBatch.num_tokens_no_spec), [`num_tokens`](input_batch.md#InputBatch.num_tokens), [`CachedRequestState`](input_batch.md#CachedRequestState), [`output_token_ids`](input_batch.md#CachedRequestState.output_token_ids), [`token_ids_cpu`](input_batch.md#InputBatch.token_ids_cpu), [`assert_mamba_state_invariants`](input_batch.md#InputBatch.assert_mamba_state_invariants), [`num_computed_tokens_cpu`](input_batch.md#InputBatch.num_computed_tokens_cpu), [`mamba_state_indices_cpu`](input_batch.md#InputBatch.mamba_state_indices_cpu), [`mamba_state_slot`](input_batch.md#CachedRequestState.mamba_state_slot), [`InputBatch`](input_batch.md#InputBatch), [`num_prompt_tokens`](input_batch.md#InputBatch.num_prompt_tokens), [`block_table`](input_batch.md#InputBatch.block_table), [`num_tokens`](input_batch.md#CachedRequestState.num_tokens), [`release_mamba_slot`](input_batch.md#InputBatch.release_mamba_slot), [`append_row`](block_table.md#MultiGroupBlockTable.append_row), [`generator`](input_batch.md#CachedRequestState.generator), [`request_distribution`](input_batch.md#InputBatch.request_distribution), [`has_mamba_layers`](input_batch.md#InputBatch.has_mamba_layers), [`mrope_positions`](input_batch.md#CachedRequestState.mrope_positions), [`mrope_position_delta`](input_batch.md#CachedRequestState.mrope_position_delta)
- used by: `_execute_model`, [`maybe_reinitialize_input_batch`](kv_cache_manager.md#KVCacheManager.maybe_reinitialize_input_batch), `persistent_batch_manager`

## Module values
- `logger` — [`L23`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/persistent_batch_manager.py#L23)

