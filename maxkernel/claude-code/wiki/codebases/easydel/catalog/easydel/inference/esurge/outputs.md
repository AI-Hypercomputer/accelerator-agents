---
title: 'Module: easydel/inference/esurge/outputs.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/outputs.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.outputs`/
symbols:
  ModelRunnerOutput: ModelRunnerOutput#
  LogprobsLists.slice: LogprobsLists#slice().
  LogprobsTensors.tolists: LogprobsTensors#tolists().
  LogprobsTensors.empty: LogprobsTensors#empty().
  swap_dict_values: swap_dict_values().
  LogprobsLists: LogprobsLists#
  LogprobsTensors: LogprobsTensors#
  ModelRunnerOutput.prompt_logprobs_dict: ModelRunnerOutput#prompt_logprobs_dict.
  ModelRunnerOutput.logprobs: ModelRunnerOutput#logprobs.
  ModelRunnerOutput.sampled_token_ids: ModelRunnerOutput#sampled_token_ids.
  ModelRunnerOutput.req_id_to_index: ModelRunnerOutput#req_id_to_index.
  ModelRunnerOutput.req_id_to_row_index: ModelRunnerOutput#req_id_to_row_index.
  ModelRunnerOutput.token_logprobs: ModelRunnerOutput#token_logprobs.
  _K: _K.
  ModelRunnerOutput.req_ids: ModelRunnerOutput#req_ids.
  ModelRunnerOutput.spec_token_ids: ModelRunnerOutput#spec_token_ids.
  ModelRunnerOutput.finished_sending: ModelRunnerOutput#finished_sending.
  ModelRunnerOutput.finished_recving: ModelRunnerOutput#finished_recving.
  LogprobsTensors.logprob_token_ids: LogprobsTensors#logprob_token_ids.
  LogprobsTensors.logprobs: LogprobsTensors#logprobs.
  LogprobsTensors.selected_token_ranks: LogprobsTensors#selected_token_ranks.
  ModelRunnerOutput.num_nans_in_logits: ModelRunnerOutput#num_nans_in_logits.
  _V: _V.
  LogprobsLists.logprob_token_ids: LogprobsLists#logprob_token_ids.
  LogprobsLists.logprobs: LogprobsLists#logprobs.
  LogprobsLists.sampled_token_ranks: LogprobsLists#sampled_token_ranks.
---
# Module: [`easydel/inference/esurge/outputs.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py)

## Classes
### `LogprobsLists`  ·  implements/extends NamedTuple
- def: [`easydel/inference/esurge/outputs.py:59`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L59)
- doc: Log probability data in Python list format.
- signature: `class LogprobsLists(NamedTuple):`
- members:
  - `slice(self, start: int, end: int)` — [`L92`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L92) — Extract a slice of logprobs data.
  - `logprob_token_ids` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L83) — ---
  - `logprobs` — [`L86`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L86) — ---
  - `sampled_token_ranks` — [`L89`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L89) — ---
- used by: [`tolists`](outputs.md#LogprobsTensors.tolists), [`logprobs`](outputs.md#ModelRunnerOutput.logprobs), [`new_logprobs`](engine_types.md#EngineCoreOutput.new_logprobs)

### `LogprobsTensors`  ·  implements/extends NamedTuple
- def: [`easydel/inference/esurge/outputs.py:118`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L118)
- doc: Log probability data in JAX array format.
- signature: `class LogprobsTensors(NamedTuple):`
- members:
  - `empty(num_positions: int, num_tokens_per_position: int)` — [`L172`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L172) — Create an empty LogprobsTensors with the specified shape.
  - `tolists(self)` — [`L152`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L152) — Convert tensor format to list format.
  - `logprob_token_ids` — [`L143`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L143) — ---
  - `logprobs` — [`L146`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L146) — ---
  - `selected_token_ranks` — [`L149`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L149) — ---
- uses (calls/refs, reference-scoped): [`LogprobsLists`](outputs.md#LogprobsLists)
- used by: [`prompt_logprobs_dict`](outputs.md#ModelRunnerOutput.prompt_logprobs_dict), [`in_progress_prompt_logprobs_cpu`](runners/sequence_buffer.md#SequenceBuffer.in_progress_prompt_logprobs_cpu), [`new_prompt_logprobs_tensors`](engine_types.md#EngineCoreOutput.new_prompt_logprobs_tensors)

### `ModelRunnerOutput`
- def: [`easydel/inference/esurge/outputs.py:199`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L199)
- doc: Complete output from a model runner step.
- signature: `class ModelRunnerOutput:`
- members:
  - `finished_recving` — [`L260`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L260) — ---
  - `finished_sending` — [`L257`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L257) — ---
  - `logprobs` — [`L248`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L248) — ---
  - `num_nans_in_logits` — [`L263`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L263) — ---
  - `prompt_logprobs_dict` — [`L251`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L251) — ---
  - `req_id_to_index` — [`L239`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L239) — ---
  - `req_id_to_row_index` — [`L254`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L254) — ---
  - `req_ids` — [`L236`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L236) — ---
  - `sampled_token_ids` — [`L242`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L242) — ---
  - `spec_token_ids` — [`L245`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L245) — ---
  - `token_logprobs` — [`L266`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L266) — ---
- uses (calls/refs, reference-scoped): [`LogprobsLists`](outputs.md#LogprobsLists), [`LogprobsTensors`](outputs.md#LogprobsTensors)
- used by: [`_execute_model_impl`](runners/model_runner.md#eSurgeRunner._execute_model_impl), [`update_from_output`](scheduler/scheduler.md#Scheduler.update_from_output), [`get_output`](runners/model_runner.md#_AsyncExecutionHandle.get_output), [`update_from_output`](scheduler/interface.md#SchedulerInterface.update_from_output), [`execute_model`](runners/model_runner.md#eSurgeRunner.execute_model), [`wait_for_execution`](runners/model_runner.md#eSurgeRunner.wait_for_execution), [`_resolved_output`](runners/model_runner.md#_AsyncExecutionHandle._resolved_output), [`__init__`](runners/model_runner.md#_AsyncExecutionHandle.__init__)

## Functions
- `swap_dict_values(obj: dict[_K, _V], key1: _K, key2: _K)` — [`L270`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L270) — Swap values for two keys in a dictionary.

## Module values
- `_K` — [`L55`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L55)
- `_V` — [`L56`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/outputs.py#L56)

