---
title: 'Module: easydel/inference/esurge/runners/async_types.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/runners/async_types.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.runners.async_types`/Async
symbols:
  AsyncPreResults: PreResults#
  AsyncWindowResult: WindowResult#
  AsyncPreResults.windows: PreResults#windows.
  AsyncWindowResult.row_positions: WindowResult#row_positions.
  AsyncWindowResult.sampled_token_ids: WindowResult#sampled_token_ids.
  AsyncWindowResult.valid_mask: WindowResult#valid_mask.
  AsyncWindowResult.token_logprobs: WindowResult#token_logprobs.
  AsyncWindowResult.req_ids: WindowResult#req_ids.
  AsyncPreResults.request_seq_lens: PreResults#request_seq_lens.
---
# Module: [`easydel/inference/esurge/runners/async_types.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/async_types.py)

## Classes
### `AsyncPreResults`
- def: [`easydel/inference/esurge/runners/async_types.py:52`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/async_types.py#L52)
- doc: Stores previous iteration's sampled-token payloads for async scheduling.
- signature: `class AsyncPreResults:`
- members:
  - `request_seq_lens` — [`L56`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/async_types.py#L56)
  - `windows` — [`L55`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/async_types.py#L55)
- uses (calls/refs, reference-scoped): [`CachedRequestState`](states.md#CachedRequestState), [`AsyncWindowResult`](async_types.md#AsyncWindowResult)
- used by: [`_execute_model_impl`](model_runner.md#eSurgeRunner._execute_model_impl), [`_modify_prev_results`](model_runner.md#eSurgeRunner._modify_prev_results), [`_pre_async_results`](model_runner.md#eSurgeRunner._pre_async_results)

### `AsyncWindowResult`
- def: [`easydel/inference/esurge/runners/async_types.py:31`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/async_types.py#L31)
- doc: Host-copy payload for one runner window.
- signature: `class AsyncWindowResult:`
- members:
  - `req_ids` — [`L44`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/async_types.py#L44)
  - `row_positions` — [`L45`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/async_types.py#L45)
  - `sampled_token_ids` — [`L46`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/async_types.py#L46)
  - `token_logprobs` — [`L48`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/async_types.py#L48)
  - `valid_mask` — [`L47`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/async_types.py#L47)
- used by: [`_execute_model_impl`](model_runner.md#eSurgeRunner._execute_model_impl), [`_modify_prev_results`](model_runner.md#eSurgeRunner._modify_prev_results), [`get_output`](model_runner.md#_AsyncExecutionHandle.get_output), [`__init__`](model_runner.md#_AsyncExecutionHandle.__init__), [`windows`](async_types.md#AsyncPreResults.windows)

