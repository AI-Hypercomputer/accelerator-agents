---
title: 'Module: easydel/inference/esurge/scheduler/async_scheduler.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/scheduler/async_scheduler.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.scheduler.async_scheduler`/
symbols:
  AsyncScheduler._update_after_schedule: AsyncScheduler#_update_after_schedule().
  AsyncScheduler._update_request_with_output: AsyncScheduler#_update_request_with_output().
  AsyncScheduler: AsyncScheduler#
  logger: logger.
---
# Module: [`easydel/inference/esurge/scheduler/async_scheduler.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/async_scheduler.py)

## Classes
### `AsyncScheduler`  ·  implements/extends Scheduler
- def: [`easydel/inference/esurge/scheduler/async_scheduler.py:68`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/async_scheduler.py#L68)
- doc: Asynchronous scheduler with placeholder-based token sampling.
- signature: `class AsyncScheduler(Scheduler):`
- members:
  - `_update_after_schedule(self, scheduler_output: SchedulerOutput)` — [`L111`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/async_scheduler.py#L111) — Update request states after scheduling, adding async placeholders.
  - `_update_request_with_output(self, request: EngineRequest, new_token_ids: list[int])` — [`L175`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/async_scheduler.py#L175) — Update request with newly generated tokens and manage placeholders.
- uses (calls/refs, reference-scoped): [`EngineRequest`](../request.md#EngineRequest), [`request_id`](../request.md#EngineRequest.request_id), [`EngineRequestStatus`](../request.md#EngineRequestStatus), [`kv_cache_manager`](scheduler.md#Scheduler.kv_cache_manager), [`SchedulerOutput`](output.md#SchedulerOutput), [`status`](../request.md#EngineRequest.status), [`num_computed_tokens`](../request.md#EngineRequest.num_computed_tokens), [`_update_after_schedule`](scheduler.md#Scheduler._update_after_schedule), [`Scheduler`](scheduler.md#Scheduler), [`cache_pages`](../core/manager.md#CacheManager.cache_pages), [`_update_request_with_output`](scheduler.md#Scheduler._update_request_with_output), [`requests`](scheduler.md#Scheduler.requests), [`num_scheduled_tokens`](output.md#SchedulerOutput.num_scheduled_tokens), [`num_tokens`](../request.md#EngineRequest.num_tokens), [`num_output_placeholders`](../request.md#EngineRequest.num_output_placeholders), [`use_structured_output`](../request.md#EngineRequest.use_structured_output), [`RUNNING`](../request.md#EngineRequestStatus.RUNNING), [`spec_token_ids`](../request.md#EngineRequest.spec_token_ids), [`num_spec_tokens`](scheduler.md#Scheduler.num_spec_tokens), [`scheduled_spec_decode_tokens`](output.md#SchedulerOutput.scheduled_spec_decode_tokens)
- used by: [`from_runner`](scheduler.md#Scheduler.from_runner), [`_update_after_schedule`](scheduler.md#Scheduler._update_after_schedule), [`Scheduler`](scheduler.md#Scheduler), [`_update_request_with_output`](scheduler.md#Scheduler._update_request_with_output)

## Module values
- `logger` — [`L65`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/async_scheduler.py#L65)

