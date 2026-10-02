---
title: 'Module: easydel/inference/esurge/scheduler/interface.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/scheduler/interface.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.scheduler.interface`/SchedulerInterface#
symbols:
  SchedulerInterface.update_from_output: update_from_output().
  SchedulerInterface: ''
  SchedulerInterface.schedule: schedule().
  SchedulerInterface.add_request: add_request().
  SchedulerInterface.finish_requests: finish_requests().
  SchedulerInterface.has_requests: has_requests().
  SchedulerInterface.get_num_unfinished_requests: get_num_unfinished_requests().
  SchedulerInterface.has_unfinished_requests: has_unfinished_requests().
  SchedulerInterface.has_finished_requests: has_finished_requests().
  SchedulerInterface.reset_prefix_cache: reset_prefix_cache().
  SchedulerInterface.get_request_counts: get_request_counts().
  SchedulerInterface.shutdown: shutdown().
---
# Module: [`easydel/inference/esurge/scheduler/interface.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/interface.py)

## Classes
### `SchedulerInterface`  ·  implements/extends ABC
- def: [`easydel/inference/esurge/scheduler/interface.py:53`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/interface.py#L53)
- doc: Abstract base class defining the scheduler interface.
- signature: `class SchedulerInterface(ABC):`
- members:
  - `add_request(self, request: EngineRequest)` — [`L154`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/interface.py#L154) — Add a new request to the scheduler's queue.
  - `finish_requests(self, request_ids: str | Iterable[str], finished_status: EngineRequestStatus)` — [`L173`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/interface.py#L173) — Mark requests as finished from external signal.
  - `get_num_unfinished_requests(self)` — [`L203`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/interface.py#L203) — Get the number of unfinished requests in the scheduler.
  - `get_request_counts(self)` — [`L282`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/interface.py#L282) — Get the counts of running and waiting requests.
  - `has_finished_requests(self)` — [`L228`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/interface.py#L228) — Check if there are finished requests pending notification.
  - `has_requests(self)` — [`L250`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/interface.py#L250) — Check if there are any requests to process.
  - `has_unfinished_requests(self)` — [`L214`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/interface.py#L214) — Check if there are unfinished requests in the scheduler.
  - `reset_prefix_cache(self)` — [`L263`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/interface.py#L263) — Reset the prefix cache for KV cache.
  - `schedule(self)` — [`L84`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/interface.py#L84) — Schedule requests to process in this scheduling step.
  - `shutdown(self)` — [`L294`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/interface.py#L294) — Shutdown the scheduler and release resources.
  - `update_from_output(self, scheduler_output: SchedulerOutput, model_runner_output: ModelRunnerOutput)` — [`L117`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/scheduler/interface.py#L117) — Update scheduler state based on model runner output.
- uses (calls/refs, reference-scoped): [`schedule`](scheduler.md#Scheduler.schedule), [`update_from_output`](scheduler.md#Scheduler.update_from_output), [`EngineRequest`](../request.md#EngineRequest), [`EngineRequestStatus`](../request.md#EngineRequestStatus), [`SchedulerOutput`](output.md#SchedulerOutput), [`finish_requests`](scheduler.md#Scheduler.finish_requests), [`Scheduler`](scheduler.md#Scheduler), [`ModelRunnerOutput`](../outputs.md#ModelRunnerOutput), [`add_request`](scheduler.md#Scheduler.add_request), [`EngineCoreOutputs`](../engine_types.md#EngineCoreOutputs), [`get_num_unfinished_requests`](scheduler.md#Scheduler.get_num_unfinished_requests), [`get_request_counts`](scheduler.md#Scheduler.get_request_counts), [`reset_prefix_cache`](scheduler.md#Scheduler.reset_prefix_cache), [`has_finished_requests`](scheduler.md#Scheduler.has_finished_requests), [`shutdown`](scheduler.md#Scheduler.shutdown)
- used by: [`Scheduler`](scheduler.md#Scheduler)

