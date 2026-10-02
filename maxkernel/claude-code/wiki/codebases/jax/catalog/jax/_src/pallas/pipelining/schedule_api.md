---
title: 'Module: jax/_src/pallas/pipelining/schedule_api.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/pipelining/schedule_api.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.pipelining.schedule_api`/
symbols:
  schedule_pipeline: schedule_pipeline().
  AsyncStage.trace: AsyncStage#trace().
  SyncStage.trace: SyncStage#trace().
  remove_duplicate_writes_between_async_stages: remove_duplicate_writes_between_async_stages().
  thread_token_deps_to_consumers: thread_token_deps_to_consumers().
  trace_fun: trace_fun().
  apply_ref_filter: apply_ref_filter().
  convert_accum_effects_to_writes: convert_accum_effects_to_writes().
  Stage: Stage.
  PipelineContext: PipelineContext.
  schedule_pipeline._get_aval: schedule_pipeline()._get_aval().
  AsyncStage.start_func: AsyncStage#start_func.
  AsyncStage.end_func: AsyncStage#end_func.
  SyncStage: SyncStage#
  stage.wrapper: stage().wrapper().
  SyncStage.func: SyncStage#func.
  stage: stage().
  AsyncStage.def_start: AsyncStage#def_start().
  AsyncStage.def_end: AsyncStage#def_end().
  AsyncStage: AsyncStage#
  AsyncStage.max_in_flight: AsyncStage#max_in_flight.
  SyncStage.max_in_flight: SyncStage#max_in_flight.
  schedule_pipeline.pipeline: schedule_pipeline().pipeline().
  SyncStage.__init__: SyncStage#__init__().
  AsyncStage.__init__: AsyncStage#__init__().
---
# Module: [`jax/_src/pallas/pipelining/schedule_api.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py)

## Classes
### `AsyncStage`
- def: [`jax/_src/pallas/pipelining/schedule_api.py:69`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L69)
- doc: Constructs an asynchronous pipeline stage.
- signature: `class AsyncStage:`
- members:
  - `def_end(self, func)` — [`L81`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L81)
  - `def_start(self, func)` — [`L77`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L77)
  - `trace(self, abstract_refs, state_avals, grid)` — [`L85`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L85)
  - `end_func` — [`L74`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L74)
  - `max_in_flight` — [`L75`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L75)
  - `start_func` — [`L73`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L73)
- protocol/private: `__init__`[`L72`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L72)
- uses (calls/refs, reference-scoped): [`PipelineStage`](internal.md#PipelineStage), [`effects`](internal.md#PipelineStage.effects), [`trace_fun`](schedule_api.md#trace_fun), [`properties`](internal.md#PipelineStage.properties), [`jaxpr`](internal.md#PipelineStage.jaxpr), [`is_async_done`](internal.md#SchedulingProperties.is_async_done), [`is_async_start`](internal.md#SchedulingProperties.is_async_start), [`ReadEffect`](internal.md#ReadEffect), [`WriteEffect`](internal.md#WriteEffect), [`max_in_flight`](internal.md#SchedulingProperties.max_in_flight), [`name`](internal.md#PipelineStage.name), [`SchedulingProperties`](internal.md#SchedulingProperties), [`make_token`](internal.md#make_token)
- used by: [`schedule_pipeline`](schedule_api.md#schedule_pipeline), [`Stage`](schedule_api.md#Stage)

### `SyncStage`
- def: [`jax/_src/pallas/pipelining/schedule_api.py:43`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L43)
- members:
  - `trace(self, abstract_refs, state_avals, grid)` — [`L50`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L50)
  - `func` — [`L47`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L47)
  - `max_in_flight` — [`L48`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L48)
- protocol/private: `__init__`[`L46`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L46)
- uses (calls/refs, reference-scoped): [`PipelineStage`](internal.md#PipelineStage), [`effects`](internal.md#PipelineStage.effects), [`trace_fun`](schedule_api.md#trace_fun), [`properties`](internal.md#PipelineStage.properties), [`jaxpr`](internal.md#PipelineStage.jaxpr), [`is_async_done`](internal.md#SchedulingProperties.is_async_done), [`is_async_start`](internal.md#SchedulingProperties.is_async_start), [`max_in_flight`](internal.md#SchedulingProperties.max_in_flight), [`name`](internal.md#PipelineStage.name), [`SchedulingProperties`](internal.md#SchedulingProperties)
- used by: [`schedule_pipeline`](schedule_api.md#schedule_pipeline), [`Stage`](schedule_api.md#Stage), [`wrapper`](schedule_api.md#stage.wrapper)

## Functions
- `_get_aval(x)` — [`L286`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L286)
- `apply_ref_filter(stages: Sequence[internal.PipelineStage], ref_filter: Any, grid, state_avals)` — [`L150`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L150) — Removes any effects belonging to Refs that do not pass the filter.
- `convert_accum_effects_to_writes(stages: Sequence[internal.PipelineStage])` — [`L176`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L176) — Replaces all accumulate effects with simple writes.
- `pipeline(*args)` — [`L314`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L314)
- `remove_duplicate_writes_between_async_stages(stages: Sequence[internal.PipelineStage])` — [`L195`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L195) — Removes duplicate writes between the async start and done stages.
- `schedule_pipeline(stages: Sequence[Stage], grid: Sequence[int], args: Sequence[Any], ref_filter: Any | None = None, initial_state: schedulers.PipelineState | None = None, scheduler: schedulers.PipelineScheduler = schedulers.static_nd_loop_scheduler, **scheduler_kwargs)` — [`L259`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L259) — Schedules stages and emits the code for a pipeline.
- `stage(max_in_flight: int)` — [`L36`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L36) — Wrapper for creating a pipeline stage.
- `thread_token_deps_to_consumers(stages: Sequence[internal.PipelineStage])` — [`L230`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L230) — Threads the async token to consumers of async op.
- `trace_fun(fun, ref_avals, state_avals, grid)` — [`L125`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L125) — Trace a stage body function to a Jaxpr.
- `wrapper(func)` — [`L38`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L38)

## Module values
- `PipelineContext` — [`L33`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L33)
- `Stage` — [`L122`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedule_api.py#L122)

