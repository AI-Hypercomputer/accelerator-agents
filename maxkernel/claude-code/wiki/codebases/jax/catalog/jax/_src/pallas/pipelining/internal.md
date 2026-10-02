---
title: 'Module: jax/_src/pallas/pipelining/internal.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/pipelining/internal.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.pipelining.internal`/
symbols:
  PipelineStage: PipelineStage#
  PipelineStage.effects: PipelineStage#effects.
  PipelineStage.get_read_idxs: PipelineStage#get_read_idxs().
  PipelineStage.get_write_idxs: PipelineStage#get_write_idxs().
  PipelineStage.properties: PipelineStage#properties.
  RefEffect: RefEffect.
  filter_write_effects: filter_write_effects().
  filter_tokens: filter_tokens().
  filter_read_effects: filter_read_effects().
  PipelineStage.jaxpr: PipelineStage#jaxpr.
  SchedulingProperties.is_async_start: SchedulingProperties#is_async_start.
  SchedulingProperties.is_async_done: SchedulingProperties#is_async_done.
  ReadEffect: ReadEffect.
  WriteEffect: WriteEffect.
  BufferIndex: BufferIndex.
  SchedulingProperties.max_in_flight: SchedulingProperties#max_in_flight.
  SchedulingProperties.__post_init__: SchedulingProperties#__post_init__().
  PipelineStage.name: PipelineStage#name.
  PipelineStage.__repr__: PipelineStage#__repr__().
  NDLoopStruct.grid: NDLoopStruct#grid.
  NDLoopStruct.stages: NDLoopStruct#stages.
  SchedulingProperties: SchedulingProperties#
  NDLoopStruct: NDLoopStruct#
  PipelineStage.__str__: PipelineStage#__str__().
  make_token: make_token().
---
# Module: [`jax/_src/pallas/pipelining/internal.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py)

## Classes
### `NDLoopStruct`
- def: [`jax/_src/pallas/pipelining/internal.py:82`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L82)
- signature: `class NDLoopStruct:`
- members:
  - `grid` — [`L84`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L84)
  - `stages` — [`L83`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L83)
- uses (calls/refs, reference-scoped): [`PipelineStage`](internal.md#PipelineStage)
- used by: [`static_nd_loop_scheduler`](schedulers.md#static_nd_loop_scheduler), [`schedule_pipeline`](schedule_api.md#schedule_pipeline), [`PipelineScheduler`](schedulers.md#PipelineScheduler)

### `PipelineStage`
- def: [`jax/_src/pallas/pipelining/internal.py:53`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L53)
- doc: An internal representation of a pipeline stage.
- signature: `class PipelineStage:`
- members:
  - `get_read_idxs(self)` — [`L60`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L60) — Returns the buffer indices that this stage reads from.
  - `get_write_idxs(self)` — [`L67`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L67) — Returns the buffer indices that this stage writes to.
  - `effects` — [`L56`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L56)
  - `jaxpr` — [`L55`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L55)
  - `name` — [`L58`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L58)
  - `properties` — [`L57`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L57)
- protocol/private: `__repr__`[`L77`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L77), `__str__`[`L74`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L74)
- uses (calls/refs, reference-scoped): [`ClosedJaxpr`](../../core.md#ClosedJaxpr), `input`, [`RefEffect`](internal.md#RefEffect), [`filter_write_effects`](internal.md#filter_write_effects), [`filter_read_effects`](internal.md#filter_read_effects), [`BufferIndex`](internal.md#BufferIndex), [`SchedulingProperties`](internal.md#SchedulingProperties)
- used by: [`check_async_done`](schedulers.md#check_async_done), [`trace`](schedule_api.md#AsyncStage.trace), [`make_ctx`](schedulers.md#make_ctx), [`check_async_start`](schedulers.md#check_async_start), [`trace`](schedule_api.md#SyncStage.trace), [`can_run_stage`](schedulers.md#static_nd_loop_scheduler.can_run_stage), [`remove_duplicate_writes_between_async_stages`](schedule_api.md#remove_duplicate_writes_between_async_stages), [`check_args_ready`](schedulers.md#check_args_ready), [`thread_token_deps_to_consumers`](schedule_api.md#thread_token_deps_to_consumers), [`apply_ref_filter`](schedule_api.md#apply_ref_filter), [`eval_stage`](schedulers.md#eval_stage), [`convert_accum_effects_to_writes`](schedule_api.md#convert_accum_effects_to_writes), [`print_stage`](pipeline_test_util.md#print_stage), [`check_pipeline`](schedulers.md#check_pipeline), [`create`](schedulers.md#Scoreboard.create), [`linearize_stages`](schedulers.md#linearize_stages), [`__call__`](schedulers.md#EvalStageFunc.__call__), [`stages`](internal.md#NDLoopStruct.stages)

### `SchedulingProperties`
- def: [`jax/_src/pallas/pipelining/internal.py:41`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L41)
- signature: `class SchedulingProperties:`
- members:
  - `is_async_done` — [`L44`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L44)
  - `is_async_start` — [`L43`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L43)
  - `max_in_flight` — [`L42`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L42)
- protocol/private: `__post_init__`[`L46`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L46)
- used by: [`check_async_done`](schedulers.md#check_async_done), [`trace`](schedule_api.md#AsyncStage.trace), [`check_async_start`](schedulers.md#check_async_start), [`trace`](schedule_api.md#SyncStage.trace), [`can_run_stage`](schedulers.md#static_nd_loop_scheduler.can_run_stage), [`remove_duplicate_writes_between_async_stages`](schedule_api.md#remove_duplicate_writes_between_async_stages), [`check_args_ready`](schedulers.md#check_args_ready), [`thread_token_deps_to_consumers`](schedule_api.md#thread_token_deps_to_consumers), [`properties`](internal.md#PipelineStage.properties)

## Functions
- `filter_read_effects(effects: set[RefEffect])` — [`L32`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L32)
- `filter_tokens(effects: set[RefEffect])` — [`L36`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L36)
- `filter_write_effects(effects: set[RefEffect])` — [`L28`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L28)
- `make_token(obj: Hashable)` — [`L87`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L87) — Returns a fake input ID used to thread data dependencies.

## Module values
- `BufferIndex` — [`L25`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L25)
- `ReadEffect` — [`L22`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L22)
- `RefEffect` — [`L24`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L24)
- `WriteEffect` — [`L23`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/internal.py#L23)

