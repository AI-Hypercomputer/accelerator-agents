---
title: 'Module: jax/_src/pallas/pipelining/schedulers.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/pipelining/schedulers.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.pipelining.schedulers`/
symbols:
  static_nd_loop_scheduler: static_nd_loop_scheduler().
  check_async_done: check_async_done().
  make_ctx: make_ctx().
  check_async_start: check_async_start().
  Scoreboard.stage_counters: Scoreboard#stage_counters.
  static_nd_loop_scheduler.can_run_stage: static_nd_loop_scheduler().can_run_stage().
  static_nd_loop_scheduler.loop_body: static_nd_loop_scheduler().loop_body().
  check_args_ready: check_args_ready().
  eval_stage: eval_stage().
  PipelineContext.aval_pytree: PipelineContext#aval_pytree().
  GridCarry.next: GridCarry#next().
  Scoreboard.copy: Scoreboard#copy().
  Scoreboard: Scoreboard#
  static_nd_loop_scheduler.compute_offsets: static_nd_loop_scheduler().compute_offsets().
  increment_grid: increment_grid().
  PipelineContext: PipelineContext#
  check_pipeline: check_pipeline().
  Scoreboard.create: Scoreboard#create().
  GridCarry: GridCarry#
  GridCarry.get_indices_for_stage: GridCarry#get_indices_for_stage().
  linearize_stages: linearize_stages().
  Scoreboard.get_stage_counter: Scoreboard#get_stage_counter().
  EvalStageFunc.__call__: EvalStageFunc#__call__().
  PipelineState: PipelineState.
  Scoreboard.get_writing_stage: Scoreboard#get_writing_stage().
  PipelineContext.pipeline_state: PipelineContext#pipeline_state.
  Scoreboard.increment_stage_counter: Scoreboard#increment_stage_counter().
  compute_grid_indices: compute_grid_indices().
  GridCarry.init: GridCarry#init().
  PipelineContext.linearized_index: PipelineContext#linearized_index.
  Scoreboard.which_stage_writes: Scoreboard#which_stage_writes.
  Scoreboard.which_stages_read: Scoreboard#which_stages_read.
  GridCarry.indices: GridCarry#indices.
  PipelineContext.loop_index: PipelineContext#loop_index.
  PipelineScheduler: PipelineScheduler.
  GridCarry.grid: GridCarry#grid.
  GridCarry.offsets: GridCarry#offsets.
  GridCarry.dynamic: GridCarry#dynamic.
  EvalStageFunc: EvalStageFunc#
---
# Module: [`jax/_src/pallas/pipelining/schedulers.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py)

## Classes
### `EvalStageFunc`  ·  implements/extends Protocol
- def: [`jax/_src/pallas/pipelining/schedulers.py:302`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L302)
- signature: `class EvalStageFunc(Protocol):`
- protocol/private: `__call__`[`L303`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L303)
- uses (calls/refs, reference-scoped): [`PipelineStage`](internal.md#PipelineStage), [`PipelineContext`](schedulers.md#PipelineContext), [`PipelineState`](schedulers.md#PipelineState)
- used by: [`static_nd_loop_scheduler`](schedulers.md#static_nd_loop_scheduler)

### `GridCarry`
- def: [`jax/_src/pallas/pipelining/schedulers.py:179`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L179)
- doc: Helper class for managing the pipeline grid indices.
- signature: `class GridCarry:`
- members:
  - `get_indices_for_stage(self, stage_idx: int)` — [`L212`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L212)
  - `init(cls, grid, offsets, dynamic=False)` — [`L195`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L195)
  - `next(self)` — [`L205`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L205)
  - `dynamic` — [`L191`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L191)
  - `grid` — [`L189`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L189)
  - `indices` — [`L192`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L192)
  - `offsets` — [`L190`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L190)
- uses (calls/refs, reference-scoped): [`Array`](../../basearray.md#Array), [`register_dataclass`](../../tree_util.md#register_dataclass), [`increment_grid`](schedulers.md#increment_grid)
- used by: [`static_nd_loop_scheduler`](schedulers.md#static_nd_loop_scheduler), [`make_ctx`](schedulers.md#make_ctx), [`loop_body`](schedulers.md#static_nd_loop_scheduler.loop_body)

### `PipelineContext`
- def: [`jax/_src/pallas/pipelining/schedulers.py:80`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L80)
- doc: Container class containing pipeline state information.
- signature: `class PipelineContext:`
- members:
  - `aval_pytree(cls, grid, state_avals)` — [`L93`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L93)
  - `linearized_index` — [`L89`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L89)
  - `loop_index` — [`L88`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L88)
  - `pipeline_state` — [`L90`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L90)
- uses (calls/refs, reference-scoped): [`Array`](../../basearray.md#Array), [`ShapedArray`](../../core.md#ShapedArray), [`int32`](../../../numpy/__init__.pyi.md#int32.int32), [`register_dataclass`](../../tree_util.md#register_dataclass), [`PipelineState`](schedulers.md#PipelineState)
- used by: [`make_ctx`](schedulers.md#make_ctx), [`trace_fun`](schedule_api.md#trace_fun), [`apply_ref_filter`](schedule_api.md#apply_ref_filter), [`eval_stage`](schedulers.md#eval_stage), [`print_stage`](pipeline_test_util.md#print_stage), [`__call__`](schedulers.md#EvalStageFunc.__call__), [`PipelineContext`](schedule_api.md#PipelineContext)

### `Scoreboard`
- def: [`jax/_src/pallas/pipelining/schedulers.py:129`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L129)
- doc: A scoreboard used to book-keep data dependencies.
- signature: `class Scoreboard:`
- members:
  - `copy(self)` — [`L168`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L168) — Returns a deep copy of the scoreboard.
  - `create(cls, stages: Sequence[internal.PipelineStage])` — [`L145`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L145)
  - `get_stage_counter(self, stage_idx: int)` — [`L156`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L156) — Returns the current stage counter for the given stage index.
  - `get_writing_stage(self, buffer_idx: internal.BufferIndex)` — [`L160`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L160) — Returns the stage index that writes to the given buffer index.
  - `increment_stage_counter(self, stage_idx: int)` — [`L164`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L164) — Increments the stage counter for the given stage index.
  - `stage_counters` — [`L142`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L142)
  - `which_stage_writes` — [`L140`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L140)
  - `which_stages_read` — [`L141`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L141)
- uses (calls/refs, reference-scoped): [`Array`](../../basearray.md#Array), [`register_dataclass`](../../tree_util.md#register_dataclass), [`PipelineStage`](internal.md#PipelineStage), [`get_read_idxs`](internal.md#PipelineStage.get_read_idxs), [`get_write_idxs`](internal.md#PipelineStage.get_write_idxs), [`BufferIndex`](internal.md#BufferIndex)
- used by: [`static_nd_loop_scheduler`](schedulers.md#static_nd_loop_scheduler), [`check_async_done`](schedulers.md#check_async_done), [`make_ctx`](schedulers.md#make_ctx), [`check_async_start`](schedulers.md#check_async_start), [`can_run_stage`](schedulers.md#static_nd_loop_scheduler.can_run_stage), [`loop_body`](schedulers.md#static_nd_loop_scheduler.loop_body), [`check_args_ready`](schedulers.md#check_args_ready), [`compute_offsets`](schedulers.md#static_nd_loop_scheduler.compute_offsets)

## Functions
- `can_run_stage(stage: internal.PipelineStage, scoreboard: Scoreboard, new_scoreboard: Scoreboard, current_stage_counter: int | jax.Array)` — [`L390`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L390)
- `check_args_ready(stage: internal.PipelineStage, scoreboard: Scoreboard, new_scoreboard: Scoreboard, current_stage_counter: int | jax.Array, dynamic=False)` — [`L216`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L216) — Returns whether all arguments to the stage have already been computed.
- `check_async_done(stage: internal.PipelineStage, scoreboard: Scoreboard, num_itrs: int | jax.Array, current_stage_counter: int | jax.Array, dynamic=False)` — [`L240`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L240) — Returns whether the async done stage can run.
- `check_async_start(stage: internal.PipelineStage, scoreboard: Scoreboard, current_stage_counter: int | jax.Array, dynamic=False)` — [`L276`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L276) — Returns whether the async start stage can run.
- `check_pipeline(stages: Sequence[internal.PipelineStage])` — [`L100`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L100) — Runs sanity checks on the pipeline.
- `compute_grid_indices(linear_index: jax.Array, grid_size: Sequence[int])` — [`L42`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L42) — Computes the grid indices for a given linear index.
- `compute_offsets(scoreboard: Scoreboard)` — [`L410`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L410)
- `eval_stage(ctx: PipelineContext, stage: internal.PipelineStage, args)` — [`L312`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L312) — Evaluates a single stage.
- `increment_grid(indices: Sequence[int | jax.Array], grid: Sequence[int], dynamic: bool = False)` — [`L53`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L53) — Increments the grid indices by 1.
- `linearize_stages(stages: Sequence[internal.PipelineStage])` — [`L325`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L325) — Computes a linearization of the pipeline stages.
- `loop_body(itr: jax.Array, carry: tuple[PipelineState, GridCarry])` — [`L481`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L481)
- `make_ctx(stage: internal.PipelineStage, stage_idx: int, scoreboard: Scoreboard, pipeline_state: PipelineState, grid_carry: GridCarry | None = None, grid: Sequence[int] | None = None, offset: int | jax.Array = 0)` — [`L348`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L348)
- `static_nd_loop_scheduler(nd_loop: internal.NDLoopStruct, args: Sequence[Any], initial_state: PipelineState | None = None, eval_fn: EvalStageFunc | None = None)` — [`L368`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L368) — Schedules and emits the pipeline into a single instruction stream.

## Module values
- `PipelineScheduler` — [`L38`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L38)
- `PipelineState` — [`L37`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/schedulers.py#L37)

