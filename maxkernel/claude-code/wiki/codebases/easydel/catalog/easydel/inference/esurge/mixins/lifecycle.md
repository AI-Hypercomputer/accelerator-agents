---
title: 'Module: easydel/inference/esurge/mixins/lifecycle.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/mixins/lifecycle.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.mixins.lifecycle`/
symbols:
  EngineLifecycleMixin._scheduler_loop: EngineLifecycleMixin#_scheduler_loop().
  EngineLifecycleMixin.terminate: EngineLifecycleMixin#terminate().
  EngineLifecycleMixin._paused: EngineLifecycleMixin#_paused.
  EngineLifecycleMixin.update_model_weights: EngineLifecycleMixin#update_model_weights().
  EngineLifecycleMixin.pause: EngineLifecycleMixin#pause().
  EngineLifecycleMixin.scheduler: EngineLifecycleMixin#scheduler.
  EngineLifecycleMixin._can_prefetch_scheduler_output: EngineLifecycleMixin#_can_prefetch_scheduler_output().
  EngineLifecycleMixin._resolve_graphdef_for_weight_update: EngineLifecycleMixin#_resolve_graphdef_for_weight_update().
  EngineLifecycleMixin.stop_profiling: EngineLifecycleMixin#stop_profiling().
  EngineLifecycleMixin._kv_cache_valid: EngineLifecycleMixin#_kv_cache_valid.
  EngineLifecycleMixin._scheduler_thread: EngineLifecycleMixin#_scheduler_thread.
  EngineLifecycleMixin._drain_runner_future: EngineLifecycleMixin#_drain_runner_future().
  EngineLifecycleMixin._split_graph_components_for_weight_update: EngineLifecycleMixin#_split_graph_components_for_weight_update().
  EngineLifecycleMixin._scheduler_running: EngineLifecycleMixin#_scheduler_running.
  EngineLifecycleMixin.release_model_state: EngineLifecycleMixin#release_model_state().
  EngineLifecycleMixin._handle_profiling_step: EngineLifecycleMixin#_handle_profiling_step().
  EngineLifecycleMixin._can_prefetch_next: EngineLifecycleMixin#_can_prefetch_next().
  EngineLifecycleMixin.resume: EngineLifecycleMixin#resume().
  EngineLifecycleMixin._raise_if_scheduler_failed: EngineLifecycleMixin#_raise_if_scheduler_failed().
  EngineLifecycleMixin.initiate: EngineLifecycleMixin#initiate().
  EngineLifecycleMixin._check_scheduler_heartbeat: EngineLifecycleMixin#_check_scheduler_heartbeat().
  EngineLifecycleMixin: EngineLifecycleMixin#
  EngineLifecycleMixin._model_overrides_esurge_graphdef: EngineLifecycleMixin#_model_overrides_esurge_graphdef().
  EngineLifecycleMixin._abort_scheduler_due_to_error: EngineLifecycleMixin#_abort_scheduler_due_to_error().
  EngineLifecycleMixin._scheduler_heartbeat_last_warn: EngineLifecycleMixin#_scheduler_heartbeat_last_warn.
  EngineLifecycleMixin._profiling_active: EngineLifecycleMixin#_profiling_active.
  EngineLifecycleMixin._reset_runner_state_if_idle: EngineLifecycleMixin#_reset_runner_state_if_idle().
  EngineLifecycleMixin._update_scheduler_heartbeat: EngineLifecycleMixin#_update_scheduler_heartbeat().
  EngineLifecycleMixin._profiling_steps_remaining: EngineLifecycleMixin#_profiling_steps_remaining.
  _SCHEDULER_HEARTBEAT_WARN_INTERVAL_S: _SCHEDULER_HEARTBEAT_WARN_INTERVAL_S.
  EngineLifecycleMixin._install_signal_diagnostics: EngineLifecycleMixin#_install_signal_diagnostics().
  EngineLifecycleMixin._dump_state: EngineLifecycleMixin#_dump_state().
  EngineLifecycleMixin._scheduler_exception: EngineLifecycleMixin#_scheduler_exception.
  EngineLifecycleMixin._scheduler_exception_tb: EngineLifecycleMixin#_scheduler_exception_tb.
  EngineLifecycleMixin.start_profiling: EngineLifecycleMixin#start_profiling().
  _SCHEDULER_HEARTBEAT_WARN_S: _SCHEDULER_HEARTBEAT_WARN_S.
  EngineLifecycleMixin._is_nonrecoverable_scheduler_error: EngineLifecycleMixin#_is_nonrecoverable_scheduler_error().
  EngineLifecycleMixin._profiling_output_dir: EngineLifecycleMixin#_profiling_output_dir.
  MAX_CONSECUTIVE_SCHEDULER_ERRORS: MAX_CONSECUTIVE_SCHEDULER_ERRORS.
  EngineLifecycleMixin._scheduler_heartbeat: EngineLifecycleMixin#_scheduler_heartbeat.
  EngineLifecycleMixin._profiling_host_level: EngineLifecycleMixin#_profiling_host_level.
  EngineLifecycleMixin._profiling_python_level: EngineLifecycleMixin#_profiling_python_level.
  EngineLifecycleMixin._track_finished_output: EngineLifecycleMixin#_track_finished_output().
---
# Module: [`easydel/inference/esurge/mixins/lifecycle.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py)

## Classes
### `EngineLifecycleMixin`
- def: [`easydel/inference/esurge/mixins/lifecycle.py:38`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L38)
- doc: Mixin managing the scheduler lifecycle for the eSurge engine.
- signature: `class EngineLifecycleMixin:`
- members:
  - `_abort_scheduler_due_to_error(self, exc: BaseException)` — [`L131`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L131) — Record a fatal scheduler error and wake all waiting callers.
  - `_can_prefetch_scheduler_output(scheduler: Scheduler, current: SchedulerOutput)` — [`L88`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L88) — Return whether the next schedule can be computed before current output lands.
  - `_check_scheduler_heartbeat(self)` — [`L217`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L217) — Warn if the scheduler hasn't produced a heartbeat recently.
  - `_drain_runner_future(self, future, scheduler_output: SchedulerOutput)` — [`L790`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L790) — Wait for an async runner execution and process results.
  - `_handle_profiling_step(self)` — [`L804`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L804) — Handle profiling step counter and auto-stop when complete.
  - `_install_signal_diagnostics(self)` — [`L172`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L172) — Install signal handlers that log engine state before exit.
  - `_is_nonrecoverable_scheduler_error(exc: BaseException)` — [`L58`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L58) — Classify scheduler errors that should abort immediately.
  - `_model_overrides_esurge_graphdef(model: EasyDeLBaseModule)` — [`L71`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L71) — Return whether the model delegates eSurge graph construction.
  - `_raise_if_scheduler_failed(self)` — [`L155`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L155) — Check for scheduler failure and raise if one occurred.
  - `_reset_runner_state_if_idle(self, reason: str)` — [`L700`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L700) — Reset runner buffers when there are no active/pending requests.
  - `_resolve_graphdef_for_weight_update(cls, model: EasyDeLBaseModule, split_graphdef=None)` — [`L111`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L111) — Resolve the graphdef to use for a model-weight refresh.
  - `_split_graph_components_for_weight_update(cls, model: EasyDeLBaseModule)` — [`L76`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L76) — Split graph components from the module that actually backs eSurge.
  - `_track_finished_output(self, request_id: str)` — [`L237`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L237) — Track and evict completed RequestOutput objects to cap memory usage.
  - `_update_scheduler_heartbeat(self)` — [`L213`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L213) — Update the scheduler heartbeat timestamp.
  - `initiate(self)` — [`L262`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L262) — Start the background scheduler thread.
  - `pause(self)` — [`L527`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L527) — Pause the background scheduler without clearing queued state.
  - `release_model_state(self, *, clear_compiled_cache: bool = False)` — [`L580`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L580) — Release runner-held model weights/state references to reduce memory.
  - `resume(self)` — [`L559`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L559) — Resume the scheduler if it was paused.
  - `start_profiling(self, output_dir: str, num_batches: int = 10, host_tracer_level: int | None = None, python_tracer_level: int | None = None)` — [`L723`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L723) — Start a JAX profiler trace for the next num_batches scheduler updates.
  - `stop_profiling(self)` — [`L772`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L772) — Stop the active JAX profiler trace, if any.
  - `terminate(self)` — [`L488`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L488) — Stop the background scheduler thread.
  - `update_model_weights(self, model: EasyDeLBaseModule | None = None, *, graphdef=None, graphstate=None, graphother=None, restart_scheduler: bool = True)` — [`L605`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L605) — Hot-swap the underlying model weights/graphs.
  - `scheduler` — [`L691`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L691)
- protocol/private: `_can_prefetch_next`[`L391`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L391), `_dump_state`[`L180`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L180), `_kv_cache_valid`[`L282`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L282), `_paused`[`L289`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L289), `_profiling_active`[`L759`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L759), `_profiling_host_level`[`L762`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L762), `_profiling_output_dir`[`L761`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L761), `_profiling_python_level`[`L763`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L763), `_profiling_steps_remaining`[`L760`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L760), `_scheduler_exception`[`L144`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L144), `_scheduler_exception_tb`[`L145`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L145), `_scheduler_heartbeat`[`L215`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L215), `_scheduler_heartbeat_last_warn`[`L230`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L230), `_scheduler_loop`[`L312`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L312), `_scheduler_running`[`L148`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L148), `_scheduler_thread`[`L481`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L481)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../../infra/base_module.md#EasyDeLBaseModule), [`schedule`](../scheduler/scheduler.md#Scheduler.schedule), [`update_from_output`](../scheduler/scheduler.md#Scheduler.update_from_output), [`from_runner`](../scheduler/scheduler.md#Scheduler.from_runner), [`logger`](../logger.md#logger), [`eSurge`](../esurge_engine.md#eSurge), [`running`](../scheduler/scheduler.md#Scheduler.running), [`SchedulerOutput`](../scheduler/output.md#SchedulerOutput), [`waiting`](../scheduler/scheduler.md#Scheduler.waiting), [`esurge_compatible_model`](../../../infra/mixins/generation.md#EasyGenerationMixin.esurge_compatible_model), [`num_computed_tokens`](../request.md#EngineRequest.num_computed_tokens), [`Scheduler`](../scheduler/scheduler.md#Scheduler), [`_esurge_graphdef_from_graphdef`](../../../infra/mixins/generation.md#EasyGenerationMixin._esurge_graphdef_from_graphdef), [`requests`](../scheduler/scheduler.md#Scheduler.requests), [`num_scheduled_tokens`](../scheduler/output.md#SchedulerOutput.num_scheduled_tokens), [`num_tokens`](../request.md#EngineRequest.num_tokens), [`graphdef`](../../../infra/base_module.md#EasyDeLBaseModule.graphdef), [`split_module`](../../../infra/base_module.md#EasyDeLBaseModule.split_module), [`esurge_graphdef`](../../../infra/mixins/generation.md#EasyGenerationMixin.esurge_graphdef), [`total_num_scheduled_tokens`](../scheduler/output.md#SchedulerOutput.total_num_scheduled_tokens), [`_SCHEDULER_HEARTBEAT_WARN_INTERVAL_S`](lifecycle.md#_SCHEDULER_HEARTBEAT_WARN_INTERVAL_S), [`_SCHEDULER_HEARTBEAT_WARN_S`](lifecycle.md#_SCHEDULER_HEARTBEAT_WARN_S), [`MAX_CONSECUTIVE_SCHEDULER_ERRORS`](lifecycle.md#MAX_CONSECUTIVE_SCHEDULER_ERRORS)
- used by: [`eSurge`](../esurge_engine.md#eSurge), [`get_esurge`](../../../infra/mixins/generation.md#EasyGenerationMixin.get_esurge), [`_distributed_config_fingerprint`](../esurge_engine.md#eSurge._distributed_config_fingerprint), [`run_benchmarks`](../../../infra/elarge/model.md#eLargeModel.run_benchmarks), [`__del__`](../esurge_engine.md#eSurge.__del__), [`_cleanup_failed_esurge_generation`](../../../trainers/base_trainer.md#BaseTrainer._cleanup_failed_esurge_generation), [`_setup`](../../evaluations/esurge_eval.md#eSurgeLMEvalAdapter._setup), [`stop`](../../evaluations/esurge_eval.md#eSurgeLMEvalAdapter.stop)

## Module values
- `MAX_CONSECUTIVE_SCHEDULER_ERRORS` — [`L30`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L30)
- `_SCHEDULER_HEARTBEAT_WARN_INTERVAL_S` — [`L32`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L32)
- `_SCHEDULER_HEARTBEAT_WARN_S` — [`L31`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/lifecycle.py#L31)

