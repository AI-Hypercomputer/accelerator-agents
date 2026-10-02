---
title: 'Module: python/sgl_jax/srt/managers/scheduler_profiler_mixing.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/managers/scheduler_profiler_mixing.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.managers.scheduler_profiler_mixing`/
symbols:
  SchedulerProfilerMixin._start_stage_profile: SchedulerProfilerMixin#_start_stage_profile().
  SchedulerProfilerMixin.profile: SchedulerProfilerMixin#profile().
  SchedulerProfilerMixin._profile_batch_predicate: SchedulerProfilerMixin#_profile_batch_predicate().
  SchedulerProfilerMixin.start_profile: SchedulerProfilerMixin#start_profile().
  SchedulerProfilerMixin.stop_profile: SchedulerProfilerMixin#stop_profile().
  _StageBasedTrigger.step: _StageBasedTrigger#step().
  SchedulerProfilerMixin.get_profile_status: SchedulerProfilerMixin#get_profile_status().
  _ProfileManager.configure: _ProfileManager#configure().
  _ProfileManager.step: _ProfileManager#step().
  _ProfileManager._do_start: _ProfileManager#_do_start().
  _StageBasedTrigger.running_state: _StageBasedTrigger#running_state.
  _ProfileManager._trigger: _ProfileManager#_trigger.
  _get_stage_from_forward_mode: _get_stage_from_forward_mode().
  _StageBasedTrigger.stage_configs: _StageBasedTrigger#stage_configs.
  _StageBasedTrigger.configure: _StageBasedTrigger#configure().
  SchedulerProfilerMixin._profile_manager: SchedulerProfilerMixin#_profile_manager.
  _ProfileManager.is_configured: _ProfileManager#is_configured().
  SchedulerProfilerMixin.profile_in_progress: SchedulerProfilerMixin#profile_in_progress.
  _StageBasedTrigger.reset: _StageBasedTrigger#reset().
  logger: logger.
  SchedulerProfilerMixin.profiler_output_dir: SchedulerProfilerMixin#profiler_output_dir.
  _ProfileManager.stop: _ProfileManager#stop().
  _ProfileManager._do_stop: _ProfileManager#_do_stop().
  SchedulerProfilerMixin.profiler_start_forward_ct: SchedulerProfilerMixin#profiler_start_forward_ct.
  SchedulerProfilerMixin.profiler_target_forward_ct: SchedulerProfilerMixin#profiler_target_forward_ct.
  _StageBasedTrigger.is_configured: _StageBasedTrigger#is_configured().
  SchedulerProfilerMixin: SchedulerProfilerMixin#
  _StageBasedTrigger._RunningState.curr_stage: _StageBasedTrigger#_RunningState#curr_stage.
  SchedulerProfilerMixin.profile_id: SchedulerProfilerMixin#profile_id.
  _StageBasedTrigger: _StageBasedTrigger#
  _StageBasedTrigger._RunningState.curr_count: _StageBasedTrigger#_RunningState#curr_count.
  _ProfileManager._host_tracer_level: _ProfileManager#_host_tracer_level.
  _ProfileManager._python_tracer_level: _ProfileManager#_python_tracer_level.
  _ProfileManager._trace_active: _ProfileManager#_trace_active.
  _StageBasedTrigger._StageConfig: _StageBasedTrigger#_StageConfig#
  _StageBasedTrigger._StageConfig.target_count: _StageBasedTrigger#_StageConfig#target_count.
  _StageBasedTrigger._RunningState: _StageBasedTrigger#_RunningState#
  _StageBasedTrigger.on_stop: _StageBasedTrigger#on_stop.
  _ProfileManager._output_dir: _ProfileManager#_output_dir.
  SchedulerProfilerMixin.profile_steps: SchedulerProfilerMixin#profile_steps.
  SchedulerProfilerMixin.host_tracer_level: SchedulerProfilerMixin#host_tracer_level.
  SchedulerProfilerMixin.python_tracer_level: SchedulerProfilerMixin#python_tracer_level.
  _StageBasedTrigger.on_start: _StageBasedTrigger#on_start.
  _ProfileManager: _ProfileManager#
  SchedulerProfilerMixin.init_profier: SchedulerProfilerMixin#init_profier().
  _StageBasedTrigger.__init__: _StageBasedTrigger#__init__().
  _ProfileManager.__init__: _ProfileManager#__init__().
---
# Module: [`python/sgl_jax/srt/managers/scheduler_profiler_mixing.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py)

## Classes
### `SchedulerProfilerMixin`
- def: [`python/sgl_jax/srt/managers/scheduler_profiler_mixing.py:149`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L149)
- signature: `class SchedulerProfilerMixin:`
- members:
  - `get_profile_status(self)` — [`L341`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L341)
  - `init_profier(self)` — [`L150`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L150)
  - `profile(self, recv_req: ProfileReq)` — [`L324`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L324)
  - `start_profile(self, output_dir: str | None, start_step: int | None, num_steps: int | None, host_tracer_level: int | None, python_tracer_level: int | None, profile_id: str)` — [`L161`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L161)
  - `stop_profile(self)` — [`L273`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L273)
  - `host_tracer_level` — [`L157`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L157)
  - `profile_id` — [`L152`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L152)
  - `profile_in_progress` — [`L156`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L156)
  - `profile_steps` — [`L155`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L155)
  - `profiler_output_dir` — [`L151`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L151)
  - `profiler_start_forward_ct` — [`L153`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L153)
  - `profiler_target_forward_ct` — [`L154`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L154)
  - `python_tracer_level` — [`L158`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L158)
- protocol/private: `_profile_batch_predicate`[`L302`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L302), `_profile_manager`[`L159`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L159), `_start_stage_profile`[`L231`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L231)
- uses (calls/refs, reference-scoped): [`Scheduler`](scheduler.md#Scheduler), [`ProfileReqOutput`](io_struct.md#ProfileReqOutput), [`configure`](scheduler_profiler_mixing.md#_ProfileManager.configure), [`step`](scheduler_profiler_mixing.md#_ProfileManager.step), [`message`](io_struct.md#ProfileReqOutput.message), [`success`](io_struct.md#ProfileReqOutput.success), [`is_configured`](scheduler_profiler_mixing.md#_ProfileManager.is_configured), [`logger`](scheduler_profiler_mixing.md#logger), [`ProfileReq`](io_struct.md#ProfileReq), [`stop`](scheduler_profiler_mixing.md#_ProfileManager.stop), [`ProfileReqType`](io_struct.md#ProfileReqType), [`host_tracer_level`](io_struct.md#ProfileReq.host_tracer_level), [`num_steps`](io_struct.md#ProfileReq.num_steps), [`output_dir`](io_struct.md#ProfileReq.output_dir), [`profile_id`](io_struct.md#ProfileReq.profile_id), [`python_tracer_level`](io_struct.md#ProfileReq.python_tracer_level), [`_ProfileManager`](scheduler_profiler_mixing.md#_ProfileManager), [`GET_STATUS`](io_struct.md#ProfileReqType.GET_STATUS), [`START_PROFILE`](io_struct.md#ProfileReqType.START_PROFILE), [`profile_by_stage`](io_struct.md#ProfileReq.profile_by_stage), [`profile_stages`](io_struct.md#ProfileReq.profile_stages), [`start_step`](io_struct.md#ProfileReq.start_step)
- used by: [`run_batch`](scheduler.md#Scheduler.run_batch), [`_request_dispatcher`](scheduler.md#Scheduler._request_dispatcher), [`Scheduler`](scheduler.md#Scheduler), [`parent_process`](scheduler.md#Scheduler.parent_process)

### `_ProfileManager`
- def: [`python/sgl_jax/srt/managers/scheduler_profiler_mixing.py:86`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L86)
- doc: Manages stage-based profiling with JAX profiler backend.
- signature: `class _ProfileManager:`
- members:
  - `configure(self, output_dir: str, num_steps: int, interesting_stages: list[str], host_tracer_level: int | None = None, python_tracer_level: int | None = None)` — [`L99`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L99)
  - `is_configured(self)` — [`L113`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L113)
  - `step(self, forward_mode: ForwardMode)` — [`L116`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L116)
  - `stop(self)` — [`L124`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L124) — Force stop any active trace (for manual /stop_profile).
- protocol/private: `__init__`[`L89`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L89), `_do_start`[`L128`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L128), `_do_stop`[`L142`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L142), `_host_tracer_level`[`L95`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L95), `_output_dir`[`L94`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L94), `_python_tracer_level`[`L96`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L96), `_trace_active`[`L97`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L97), `_trigger`[`L90`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L90)
- uses (calls/refs, reference-scoped): [`ForwardMode`](../model_executor/forward_batch_info.md#ForwardMode), [`step`](scheduler_profiler_mixing.md#_StageBasedTrigger.step), [`_get_stage_from_forward_mode`](scheduler_profiler_mixing.md#_get_stage_from_forward_mode), [`configure`](scheduler_profiler_mixing.md#_StageBasedTrigger.configure), [`reset`](scheduler_profiler_mixing.md#_StageBasedTrigger.reset), [`logger`](scheduler_profiler_mixing.md#logger), [`is_configured`](scheduler_profiler_mixing.md#_StageBasedTrigger.is_configured), [`_StageBasedTrigger`](scheduler_profiler_mixing.md#_StageBasedTrigger)
- used by: [`_start_stage_profile`](scheduler_profiler_mixing.md#SchedulerProfilerMixin._start_stage_profile), [`_profile_batch_predicate`](scheduler_profiler_mixing.md#SchedulerProfilerMixin._profile_batch_predicate), [`stop_profile`](scheduler_profiler_mixing.md#SchedulerProfilerMixin.stop_profile), [`get_profile_status`](scheduler_profiler_mixing.md#SchedulerProfilerMixin.get_profile_status), [`_profile_manager`](scheduler_profiler_mixing.md#SchedulerProfilerMixin._profile_manager)

### `_RunningState`
- def: [`python/sgl_jax/srt/managers/scheduler_profiler_mixing.py:37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L37)
- signature: `class _RunningState:`
- members:
  - `curr_count` — [`L39`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L39)
  - `curr_stage` — [`L38`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L38)
- used by: [`step`](scheduler_profiler_mixing.md#_StageBasedTrigger.step), [`running_state`](scheduler_profiler_mixing.md#_StageBasedTrigger.running_state)

### `_StageBasedTrigger`
- def: [`python/sgl_jax/srt/managers/scheduler_profiler_mixing.py:26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L26)
- doc: State machine that triggers profiler start/stop based on stage transitions.
- signature: `class _StageBasedTrigger:`
- members:
  - `configure(self, num_steps: int, interesting_stages: list[str])` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L47)
  - `is_configured(self)` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L54)
  - `reset(self)` — [`L79`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L79)
  - `step(self, stage: str)` — [`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L57)
  - `on_start` — [`L42`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L42)
  - `on_stop` — [`L43`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L43)
  - `running_state` — [`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L44)
  - `stage_configs` — [`L45`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L45)
- protocol/private: `__init__`[`L41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L41)
- uses (calls/refs, reference-scoped): [`curr_stage`](scheduler_profiler_mixing.md#_StageBasedTrigger._RunningState.curr_stage), [`curr_count`](scheduler_profiler_mixing.md#_StageBasedTrigger._RunningState.curr_count), [`_RunningState`](scheduler_profiler_mixing.md#_StageBasedTrigger._RunningState), [`_StageConfig`](scheduler_profiler_mixing.md#_StageBasedTrigger._StageConfig), [`target_count`](scheduler_profiler_mixing.md#_StageBasedTrigger._StageConfig.target_count)
- used by: [`configure`](scheduler_profiler_mixing.md#_ProfileManager.configure), [`step`](scheduler_profiler_mixing.md#_ProfileManager.step), [`_trigger`](scheduler_profiler_mixing.md#_ProfileManager._trigger), [`is_configured`](scheduler_profiler_mixing.md#_ProfileManager.is_configured), [`stop`](scheduler_profiler_mixing.md#_ProfileManager.stop)

### `_StageConfig`
- def: [`python/sgl_jax/srt/managers/scheduler_profiler_mixing.py:33`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L33)
- signature: `class _StageConfig:`
- members:
  - `target_count` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L34)
- used by: [`step`](scheduler_profiler_mixing.md#_StageBasedTrigger.step), [`configure`](scheduler_profiler_mixing.md#_StageBasedTrigger.configure), [`stage_configs`](scheduler_profiler_mixing.md#_StageBasedTrigger.stage_configs)

## Functions
- `_get_stage_from_forward_mode(forward_mode: ForwardMode)` — [`L15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L15)

## Module values
- `logger` — [`L12`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/scheduler_profiler_mixing.py#L12)

