---
title: 'Module: axlearn/common/trainer.py'
type: catalog
provenance: extracted
module: axlearn/common/trainer.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.trainer`/
symbols:
  SpmdTrainer.run: SpmdTrainer#run().
  SpmdTrainer._init_with_prebuilt_state: SpmdTrainer#_init_with_prebuilt_state().
  SpmdTrainer._train_step: SpmdTrainer#_train_step().
  SpmdTrainer._get_compiled_train_step_fn: SpmdTrainer#_get_compiled_train_step_fn().
  SpmdTrainer.compile_train_step: SpmdTrainer#compile_train_step().
  SpmdTrainer._watchdog_loop: SpmdTrainer#_watchdog_loop().
  SpmdTrainer._prepare_training: SpmdTrainer#_prepare_training().
  SpmdTrainer.restore_checkpoint: SpmdTrainer#restore_checkpoint().
  SpmdTrainer._step_log: SpmdTrainer#_step_log().
  SpmdTrainer.init: SpmdTrainer#init().
  SpmdTrainer._trainer_state_specs: SpmdTrainer#_trainer_state_specs.
  SpmdTrainer._restore_from_builder: SpmdTrainer#_restore_from_builder().
  SpmdTrainer._run_step: SpmdTrainer#_run_step().
  SpmdTrainer._input_iter: SpmdTrainer#_input_iter.
  SpmdTrainer.step: SpmdTrainer#step().
  SpmdTrainer._init_state: SpmdTrainer#_init_state().
  SpmdTrainer._evalers: SpmdTrainer#_evalers.
  TrainerState: TrainerState#
  SpmdTrainer._opt_params: SpmdTrainer#_opt_params().
  SpmdTrainer.save_checkpoint: SpmdTrainer#save_checkpoint().
  SpmdTrainer._forward: SpmdTrainer#_forward().
  select_mesh_config: select_mesh_config().
  SpmdTrainer._is_initialized: SpmdTrainer#_is_initialized.
  SpmdTrainer._run_eval: SpmdTrainer#_run_eval().
  TrainerState.model: TrainerState#model.
  SpmdTrainer._per_param_train_dtype: SpmdTrainer#_per_param_train_dtype.
  SpmdTrainer._log_trainer_state_stats: SpmdTrainer#_log_trainer_state_stats().
  SpmdTrainer.input: SpmdTrainer#input.
  SpmdTrainer._maybe_record_event: SpmdTrainer#_maybe_record_event().
  SpmdTrainer.Config.model: SpmdTrainer#Config#model.
  SpmdTrainer._start_watchdog: SpmdTrainer#_start_watchdog().
  SpmdTrainer.Config: SpmdTrainer#Config#
  SpmdTrainer._trainer_state: SpmdTrainer#_trainer_state.
  SpmdTrainer.Config.mesh_shape: SpmdTrainer#Config#mesh_shape.
  SpmdTrainer._compiled_train_step: SpmdTrainer#_compiled_train_step.
  SpmdTrainer._model_param_specs: SpmdTrainer#_model_param_specs.
  SpmdTrainer.Config.input: SpmdTrainer#Config#input.
  SpmdTrainer.Config.summary_writer: SpmdTrainer#Config#summary_writer.
  SpmdTrainer.Config.checkpointer: SpmdTrainer#Config#checkpointer.
  SpmdTrainer.__init__: SpmdTrainer#__init__().
  aot_model_analysis.mb_or_gb: aot_model_analysis().mb_or_gb().
  TrainerState.prng_key: TrainerState#prng_key.
  SpmdTrainer.Config.learner: SpmdTrainer#Config#learner.
  SpmdTrainer._learner_state_partition_specs: SpmdTrainer#_learner_state_partition_specs.
  SpmdTrainer.merge_model_states: SpmdTrainer#merge_model_states().
  SpmdTrainer: SpmdTrainer#
  SpmdTrainer.model_params_for_eval: SpmdTrainer#model_params_for_eval().
  SpmdTrainer._trainer_state_partition_specs: SpmdTrainer#_trainer_state_partition_specs.
  SpmdTrainer._context_manager: SpmdTrainer#_context_manager.
  SpmdTrainer.mesh: SpmdTrainer#mesh().
  SpmdTrainer._pjit_train_step: SpmdTrainer#_pjit_train_step().
  SpmdTrainer._step: SpmdTrainer#_step.
  SpmdTrainer._device_monitor: SpmdTrainer#_device_monitor.
  SpmdTrainer._recorder: SpmdTrainer#_recorder.
  SpmdTrainer._maybe_stop_or_start_tracing: SpmdTrainer#_maybe_stop_or_start_tracing().
  SpmdTrainer.Config.dir: SpmdTrainer#Config#dir.
  SpmdTrainer._train_step_input_partition_specs: SpmdTrainer#_train_step_input_partition_specs().
  TrainerState.learner: TrainerState#learner.
  SpmdTrainer.Config.init_state_builder: SpmdTrainer#Config#init_state_builder.
  SpmdTrainer.Config.mesh_axis_names: SpmdTrainer#Config#mesh_axis_names.
  SpmdTrainer.train_cast: SpmdTrainer#train_cast().
  SpmdTrainer.trainer_state_specs: SpmdTrainer#trainer_state_specs().
  SpmdTrainer.Config.evalers: SpmdTrainer#Config#evalers.
  SpmdTrainer.Config.train_dtype: SpmdTrainer#Config#train_dtype.
  SpmdTrainer.Config.device_monitor: SpmdTrainer#Config#device_monitor.
  SpmdTrainer._watchdog_thread: SpmdTrainer#_watchdog_thread.
  SpmdTrainer._watchdog: SpmdTrainer#_watchdog().
  SpmdTrainer._stop_watchdog: SpmdTrainer#_stop_watchdog().
  SpmdTrainer._should_force_run_evals: SpmdTrainer#_should_force_run_evals().
  aot_model_analysis: aot_model_analysis().
  SpmdTrainer.trainer_state: SpmdTrainer#trainer_state().
  SpmdTrainer.Config.mesh_rules: SpmdTrainer#Config#mesh_rules.
  SpmdTrainer.Config.xsc_check_policy: SpmdTrainer#Config#xsc_check_policy.
  SpmdTrainer.Config.batch_axis_names: SpmdTrainer#Config#batch_axis_names.
  SpmdTrainer.Config.watchdog_timeout_seconds: SpmdTrainer#Config#watchdog_timeout_seconds.
  SpmdTrainer.Config.recorder: SpmdTrainer#Config#recorder.
  SpmdTrainer.Config.context_manager: SpmdTrainer#Config#context_manager.
  SpmdTrainer._maybe_monitor_all: SpmdTrainer#_maybe_monitor_all().
  SpmdTrainer.Config.save_input_iterator: SpmdTrainer#Config#save_input_iterator.
  SpmdTrainer._watchdog_stopping: SpmdTrainer#_watchdog_stopping.
  aot_model_analysis.m_or_g: aot_model_analysis().m_or_g().
  SpmdTrainer.trainer_state_partition_specs: SpmdTrainer#trainer_state_partition_specs().
  SpmdTrainer.Config.crash_on_hang_timeout_seconds: SpmdTrainer#Config#crash_on_hang_timeout_seconds.
  SpmdTrainer._jit_train_step: SpmdTrainer#_jit_train_step.
  SpmdTrainer._mesh: SpmdTrainer#_mesh.
  SpmdTrainer._xsc_check_policy: SpmdTrainer#_xsc_check_policy.
  SpmdTrainer.Config.prune_empty_state_updates: SpmdTrainer#Config#prune_empty_state_updates.
  SpmdTrainer.Config.max_step: SpmdTrainer#Config#max_step.
  SpmdTrainer.Config.start_trace_steps: SpmdTrainer#Config#start_trace_steps.
  SpmdTrainer.Config.n_steps_for_each_trace: SpmdTrainer#Config#n_steps_for_each_trace.
  SpmdTrainer.Config.start_trace_process_indices: SpmdTrainer#Config#start_trace_process_indices.
  SpmdTrainer.Config.tpu_trace_mode: SpmdTrainer#Config#tpu_trace_mode.
  SpmdTrainer.Config.host_tracer_level: SpmdTrainer#Config#host_tracer_level.
  SpmdTrainer.Config.device_tracer_level: SpmdTrainer#Config#device_tracer_level.
  SpmdTrainer.Config.python_tracer_level: SpmdTrainer#Config#python_tracer_level.
  SpmdTrainer.Config.cache_compiled_train_step: SpmdTrainer#Config#cache_compiled_train_step.
  SpmdTrainer.Config.log_every_n_steps: SpmdTrainer#Config#log_every_n_steps.
---
# Module: [`axlearn/common/trainer.py`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py)

## Classes
### `Config`  ·  implements/extends Config
- def: [`axlearn/common/trainer.py:85`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L85)
- doc: Configures SpmdTrainer.
- signature: `class Config(Module.Config):`
- members:
  - `batch_axis_names` — [`L131`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L131)
  - `cache_compiled_train_step` — [`L240`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L240)
  - `checkpointer` — [`L151`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L151)
  - `context_manager` — [`L232`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L232)
  - `crash_on_hang_timeout_seconds` — [`L221`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L221)
  - `device_monitor` — [`L225`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L225)
  - `device_tracer_level` — [`L184`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L184)
  - `dir` — [`L97`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L97)
  - `evalers` — [`L153`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L153)
  - `host_tracer_level` — [`L181`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L181)
  - `init_state_builder` — [`L101`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L101)
  - `input` — [`L89`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L89)
  - `learner` — [`L149`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L149)
  - `log_every_n_steps` — [`L244`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L244)
  - `max_step` — [`L104`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L104)
  - `mesh_axis_names` — [`L128`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L128)
  - `mesh_rules` — [`L143`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L143)
  - `mesh_shape` — [`L126`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L126)
  - `model` — [`L146`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L146)
  - `n_steps_for_each_trace` — [`L170`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L170)
  - `prune_empty_state_updates` — [`L200`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L200)
  - `python_tracer_level` — [`L187`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L187)
  - `recorder` — [`L228`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L228)
  - `save_input_iterator` — [`L164`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L164)
  - `start_trace_process_indices` — [`L172`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L172)
  - `start_trace_steps` — [`L168`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L168)
  - `summary_writer` — [`L92`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L92)
  - `tpu_trace_mode` — [`L177`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L177)
  - `train_dtype` — [`L209`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L209)
  - `watchdog_timeout_seconds` — [`L213`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L213)
  - `xsc_check_policy` — [`L193`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L193)
- uses (calls/refs, reference-scoped): [`default_config`](config.md#Configurable.default_config), [`config_class`](config.md#config_class), [`Module`](module.md#Module), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`Builder`](state_builder.md#Builder), [`InstantiableConfig`](config.md#InstantiableConfig), [`Config`](base_layer.md#BaseLayer.Config), [`ConfigOr`](config.md#ConfigOr), [`Config`](module.md#Module.Config), [`BaseWriter`](summary_writer.md#BaseWriter), [`Input`](input_base.md#Input), [`BaseModel`](base_model.md#BaseModel), [`Config`](state_builder.md#Builder.Config), [`BaseCheckpointer`](checkpointer.md#BaseCheckpointer), [`Config`](input_base.md#Input.Config), [`MeshShape`](utils.md#MeshShape), [`Config`](summary_writer.md#BaseWriter.Config), [`Config`](checkpointer.md#BaseCheckpointer.Config), [`Config`](evaler.md#SpmdEvaler.Config), [`Config`](learner.md#Learner.Config), [`Config`](monitoring/device_monitor.md#DeviceMonitor.Config), [`Learner`](learner.md#Learner), [`Checkpointer`](checkpointer.md#Checkpointer), [`PerParamFn`](utils.md#PerParamFn), [`SummaryWriter`](summary_writer.md#SummaryWriter), [`HybridMeshShape`](utils.md#HybridMeshShape), [`SpmdEvaler`](evaler.md#SpmdEvaler), [`DeviceMonitor`](monitoring/device_monitor.md#DeviceMonitor)
- used by: [`_get_compiled_train_step_fn`](trainer.md#SpmdTrainer._get_compiled_train_step_fn), [`_watchdog_loop`](trainer.md#SpmdTrainer._watchdog_loop), [`restore_checkpoint`](trainer.md#SpmdTrainer.restore_checkpoint), [`_input_iter`](trainer.md#SpmdTrainer._input_iter), [`_evalers`](trainer.md#SpmdTrainer._evalers), [`save_checkpoint`](trainer.md#SpmdTrainer.save_checkpoint), [`select_mesh_config`](trainer.md#select_mesh_config), [`_is_initialized`](trainer.md#SpmdTrainer._is_initialized), [`_per_param_train_dtype`](trainer.md#SpmdTrainer._per_param_train_dtype), [`input`](trainer.md#SpmdTrainer.input), [`_compiled_train_step`](trainer.md#SpmdTrainer._compiled_train_step), [`__init__`](trainer.md#SpmdTrainer.__init__), [`_context_manager`](trainer.md#SpmdTrainer._context_manager), [`_device_monitor`](trainer.md#SpmdTrainer._device_monitor), [`_recorder`](trainer.md#SpmdTrainer._recorder)  (2 test-only)

### `SpmdTrainer`  ·  implements/extends Module
- def: [`axlearn/common/trainer.py:80`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L80)
- doc: A trainer implementation that supports partitioning of computation and data with GSPMD.
- signature: `class SpmdTrainer(Module):`
- members:
  - `_get_compiled_train_step_fn(self, *, trainer_state: TrainerState, input_batch: NestedTensor, with_xsc: bool = False)` — [`L1050`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L1050) — Build a fully compiled train step function. — documented in [axlearn-common-trainer](../../../concepts/axlearn-common-trainer.md)
  - `_init_with_prebuilt_state(self, prng_key: Tensor, *, prebuilt_state: Optional[TrainerStateBuilder.State])` — [`L717`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L717) — Initializes `self._step` and `self._trainer_state`, optionally from `prebuilt_state`.
  - `_maybe_stop_or_start_tracing(self, stop_trace_step: Optional[int], output: Optional[dict[str, Any]])` — [`L1344`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L1344) — Stops or starts jax profiler tracing if necessary.
  - `_opt_params(self, model_params: NestedTensor)` — [`L675`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L675) — Returns a tree of OptParam for Learner.{init,update}.
  - `_prepare_training(self, prng_key: Tensor)` — [`L891`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L891) — Prepares training.
  - `_restore_from_builder(self)` — [`L1028`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L1028) — Restores trainer state by building it with init_state_builder.
  - `_run_eval(self, *, train_summaries: Optional[NestedTensor] = None, force_runs: Optional[set[str]] = None)` — [`L1177`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L1177) — Runs evaluations and returns the corresponding summaries.
  - `_run_step(self, input_batch: NestedTensor, *, force_run_evals: Optional[set[str]] = None)` — [`L1128`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L1128) — Runs a single training step.
  - `_should_force_run_evals(self, *, return_evaler_summaries: Optional[Union[bool, set[str]]] = None, evalers: dict[str, SpmdEvaler.Config])` — [`L519`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L519) — Determines which, if any, evalers to force run at the last training step.
  - `_stop_watchdog(self)` — [`L448`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L448) — Stops the checkpointer. Waits for async writes and garbage collection loop to finish.
  - `compile_train_step(self, *, trainer_state: Optional[TrainerState] = None, input_batch: Optional[dict[str, Any]] = None, compiler_options: Optional[dict[str, Union[str, bool]]] = None)` — [`L1223`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L1223) — Produce a lowered and compiled training step.
  - `init(self, prng_key: Tensor)` — [`L691`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L691) — Initializes self._step and self._trainer_state.
  - `merge_model_states(prebuilt_model_params: Nested[Union[Tensor, ParameterSpec]], initialized_model_params: Nested[Optional[NestedTensor]])` — [`L789`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L789) — Merges prebuilt and initialized params to a single tree.
  - `mesh(self)` — [`L424`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L424)
  - `model_params_for_eval(self)` — [`L404`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L404)
  - `restore_checkpoint(self, restore_step: Optional[int] = None)` — [`L944`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L944) — Restores trainer state from checkpoint.
  - `run(self, prng_key: Tensor, *, return_evaler_summaries: Optional[Union[bool, set[str]]] = None)` — [`L571`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L571) — Runs training. — documented in [axlearn-common-trainer](../../../concepts/axlearn-common-trainer.md)
  - `save_checkpoint(self, evaler_summaries: Optional[dict[str, Any]])` — [`L1017`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L1017) — Saves a checkpoint (subject to checkpointer policy).
  - `step(self)` — [`L384`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L384)
  - `train_cast(in_tree)` — [`L1273`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L1273)
  - `trainer_state(self)` — [`L388`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L388)
  - `trainer_state_partition_specs(self)` — [`L396`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L396)
  - `trainer_state_specs(self)` — [`L392`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L392)
  - `input` — [`L330`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L330)
- protocol/private: `__init__`[`L246`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L246), `_compiled_train_step`[`L321`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L321), `_context_manager`[`L308`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L308), `_device_monitor`[`L267`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L267), `_evalers`[`L366`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L366), `_forward`[`L1292`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L1292), `_init_state`[`L804`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L804), `_input_iter`[`L334`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L334), `_is_initialized`[`L269`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L269), `_jit_train_step`[`L264`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L264), `_learner_state_partition_specs`[`L352`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L352), `_log_trainer_state_stats`[`L838`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L838), `_maybe_monitor_all`[`L563`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L563), `_maybe_record_event`[`L559`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L559), `_mesh`[`L307`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L307), `_model_param_specs`[`L346`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L346), `_per_param_train_dtype`[`L281`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L281), `_pjit_train_step`[`L1205`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L1205), `_recorder`[`L268`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L268), `_start_watchdog`[`L435`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L435), `_step`[`L262`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L262), `_step_log`[`L414`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L414), `_train_step`[`L1268`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L1268), `_train_step_input_partition_specs`[`L399`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L399), `_trainer_state`[`L263`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L263), `_trainer_state_partition_specs`[`L362`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L362), `_trainer_state_specs`[`L357`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L357), `_watchdog`[`L428`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L428), `_watchdog_loop`[`L457`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L457), `_watchdog_stopping`[`L265`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L265), `_watchdog_thread`[`L266`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L266), `_xsc_check_policy`[`L320`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L320)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`default_config`](config.md#Configurable.default_config), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`functional`](module.md#functional), [`_add_child`](module.md#Module._add_child), [`ParameterSpec`](base_layer.md#ParameterSpec), [`Builder`](state_builder.md#Builder), [`shape`](utils.md#TensorSpec.shape), [`flatten_items`](utils.md#flatten_items), [`mesh_axes`](utils.md#TensorSpec.mesh_axes), [`dtype`](utils.md#TensorSpec.dtype), [`maybe_instantiate`](config.md#maybe_instantiate), [`State`](state_builder.md#Builder.State), [`child_context`](module.md#child_context), [`trainer_state`](state_builder.md#Builder.State.trainer_state), [`vlog`](module.md#Module.vlog), [`shapes`](utils.md#shapes), [`NestedOptParam`](optimizer_base.md#NestedOptParam), [`__init__`](module.md#Module.__init__), [`new_output_collection`](module.md#new_output_collection), [`path`](module.md#Module.path), [`OptParam`](optimizer_base.md#OptParam), [`summaries`](module.md#OutputCollection.summaries), [`maybe_set_config`](config.md#maybe_set_config), [`dispatch_global_batch`](input_base.md#Input.dispatch_global_batch), [`children`](module.md#Module.children), [`StateType`](state_builder.md#Builder.StateType), [`TrainerState`](trainer.md#TrainerState), [`Input`](input_base.md#Input), [`batches`](input_base.md#Input.batches), [`InvocationContext`](module.md#InvocationContext), [`built_keys`](state_builder.md#Builder.State.built_keys), [`host_to_global_array`](utils.md#host_to_global_array), [`step`](state_builder.md#Builder.State.step), [`create_device_mesh`](utils.md#create_device_mesh), [`input_dispatcher`](input_base.md#Input.input_dispatcher), [`model`](trainer.md#TrainerState.model)  (+62 more)
- used by: [`Module`](module.md#Module), [`select_mesh_config`](trainer.md#select_mesh_config)  (2 test-only)

### `TrainerState`  ·  implements/extends NamedTuple
- def: [`axlearn/common/trainer.py:73`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L73)
- signature: `class TrainerState(NamedTuple):`
- members:
  - `learner` — [`L76`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L76)
  - `model` — [`L75`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L75)
  - `prng_key` — [`L74`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L74)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`NestedTensor`](utils.md#NestedTensor), [`TensorSpec`](utils.md#TensorSpec)
- used by: [`_init_with_prebuilt_state`](trainer.md#SpmdTrainer._init_with_prebuilt_state), [`_train_step`](trainer.md#SpmdTrainer._train_step), [`_get_compiled_train_step_fn`](trainer.md#SpmdTrainer._get_compiled_train_step_fn), [`compile_train_step`](trainer.md#SpmdTrainer.compile_train_step), [`restore_checkpoint`](trainer.md#SpmdTrainer.restore_checkpoint), [`_trainer_state_specs`](trainer.md#SpmdTrainer._trainer_state_specs), [`_init_state`](trainer.md#SpmdTrainer._init_state), [`_run_eval`](trainer.md#SpmdTrainer._run_eval), [`_log_trainer_state_stats`](trainer.md#SpmdTrainer._log_trainer_state_stats), [`_trainer_state`](trainer.md#SpmdTrainer._trainer_state), [`model_params_for_eval`](trainer.md#SpmdTrainer.model_params_for_eval), [`_trainer_state_partition_specs`](trainer.md#SpmdTrainer._trainer_state_partition_specs)

## Functions
- `aot_model_analysis(compiled: jax.stages.Compiled)` — [`L1426`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L1426) — Performs the model analysis on the AOT compiled JAX program.
- `m_or_g(x, suffix="")` — [`L1450`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L1450)
- `mb_or_gb(x)` — [`L1460`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L1460)
- `select_mesh_config(trainer_config: SpmdTrainer.Config, *, mesh_selector: str)` — [`L1401`](../../../../../../raw/code/axlearn/axlearn/common/trainer.py#L1401) — Selects a mesh rule (if one matches `mesh_selector` to override mesh config.

