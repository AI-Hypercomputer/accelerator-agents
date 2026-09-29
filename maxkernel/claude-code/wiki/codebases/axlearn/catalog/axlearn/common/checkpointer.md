---
title: 'Module: axlearn/common/checkpointer.py'
type: catalog
provenance: extracted
module: axlearn/common/checkpointer.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.checkpointer`/
symbols:
  TensorStoreStateStorage._get_spec: TensorStoreStateStorage#_get_spec().
  TensorStoreStateStorage.save_to_dir: TensorStoreStateStorage#save_to_dir().
  Checkpointer.restore: Checkpointer#restore().
  TensorStoreStateStorage.restore_from_dir: TensorStoreStateStorage#restore_from_dir().
  TensorStoreStateStorage._restore_tensorstore_state: TensorStoreStateStorage#_restore_tensorstore_state().
  Checkpointer._run_garbage_collection: Checkpointer#_run_garbage_collection().
  TensorStoreStateStorage._manager: TensorStoreStateStorage#_manager.
  Checkpointer.save: Checkpointer#save().
  CheckpointValidationType: CheckpointValidationType#
  async_save_tf_savables: async_save_tf_savables().
  Checkpointer._gc_loop: Checkpointer#_gc_loop().
  check_state_structure: check_state_structure().
  Checkpointer._save_policy: Checkpointer#_save_policy.
  BaseCheckpointer.Config.dir: BaseCheckpointer#Config#dir.
  Checkpointer.validate_and_restore: Checkpointer#validate_and_restore().
  restore_tf_savables: restore_tf_savables().
  Checkpointer.__init__: Checkpointer#__init__().
  BaseCheckpointer: BaseCheckpointer#
  Checkpointer.ckpt_dir: Checkpointer#ckpt_dir().
  StateStorage.restore_from_dir: StateStorage#restore_from_dir().
  Checkpointer.Config: Checkpointer#Config#
  Checkpointer._start_gc_thread: Checkpointer#_start_gc_thread().
  maybe_save_python_savables: maybe_save_python_savables().
  maybe_restore_python_savables: maybe_restore_python_savables().
  TensorStoreStateStorage.__init__: TensorStoreStateStorage#__init__().
  Checkpointer.stop: Checkpointer#stop().
  BaseCheckpointer.Config.save_policy: BaseCheckpointer#Config#save_policy.
  Checkpointer._index_writer: Checkpointer#_index_writer.
  Checkpointer._storage: Checkpointer#_storage.
  StateStorage: StateStorage#
  Checkpointer.Config.storage: Checkpointer#Config#storage.
  BaseCheckpointer.Config: BaseCheckpointer#Config#
  drop_learner_optimizer_cfg: drop_learner_optimizer_cfg().
  read_state_spec: read_state_spec().
  BestMetricPolicy.__call__: BestMetricPolicy#__call__().
  BaseCheckpointer.restore: BaseCheckpointer#restore().
  check_state_structure.filter_for_validation: check_state_structure().filter_for_validation().
  BestMetricPolicy.best_metric: BestMetricPolicy#best_metric.
  BaseCheckpointer.__init__: BaseCheckpointer#__init__().
  Checkpointer: Checkpointer#
  TensorStoreStateStorage.CheckpointSpec.index: TensorStoreStateStorage#CheckpointSpec#index.
  Checkpointer.Config.summary_writer: Checkpointer#Config#summary_writer.
  StateStorage.save_to_dir: StateStorage#save_to_dir().
  _parse_tensor_spec: _parse_tensor_spec().
  TensorStoreStateStorage.Config: TensorStoreStateStorage#Config#
  BestMetricPolicy.Config: BestMetricPolicy#Config#
  CheckpointValidationType.EXACT: CheckpointValidationType#EXACT.
  RestoreStateFilter: RestoreStateFilter.
  build_step_dir: build_step_dir().
  parse_step_from_dir: parse_step_from_dir().
  BestMetricPolicy.Config.metric: BestMetricPolicy#Config#metric.
  every_n_steps_and_last_policy: every_n_steps_and_last_policy().
  BaseCheckpointer.save: BaseCheckpointer#save().
  every_n_steps_policy: every_n_steps_policy().
  Checkpointer.Config.index_writer: Checkpointer#Config#index_writer.
  Checkpointer.Config.restore_state_filter: Checkpointer#Config#restore_state_filter.
  TensorStoreStateStorage.CheckpointSpec.gda_values: TensorStoreStateStorage#CheckpointSpec#gda_values.
  drop_learner_optimizer_cfg.fn: drop_learner_optimizer_cfg().fn().
  TensorStoreStateStorage.CheckpointSpec.tf_ckpt_map: TensorStoreStateStorage#CheckpointSpec#tf_ckpt_map.
  TensorStoreStateStorage.CheckpointSpec.python_ckpt_map: TensorStoreStateStorage#CheckpointSpec#python_ckpt_map.
  BaseCheckpointer.checkpoint_paths: BaseCheckpointer#checkpoint_paths().
  BaseCheckpointer.checkpoint_steps: BaseCheckpointer#checkpoint_steps().
  BaseCheckpointer.stop: BaseCheckpointer#stop().
  Checkpointer._gc_thread: Checkpointer#_gc_thread.
  Checkpointer.wait_until_finished: Checkpointer#wait_until_finished().
  TensorStoreStateStorage: TensorStoreStateStorage#
  PythonSavable: PythonSavable#
  TensorStoreStateStorage.CheckpointSpec.tensorstore_specs: TensorStoreStateStorage#CheckpointSpec#tensorstore_specs.
  BestMetricPolicy.__init__: BestMetricPolicy#__init__().
  BaseCheckpointer.__exit__: BaseCheckpointer#__exit__().
  BaseCheckpointer.wait_until_finished: BaseCheckpointer#wait_until_finished().
  Checkpointer.checkpoint_steps: Checkpointer#checkpoint_steps().
  Checkpointer.__enter__: Checkpointer#__enter__().
  read_index_file: read_index_file().
  Checkpointer._all_checkpoint_paths: Checkpointer#_all_checkpoint_paths().
  Checkpointer.checkpoint_paths: Checkpointer#checkpoint_paths().
  STEP_NUM_DIGITS: STEP_NUM_DIGITS.
  STEP_PREFIX: STEP_PREFIX.
  TensorStoreStateStorage.Config.max_concurrent_restore_gb: TensorStoreStateStorage#Config#max_concurrent_restore_gb.
  TensorStoreStateStorage.CheckpointSpec.shardings: TensorStoreStateStorage#CheckpointSpec#shardings.
  CheckpointPolicy: CheckpointPolicy#
  PythonSavable.get_state: PythonSavable#get_state().
  PythonSavable.set_state: PythonSavable#set_state().
  StateStorage.wait_until_finished: StateStorage#wait_until_finished().
  StateStorage.stop: StateStorage#stop().
  TensorStoreStateStorage._max_concurrent_restore_gb: TensorStoreStateStorage#_max_concurrent_restore_gb.
  TensorStoreStateStorage.commit: TensorStoreStateStorage#commit().
  BaseCheckpointer.latest_checkpoint_path: BaseCheckpointer#latest_checkpoint_path().
  BaseCheckpointer.__enter__: BaseCheckpointer#__enter__().
  Checkpointer.Config.validation: Checkpointer#Config#validation.
  TensorStoreStateStorage.Config.shard_threshold_bytes: TensorStoreStateStorage#Config#shard_threshold_bytes.
  TensorStoreStateStorage._executor: TensorStoreStateStorage#_executor.
  TensorStoreStateStorage.CheckpointSpec: TensorStoreStateStorage#CheckpointSpec#
  TensorStoreStateStorage.CheckpointSpec.storage_paths: TensorStoreStateStorage#CheckpointSpec#storage_paths.
  TensorStoreStateStorage.CheckpointSpec.shapes: TensorStoreStateStorage#CheckpointSpec#shapes.
  TensorStoreStateStorage.CheckpointSpec.dtypes: TensorStoreStateStorage#CheckpointSpec#dtypes.
  BaseCheckpointer._within_context: BaseCheckpointer#_within_context.
  Checkpointer.Config.keep_every_n_steps: Checkpointer#Config#keep_every_n_steps.
  Checkpointer._gc_stopping: Checkpointer#_gc_stopping.
  TensorStoreStateStorage.wait_until_finished: TensorStoreStateStorage#wait_until_finished().
  TensorStoreStateStorage.stop: TensorStoreStateStorage#stop().
  BestMetricPolicy: BestMetricPolicy#
  BaseCheckpointer.latest_checkpoint_step: BaseCheckpointer#latest_checkpoint_step().
  CheckpointValidationType.EXACT_UP_TO_DTYPE: CheckpointValidationType#EXACT_UP_TO_DTYPE.
  CheckpointValidationType.CONTAINS_STATE_UP_TO_DTYPE: CheckpointValidationType#CONTAINS_STATE_UP_TO_DTYPE.
  StateStorageCommitCallback: StateStorageCommitCallback#
  write_index_file: write_index_file().
  TensorStoreStateStorage.Config.timeout_secs: TensorStoreStateStorage#Config#timeout_secs.
  TensorStoreStateStorage.Config.max_data_shard_degree: TensorStoreStateStorage#Config#max_data_shard_degree.
  TensorStoreStateStorage.Config.max_concurrent_gb: TensorStoreStateStorage#Config#max_concurrent_gb.
  CheckpointValidationType.CONTAINS_STATE: CheckpointValidationType#CONTAINS_STATE.
  drop_learner_optimizer_cfg.fn.filter_fn: drop_learner_optimizer_cfg().fn().filter_fn().
  _upload_dir: _upload_dir().
  IndexFileWriter: IndexFileWriter#
  TensorStoreStateStorage._spec_from_path: TensorStoreStateStorage#_spec_from_path().
  EvalMetric: EvalMetric#
  every_n_steps_policy.fn: every_n_steps_policy().fn().
  every_n_steps_and_last_policy.fn: every_n_steps_and_last_policy().fn().
  Checkpointer.Config.keep_last_n: Checkpointer#Config#keep_last_n.
  Checkpointer.Config.gc_loop_interval_seconds: Checkpointer#Config#gc_loop_interval_seconds.
  Checkpointer.cleanup_checkpoint: Checkpointer#cleanup_checkpoint().
  Checkpointer._index_exists: Checkpointer#_index_exists().
  StateStorageCommitCallback.__call__: StateStorageCommitCallback#__call__().
  IndexFileWriter.__call__: IndexFileWriter#__call__().
  CheckpointPolicy.__call__: CheckpointPolicy#__call__().
  EvalMetric.evaler_name: EvalMetric#evaler_name.
  EvalMetric.metric_name: EvalMetric#metric_name.
  BestMetricPolicy.Config.mode: BestMetricPolicy#Config#mode.
---
# Module: [`axlearn/common/checkpointer.py`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py)

## Classes
### `BaseCheckpointer`  ·  implements/extends Module
- def: [`axlearn/common/checkpointer.py:800`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L800)
- doc: A base checkpointer interface.
- signature: `class BaseCheckpointer(Module):`
- members:
  - `__enter__(self)` — [`L879`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L879) — Enters the checkpointer context manager.
  - `__exit__(self, exc_type: Optional[type[BaseException]], exc: Optional[BaseException], traceback: Optional[TracebackType])` — [`L891`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L891) — Exits the checkpointer context manager.
  - `checkpoint_paths(cls, base_dir: str)` — [`L827`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L827) — Returns complete checkpoint paths under base dir.
  - `checkpoint_steps(cls, base_dir: str)` — [`L858`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L858) — Returns complete checkpoint steps under base dir.
  - `latest_checkpoint_path(cls, base_dir: str)` — [`L839`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L839) — Returns the most recent (highest step count) complete checkpoint under base dir.
  - `latest_checkpoint_step(cls, base_dir: str)` — [`L870`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L870) — Returns the most recent (highest step count) checkpoint step under base dir.
  - `restore(self, *, step: Optional[int] = None, state: Union[NestedTensor, NestedTensorSpec])` — [`L922`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L922) — Restores from the checkpoint directory.
  - `save(self, *, step: int, state: NestedTensor, evaler_summaries: Optional[dict[str, Any]] = None)` — [`L909`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L909) — Saves `state` at the given `step`.
  - `stop(self, *, has_exception: bool = False)` — [`L947`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L947) — Stops the checkpointer. Waits for async writes, garbage collection, etc. to finish.
  - `wait_until_finished(self)` — [`L943`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L943) — Waits for pending asynchronous saves to finish.
- protocol/private: `__init__`[`L875`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L875), `_within_context`[`L877`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L877)
- uses (calls/refs, reference-scoped): [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`Config`](module.md#Module.Config), [`restore`](checkpointer.md#Checkpointer.restore), [`__init__`](module.md#Module.__init__), [`restore`](checkpointer_orbax.md#OrbaxCheckpointer.restore), [`save`](checkpointer.md#Checkpointer.save), [`NestedTensorSpec`](utils.md#NestedTensorSpec), [`save`](checkpointer_orbax.md#OrbaxCheckpointer.save), [`stop`](checkpointer.md#Checkpointer.stop), [`Checkpointer`](checkpointer.md#Checkpointer), [`OrbaxCheckpointer`](checkpointer_orbax.md#OrbaxCheckpointer), [`wait_until_finished`](checkpointer.md#Checkpointer.wait_until_finished), [`checkpoint_steps`](checkpointer.md#Checkpointer.checkpoint_steps), [`checkpoint_paths`](checkpointer.md#Checkpointer.checkpoint_paths), [`stop`](checkpointer_orbax.md#OrbaxCheckpointer.stop), [`wait_until_finished`](checkpointer_orbax.md#OrbaxCheckpointer.wait_until_finished), [`checkpoint_paths`](checkpointer_orbax.md#OrbaxCheckpointer.checkpoint_paths), [`checkpoint_steps`](checkpointer_orbax.md#OrbaxCheckpointer.checkpoint_steps)
- used by: [`Module`](module.md#Module), [`restore`](checkpointer.md#Checkpointer.restore), [`__init__`](checkpointer_orbax.md#OrbaxCheckpointer.__init__), [`__init__`](checkpointer.md#Checkpointer.__init__), [`Config`](checkpointer.md#Checkpointer.Config), [`checkpointer`](trainer.md#SpmdTrainer.Config.checkpointer), [`Config`](checkpointer_orbax.md#OrbaxCheckpointer.Config), [`Checkpointer`](checkpointer.md#Checkpointer), [`OrbaxCheckpointer`](checkpointer_orbax.md#OrbaxCheckpointer), [`__enter__`](checkpointer.md#Checkpointer.__enter__)

### `BestMetricPolicy`  ·  implements/extends Configurable
- def: [`axlearn/common/checkpointer.py:698`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L698)
- doc: A CheckpointPolicy that saves checkpoint only when there is a better eval metric.
- signature: `class BestMetricPolicy(Configurable):`
- members:
  - `best_metric` — [`L720`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L720)
- protocol/private: `__call__`[`L722`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L722), `__init__`[`L718`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L718)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`Configurable`](config.md#Configurable), [`WeightedSummary`](metrics.md#WeightedSummary), [`__init__`](config.md#Configurable.__init__), [`Config`](checkpointer.md#BestMetricPolicy.Config), [`mean`](metrics.md#WeightedValue.mean)
- used by: [`Configurable`](config.md#Configurable)

### `CheckpointPolicy`  ·  implements/extends Protocol
- def: [`axlearn/common/checkpointer.py:669`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L669)
- doc: Decides whether checkpointer should save at the given step.
- signature: `class CheckpointPolicy(Protocol):`
- members:
  - `__call__(self, *, step: int, evaler_summaries: dict[str, Any])` — [`L672`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L672) — Implements the policy.
- used by: [`_save_policy`](checkpointer.md#Checkpointer._save_policy), [`save_policy`](checkpointer.md#BaseCheckpointer.Config.save_policy), [`every_n_steps_and_last_policy`](checkpointer.md#every_n_steps_and_last_policy), [`every_n_steps_policy`](checkpointer.md#every_n_steps_policy)

### `CheckpointSpec`
- def: [`axlearn/common/checkpointer.py:483`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L483)
- signature: `class CheckpointSpec:`
- members:
  - `dtypes` — [`L488`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L488)
  - `gda_values` — [`L490`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L490)
  - `index` — [`L484`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L484)
  - `python_ckpt_map` — [`L492`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L492)
  - `shapes` — [`L487`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L487)
  - `shardings` — [`L489`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L489)
  - `storage_paths` — [`L485`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L485)
  - `tensorstore_specs` — [`L486`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L486)
  - `tf_ckpt_map` — [`L491`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L491)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor)
- used by: [`_get_spec`](checkpointer.md#TensorStoreStateStorage._get_spec), [`save_to_dir`](checkpointer.md#TensorStoreStateStorage.save_to_dir), [`restore_from_dir`](checkpointer.md#TensorStoreStateStorage.restore_from_dir), [`_restore_tensorstore_state`](checkpointer.md#TensorStoreStateStorage._restore_tensorstore_state), [`commit`](checkpointer.md#TensorStoreStateStorage.commit)

### `CheckpointValidationType`  ·  implements/extends Enum, str
- def: [`axlearn/common/checkpointer.py:66`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L66)
- doc: Represents a type of checkpoint validation.
- signature: `class CheckpointValidationType(str, enum.Enum):`
- members:
  - `CONTAINS_STATE` — [`L83`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L83)
  - `CONTAINS_STATE_UP_TO_DTYPE` — [`L84`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L84)
  - `EXACT` — [`L81`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L81)
  - `EXACT_UP_TO_DTYPE` — [`L82`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L82)
- used by: [`restore`](checkpointer.md#Checkpointer.restore), [`restore_from_dir`](checkpointer.md#TensorStoreStateStorage.restore_from_dir), [`check_state_structure`](checkpointer.md#check_state_structure), [`restore_from_dir`](checkpointer.md#StateStorage.restore_from_dir), [`filter_for_validation`](checkpointer.md#check_state_structure.filter_for_validation), [`validation`](state_builder.md#BaseStateStorageBuilder.Config.validation), [`validation_type`](checkpointer_orbax.md#OrbaxCheckpointer.Config.validation_type), [`validation`](state_builder.md#ChainBuilder.Config.validation), [`validation`](checkpointer.md#Checkpointer.Config.validation)

### `Checkpointer`  ·  implements/extends BaseCheckpointer
- def: [`axlearn/common/checkpointer.py:957`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L957)
- doc: A checkpointer that supports various StateStorage implementations.
- signature: `class Checkpointer(BaseCheckpointer):`
- members:
  - `_all_checkpoint_paths(cls, base_dir: str)` — [`L994`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L994) — Like `checkpoint_paths`, but also include non-committed checkpoints.
  - `_gc_loop(self, *, context_stack: list[InvocationContext])` — [`L1093`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1093) — Starts garbage collection loop. Will block the current thread.
  - `_run_garbage_collection(self)` — [`L1132`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1132) — Runs one round of garbage collection of past checkpoints.
  - `_start_gc_thread(self)` — [`L1063`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1063) — Starts garbage collection (if not already started) in a separate thread.
  - `checkpoint_paths(cls, base_dir: str)` — [`L1006`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1006) — See `BaseCheckpointer.checkpointer_paths`.
  - `checkpoint_steps(cls, base_dir: str)` — [`L1019`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1019)
  - `ckpt_dir(self, step: int)` — [`L1103`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1103) — Obtains the checkpoint dir for the given step.
  - `cleanup_checkpoint(cls, ckpt_dir: str, *, sync: bool = True)` — [`L1023`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1023) — Removes ckpt_dir if it exists.
  - `restore(self, *, step: Optional[int] = None, state: Union[NestedTensor, NestedTensorSpec])` — [`L1218`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1218) — See `BaseCheckpointer.restore` docstring for details.
  - `save(self, *, step: int, state: NestedTensor, evaler_summaries: Optional[dict[str, Any]] = None)` — [`L1108`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1108) — See `BaseCheckpointer.save` for details.
  - `stop(self, *, has_exception: bool = False)` — [`L1074`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1074) — See `BaseCheckpointer.stop` for details.
  - `validate_and_restore(*, step: int, ckpt_dir: str)` — [`L1236`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1236)
  - `wait_until_finished(self)` — [`L1210`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1210) — See `BaseCheckpointer.wait_until_finished` docstring for details.
  - `_index_writer` — [`L1047`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1047) — ---
- protocol/private: `__enter__`[`L1059`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1059), `__init__`[`L1043`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1043), `_gc_stopping`[`L1052`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1052), `_gc_thread`[`L1053`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1053), `_index_exists`[`L1214`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1214), `_save_policy`[`L1054`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1054), `_storage`[`L1051`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L1051)
- uses (calls/refs, reference-scoped): [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`instantiate`](config.md#Configurable.Config.instantiate), [`_add_child`](module.md#Module._add_child), [`maybe_instantiate`](config.md#maybe_instantiate), [`path`](module.md#Module.path), [`instantiate`](config.md#InstantiableConfig.instantiate), [`children`](module.md#Module.children), [`CheckpointValidationType`](checkpointer.md#CheckpointValidationType), [`InvocationContext`](module.md#InvocationContext), [`dir`](checkpointer.md#BaseCheckpointer.Config.dir), [`NestedTensorSpec`](utils.md#NestedTensorSpec), [`BaseCheckpointer`](checkpointer.md#BaseCheckpointer), [`install_context_stack`](module.md#install_context_stack), [`Config`](checkpointer.md#Checkpointer.Config), [`restore_from_dir`](checkpointer.md#StateStorage.restore_from_dir), [`save_policy`](checkpointer.md#BaseCheckpointer.Config.save_policy), [`clone_context_stack`](module.md#clone_context_stack), [`StateStorage`](checkpointer.md#StateStorage), [`dir`](summary_writer.md#BaseWriter.Config.dir), [`storage`](checkpointer.md#Checkpointer.Config.storage), [`__init__`](checkpointer.md#BaseCheckpointer.__init__), [`summary_writer`](checkpointer.md#Checkpointer.Config.summary_writer), [`save_to_dir`](checkpointer.md#StateStorage.save_to_dir), [`EXACT`](checkpointer.md#CheckpointValidationType.EXACT), [`build_step_dir`](checkpointer.md#build_step_dir), [`parse_step_from_dir`](checkpointer.md#parse_step_from_dir), [`index_writer`](checkpointer.md#Checkpointer.Config.index_writer), [`restore_state_filter`](checkpointer.md#Checkpointer.Config.restore_state_filter), [`CheckpointerAction`](summary_writer.md#CheckpointerAction), [`CheckpointPolicy`](checkpointer.md#CheckpointPolicy), [`STEP_PREFIX`](checkpointer.md#STEP_PREFIX), [`__enter__`](checkpointer.md#BaseCheckpointer.__enter__), [`latest_checkpoint_path`](checkpointer.md#BaseCheckpointer.latest_checkpoint_path), [`stop`](checkpointer.md#StateStorage.stop), [`validation`](checkpointer.md#Checkpointer.Config.validation), [`wait_until_finished`](checkpointer.md#StateStorage.wait_until_finished), [`keep_every_n_steps`](checkpointer.md#Checkpointer.Config.keep_every_n_steps), [`write_index_file`](checkpointer.md#write_index_file)  (+4 more)
- used by: [`BaseCheckpointer`](checkpointer.md#BaseCheckpointer), [`checkpointer`](trainer.md#SpmdTrainer.Config.checkpointer), [`restore`](checkpointer.md#BaseCheckpointer.restore), [`save`](checkpointer.md#BaseCheckpointer.save), [`checkpoint_paths`](checkpointer.md#BaseCheckpointer.checkpoint_paths), [`checkpoint_steps`](checkpointer.md#BaseCheckpointer.checkpoint_steps), [`stop`](checkpointer.md#BaseCheckpointer.stop), [`wait_until_finished`](checkpointer.md#BaseCheckpointer.wait_until_finished)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/checkpointer.py:970`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L970)
- doc: Configures Checkpointer.
- signature: `class Config(BaseCheckpointer.Config):`
- members:
  - `dir` — [`L819`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L819)
  - `gc_loop_interval_seconds` — [`L977`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L977)
  - `index_writer` — [`L983`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L983)
  - `keep_every_n_steps` — [`L975`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L975)
  - `keep_last_n` — [`L973`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L973)
  - `max_concurrent_gb` — [`L451`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L451)
  - `max_concurrent_restore_gb` — [`L452`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L452)
  - `max_data_shard_degree` — [`L449`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L449)
  - `metric` — [`L712`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L712)
  - `mode` — [`L716`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L716)
  - `restore_state_filter` — [`L991`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L991)
  - `save_policy` — [`L820`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L820)
  - `shard_threshold_bytes` — [`L453`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L453)
  - `storage` — [`L979`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L979)
  - `summary_writer` — [`L981`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L981)
  - `timeout_secs` — [`L448`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L448)
  - `validation` — [`L987`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L987)
- uses (calls/refs, reference-scoped): [`default_config`](config.md#Configurable.default_config), [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`InstantiableConfig`](config.md#InstantiableConfig), [`config_for_function`](config.md#config_for_function), [`ConfigOr`](config.md#ConfigOr), [`Config`](config.md#Configurable.Config), [`CheckpointValidationType`](checkpointer.md#CheckpointValidationType), [`BaseCheckpointer`](checkpointer.md#BaseCheckpointer), [`StateStorage`](checkpointer.md#StateStorage), [`Config`](checkpointer.md#BaseCheckpointer.Config), [`Config`](summary_writer.md#SummaryWriter.Config), [`SummaryWriter`](summary_writer.md#SummaryWriter), [`RestoreStateFilter`](checkpointer.md#RestoreStateFilter), [`every_n_steps_policy`](checkpointer.md#every_n_steps_policy), [`TensorStoreStateStorage`](checkpointer.md#TensorStoreStateStorage), [`CheckpointPolicy`](checkpointer.md#CheckpointPolicy), [`EvalMetric`](checkpointer.md#EvalMetric), [`IndexFileWriter`](checkpointer.md#IndexFileWriter)
- used by: [`__call__`](state_builder.md#OrbaxStateBuilder.__call__), [`restore`](checkpointer.md#Checkpointer.restore), [`restore`](checkpointer_orbax.md#OrbaxCheckpointer.restore), [`_input_iter`](trainer.md#SpmdTrainer._input_iter), [`__init__`](checkpointer_orbax.md#OrbaxCheckpointer.__init__), [`_manager`](checkpointer_orbax.md#OrbaxCheckpointer._manager), [`_run_garbage_collection`](checkpointer.md#Checkpointer._run_garbage_collection), [`_manager`](checkpointer.md#TensorStoreStateStorage._manager), [`_gc_loop`](checkpointer.md#Checkpointer._gc_loop), [`_save_policy`](checkpointer.md#Checkpointer._save_policy), [`__init__`](checkpointer.md#Checkpointer.__init__), [`ckpt_dir`](checkpointer.md#Checkpointer.ckpt_dir), [`__init__`](checkpointer.md#TensorStoreStateStorage.__init__), [`_index_writer`](checkpointer.md#Checkpointer._index_writer), [`save_fn_with_summaries`](checkpointer_orbax.md#OrbaxCheckpointer.save_fn_with_summaries), [`_storage`](checkpointer.md#Checkpointer._storage), [`_max_concurrent_restore_gb`](checkpointer.md#TensorStoreStateStorage._max_concurrent_restore_gb)

### `EvalMetric`  ·  implements/extends NamedTuple
- def: [`axlearn/common/checkpointer.py:688`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L688)
- doc: Tuple used to fetch a metric from evaler_summaries dict.
- signature: `class EvalMetric(NamedTuple):`
- members:
  - `evaler_name` — [`L694`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L694)
  - `metric_name` — [`L695`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L695)
- used by: [`metric`](checkpointer.md#BestMetricPolicy.Config.metric)

### `IndexFileWriter`  ·  implements/extends Protocol
- def: [`axlearn/common/checkpointer.py:349`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L349)
- signature: `class IndexFileWriter(Protocol):`
- protocol/private: `__call__`[`L350`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L350)
- used by: [`index_writer`](checkpointer.md#Checkpointer.Config.index_writer)

### `PythonSavable`  ·  implements/extends Protocol
- def: [`axlearn/common/checkpointer.py:257`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L257)
- doc: A Python object that implements save/restore logic.
- signature: `class PythonSavable(Protocol):`
- members:
  - `get_state(self)` — [`L263`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L263) — Gets checkpoint state.
  - `set_state(self, state: Union[bytes, Nested[Any]])` — [`L269`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L269) — Sets checkpoint state.
- uses (calls/refs, reference-scoped): [`Nested`](utils.md#Nested)
- used by: [`_get_spec`](checkpointer.md#TensorStoreStateStorage._get_spec), [`_get_spec`](checkpointer_orbax.md#OrbaxCheckpointer._get_spec), [`maybe_restore_python_savables`](checkpointer.md#maybe_restore_python_savables), [`maybe_save_python_savables`](checkpointer.md#maybe_save_python_savables)

### `StateStorage`  ·  implements/extends Configurable
- def: [`axlearn/common/checkpointer.py:311`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L311)
- doc: Base StateStorage.
- signature: `class StateStorage(Configurable):`
- members:
  - `restore_from_dir(self, step: int, state: Union[NestedTensor, NestedTensorSpec], *, ckpt_dir: str, validation: CheckpointValidationType = CheckpointValidationType.EXACT)` — [`L334`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L334)
  - `save_to_dir(self, *, step: int, state: NestedTensor, ckpt_dir: str, on_commit_callback: StateStorageCommitCallback)` — [`L314`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L314) — Starts a save to the given directories.
  - `stop(self)` — [`L344`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L344) — Stops and disposes resources.
  - `wait_until_finished(self)` — [`L330`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L330) — Waits for async writes to finish.
- uses (calls/refs, reference-scoped): [`NestedTensor`](utils.md#NestedTensor), [`Configurable`](config.md#Configurable), [`save_to_dir`](checkpointer.md#TensorStoreStateStorage.save_to_dir), [`restore_from_dir`](checkpointer.md#TensorStoreStateStorage.restore_from_dir), [`CheckpointValidationType`](checkpointer.md#CheckpointValidationType), [`NestedTensorSpec`](utils.md#NestedTensorSpec), [`EXACT`](checkpointer.md#CheckpointValidationType.EXACT), [`TensorStoreStateStorage`](checkpointer.md#TensorStoreStateStorage), [`stop`](checkpointer.md#TensorStoreStateStorage.stop), [`wait_until_finished`](checkpointer.md#TensorStoreStateStorage.wait_until_finished), [`StateStorageCommitCallback`](checkpointer.md#StateStorageCommitCallback)
- used by: [`Configurable`](config.md#Configurable), [`save`](checkpointer.md#Checkpointer.save), [`validate_and_restore`](checkpointer.md#Checkpointer.validate_and_restore), [`stop`](checkpointer.md#Checkpointer.stop), [`storage`](state_builder.md#TensorStoreStateStorageBuilder.Config.storage), [`_storage`](checkpointer.md#Checkpointer._storage), [`storage`](checkpointer.md#Checkpointer.Config.storage), [`Config`](checkpointer.md#TensorStoreStateStorage.Config), [`wait_until_finished`](checkpointer.md#Checkpointer.wait_until_finished), [`TensorStoreStateStorage`](checkpointer.md#TensorStoreStateStorage)

### `StateStorageCommitCallback`  ·  implements/extends Protocol
- def: [`axlearn/common/checkpointer.py:304`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L304)
- doc: StateStorage commit callback protocol.
- signature: `class StateStorageCommitCallback(Protocol):`
- members:
  - `__call__(self, *, ckpt_dir: str, index: Any)` — [`L307`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L307) — Commits a checkpoint with the given directory and index.
- used by: [`save_to_dir`](checkpointer.md#TensorStoreStateStorage.save_to_dir), [`save_to_dir`](checkpointer.md#StateStorage.save_to_dir)

### `TensorStoreStateStorage`  ·  implements/extends StateStorage
- def: [`axlearn/common/checkpointer.py:423`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L423)
- doc: A StateStorage implementation using TensorStore.
- signature: `class TensorStoreStateStorage(StateStorage):`
- members:
  - `commit()` — [`L591`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L591)
  - `restore_from_dir(self, step: int, state: Union[NestedTensor, NestedTensorSpec], *, ckpt_dir: str, validation: CheckpointValidationType = CheckpointValidationType.EXACT)` — [`L613`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L613)
  - `save_to_dir(self, *, step: int, state: NestedTensor, ckpt_dir: str, on_commit_callback: StateStorageCommitCallback = write_index_file)` — [`L555`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L555)
  - `stop(self)` — [`L665`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L665)
  - `wait_until_finished(self)` — [`L610`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L610)
- protocol/private: `__init__`[`L455`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L455), `_executor`[`L480`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L480), `_get_spec`[`L498`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L498), `_manager`[`L461`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L461), `_max_concurrent_restore_gb`[`L479`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L479), `_restore_tensorstore_state`[`L635`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L635), `_spec_from_path`[`L494`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L494)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`flatten_items`](utils.md#flatten_items), [`shapes`](utils.md#shapes), [`TensorSpec`](utils.md#TensorSpec), [`serialize`](array_serialization.md#BoundedDataShardedAsyncCheckpointManager.serialize), [`CheckpointValidationType`](checkpointer.md#CheckpointValidationType), [`async_save_tf_savables`](checkpointer.md#async_save_tf_savables), [`check_state_structure`](checkpointer.md#check_state_structure), `env`, [`NestedTensorSpec`](utils.md#NestedTensorSpec), [`deserialize`](array_serialization.md#GlobalAsyncCheckpointManager.deserialize), [`restore_tf_savables`](checkpointer.md#restore_tf_savables), [`maybe_restore_python_savables`](checkpointer.md#maybe_restore_python_savables), [`maybe_save_python_savables`](checkpointer.md#maybe_save_python_savables), [`StateStorage`](checkpointer.md#StateStorage), [`__init__`](config.md#Configurable.__init__), [`index`](checkpointer.md#TensorStoreStateStorage.CheckpointSpec.index), [`Config`](checkpointer.md#TensorStoreStateStorage.Config), [`EXACT`](checkpointer.md#CheckpointValidationType.EXACT), [`gda_values`](checkpointer.md#TensorStoreStateStorage.CheckpointSpec.gda_values), [`ElasticDatasetIterator`](elastic_input.md#ElasticDatasetIterator), [`python_ckpt_map`](checkpointer.md#TensorStoreStateStorage.CheckpointSpec.python_ckpt_map), [`tf_ckpt_map`](checkpointer.md#TensorStoreStateStorage.CheckpointSpec.tf_ckpt_map), [`GlobalAsyncCheckpointManager`](array_serialization.md#GlobalAsyncCheckpointManager), [`PythonSavable`](checkpointer.md#PythonSavable), [`tensorstore_specs`](checkpointer.md#TensorStoreStateStorage.CheckpointSpec.tensorstore_specs), [`BoundedDataShardedAsyncCheckpointManager`](array_serialization.md#BoundedDataShardedAsyncCheckpointManager), [`read_index_file`](checkpointer.md#read_index_file), [`max_concurrent_restore_gb`](checkpointer.md#TensorStoreStateStorage.Config.max_concurrent_restore_gb), [`shardings`](checkpointer.md#TensorStoreStateStorage.CheckpointSpec.shardings), [`CheckpointSpec`](checkpointer.md#TensorStoreStateStorage.CheckpointSpec), [`dtypes`](checkpointer.md#TensorStoreStateStorage.CheckpointSpec.dtypes), [`shapes`](checkpointer.md#TensorStoreStateStorage.CheckpointSpec.shapes), [`shard_threshold_bytes`](checkpointer.md#TensorStoreStateStorage.Config.shard_threshold_bytes), [`storage_paths`](checkpointer.md#TensorStoreStateStorage.CheckpointSpec.storage_paths), [`StateStorageCommitCallback`](checkpointer.md#StateStorageCommitCallback), [`max_concurrent_gb`](checkpointer.md#TensorStoreStateStorage.Config.max_concurrent_gb), [`max_data_shard_degree`](checkpointer.md#TensorStoreStateStorage.Config.max_data_shard_degree)  (+2 more)
- used by: [`restore_from_dir`](checkpointer.md#StateStorage.restore_from_dir), [`storage`](state_builder.md#TensorStoreStateStorageBuilder.Config.storage), [`StateStorage`](checkpointer.md#StateStorage), [`storage`](checkpointer.md#Checkpointer.Config.storage), [`save_to_dir`](checkpointer.md#StateStorage.save_to_dir), [`stop`](checkpointer.md#StateStorage.stop), [`wait_until_finished`](checkpointer.md#StateStorage.wait_until_finished)

## Functions
- `_parse_tensor_spec(spec_dict: dict[str, str])` — [`L367`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L367)
- `_upload_dir(src_dir_handle: tempfile.TemporaryDirectory, *, dst_dir: str)` — [`L181`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L181) — Upload a directory (non-recursively) from a temporary dir to dst_dir.
- `async_save_tf_savables(value_map: Nested[Any], *, executor: futures.ThreadPoolExecutor, dir: str)` — [`L203`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L203) — Asynchronously saves TF savables from `value_map` into `dir`.
- `build_step_dir(base_dir: str, *, step: int)` — [`L122`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L122) — Returns the path of checkpoint at `step` under `base_dir`.
- `check_state_structure(ckpt_structure: list[tuple[str, Any]], target_structure: list[tuple[str, Any]], *, validation: CheckpointValidationType = CheckpointValidationType.EXACT)` — [`L127`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L127)
- `drop_learner_optimizer_cfg()` — [`L93`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L93) — Returns a config that instantiates to a filter dropping `learner/optimizer` from state.
- `every_n_steps_and_last_policy(n: int = 1, *, min_step: int = 1, max_step: int)` — [`L780`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L780) — Checkpoints every n steps, but not before `min_step`, and at the last training iteration
- `every_n_steps_policy(n: int = 1, *, min_step: int = 1)` — [`L770`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L770) — Checkpoints every n steps, but not before `min_step`.
- `filter_fn(state)` — [`L102`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L102)
- `filter_for_validation(structure)` — [`L134`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L134)
- `fn()` — [`L101`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L101)
- `fn(*, step: int, evaler_summaries: dict[str, Any])` — [`L773`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L773)
- `fn(*, step: int, evaler_summaries: dict[str, Any])` — [`L794`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L794)
- `maybe_restore_python_savables(value_map: Nested[Any], *, dir: str)` — [`L290`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L290) — Restores Python savables from `dir` into `value_map`.
- `maybe_save_python_savables(value_map: Nested[Any], *, dir: str)` — [`L277`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L277) — Saves python savables from `value_map` into `dir`.
- `parse_step_from_dir(step_dir: str)` — [`L117`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L117)
- `read_index_file(ckpt_dir: str)` — [`L361`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L361) — Reads index files written with `write_index_file`.
- `read_state_spec(ckpt_dir: str)` — [`L393`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L393) — Reads TensorSpecs from the given checkpoint dir.
- `restore_tf_savables(value_map: Nested[Any], *, dir: str)` — [`L233`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L233) — Restores TF savables from `dir` into `value_map` in-place.
- `write_index_file(*, ckpt_dir: str, index: Any)` — [`L353`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L353) — An on_commit_callback that writes an index file to ckpt_dir.

## Module values
- `RestoreStateFilter` — [`L88`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L88)
- `STEP_NUM_DIGITS` — [`L61`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L61)
- `STEP_PREFIX` — [`L63`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer.py#L63)

