---
title: 'Module: axlearn/common/checkpointer_orbax.py'
type: catalog
provenance: extracted
module: axlearn/common/checkpointer_orbax.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.checkpointer_orbax`/
symbols:
  OrbaxCheckpointer._restore_args: OrbaxCheckpointer#_restore_args().
  OrbaxCheckpointer.restore: OrbaxCheckpointer#restore().
  OrbaxCheckpointer.__init__: OrbaxCheckpointer#__init__().
  OrbaxCheckpointer._manager: OrbaxCheckpointer#_manager.
  _GrainDatasetIteratorHandler.deserialize: _GrainDatasetIteratorHandler#deserialize().
  _TfIteratorHandler.deserialize: _TfIteratorHandler#deserialize().
  OrbaxCheckpointer._get_spec: OrbaxCheckpointer#_get_spec().
  OrbaxCheckpointer.save: OrbaxCheckpointer#save().
  OrbaxCheckpointer.save_fn_with_summaries: OrbaxCheckpointer#save_fn_with_summaries().
  OrbaxCheckpointer.Config: OrbaxCheckpointer#Config#
  _TfIteratorHandler.serialize: _TfIteratorHandler#serialize().
  _GrainDatasetIteratorHandler.serialize: _GrainDatasetIteratorHandler#serialize().
  OrbaxCheckpointer: OrbaxCheckpointer#
  OrbaxCheckpointer._name_format: OrbaxCheckpointer#_name_format.
  OrbaxCheckpointer.Config.validation_type: OrbaxCheckpointer#Config#validation_type.
  _CheckpointManagerWithTrackerFile.tracker_file_path: _CheckpointManagerWithTrackerFile#tracker_file_path().
  _GRAIN_INSTALLED: _GRAIN_INSTALLED.
  _GrainDatasetIteratorHandler.RestoreArgs.item: _GrainDatasetIteratorHandler#RestoreArgs#item.
  _CheckpointManagerWithTrackerFile._finalize_checkpoint: _CheckpointManagerWithTrackerFile#_finalize_checkpoint().
  OrbaxCheckpointer._eval_summaries: OrbaxCheckpointer#_eval_summaries.
  OrbaxCheckpointer.ckpt_dir: OrbaxCheckpointer#ckpt_dir().
  _TfIteratorHandler._ckpt_dir: _TfIteratorHandler#_ckpt_dir().
  _TfIteratorHandler._get_or_create_tf_ckpt: _TfIteratorHandler#_get_or_create_tf_ckpt().
  _replica_devices: _replica_devices().
  _GrainIterator: _GrainIterator.
  _TfIteratorHandler._tf_ckpt_cache: _TfIteratorHandler#_tf_ckpt_cache.
  _TfIteratorHandler.RestoreArgs: _TfIteratorHandler#RestoreArgs#
  _TfIteratorHandler.RestoreArgs.item: _TfIteratorHandler#RestoreArgs#item.
  _GrainDatasetIteratorHandler.RestoreArgs: _GrainDatasetIteratorHandler#RestoreArgs#
  _TfIteratorHandler.metadata: _TfIteratorHandler#metadata().
  _GrainDatasetIteratorHandler.metadata: _GrainDatasetIteratorHandler#metadata().
  _CheckpointManagerWithTrackerFile._load_checkpoint_infos: _CheckpointManagerWithTrackerFile#_load_checkpoint_infos().
  OrbaxCheckpointer.wait_until_finished: OrbaxCheckpointer#wait_until_finished().
  OrbaxCheckpointer.stop: OrbaxCheckpointer#stop().
  _TfIteratorHandler: _TfIteratorHandler#
  _TfIteratorHandler._executor: _TfIteratorHandler#_executor.
  _TfIteratorHandler._sync_tf_ckpt_and_check_error: _TfIteratorHandler#_sync_tf_ckpt_and_check_error().
  _GrainDatasetIteratorHandler: _GrainDatasetIteratorHandler#
  _GrainDatasetIteratorHandler._executor: _GrainDatasetIteratorHandler#_executor.
  _GrainDatasetIteratorHandler._ckpt_dir: _GrainDatasetIteratorHandler#_ckpt_dir().
  OrbaxCheckpointer.Config.max_concurrent_restore_gb: OrbaxCheckpointer#Config#max_concurrent_restore_gb.
  OrbaxCheckpointer.Config.enable_single_replica_ckpt_restoring: OrbaxCheckpointer#Config#enable_single_replica_ckpt_restoring.
  OrbaxCheckpointer.Config.replica_axis_index: OrbaxCheckpointer#Config#replica_axis_index.
  _CheckpointManagerWithTrackerFile: _CheckpointManagerWithTrackerFile#
  _CheckpointManagerWithTrackerFile._tracker_filename: _CheckpointManagerWithTrackerFile#_tracker_filename.
  OrbaxCheckpointer.Config.keep_last_n: OrbaxCheckpointer#Config#keep_last_n.
  OrbaxCheckpointer.Config.keep_period: OrbaxCheckpointer#Config#keep_period.
  OrbaxCheckpointer.Config.async_timeout_secs: OrbaxCheckpointer#Config#async_timeout_secs.
  OrbaxCheckpointer.Config.max_concurrent_save_gb: OrbaxCheckpointer#Config#max_concurrent_save_gb.
  OrbaxCheckpointer.Config.skip_uncommitted_checkpoint: OrbaxCheckpointer#Config#skip_uncommitted_checkpoint.
  OrbaxCheckpointer.Config.read_latest_checkpoint_from_tracker_file: OrbaxCheckpointer#Config#read_latest_checkpoint_from_tracker_file.
  _find_idx: _find_idx().
  _TfIteratorHandler.__init__: _TfIteratorHandler#__init__().
  _TfIteratorHandler.typestr: _TfIteratorHandler#typestr().
  _GrainDatasetIteratorHandler.__init__: _GrainDatasetIteratorHandler#__init__().
  _GrainDatasetIteratorHandler.typestr: _GrainDatasetIteratorHandler#typestr().
  _CheckpointManagerWithTrackerFile.__init__: _CheckpointManagerWithTrackerFile#__init__().
  OrbaxCheckpointer.checkpoint_paths: OrbaxCheckpointer#checkpoint_paths().
  OrbaxCheckpointer.checkpoint_steps: OrbaxCheckpointer#checkpoint_steps().
---
# Module: [`axlearn/common/checkpointer_orbax.py`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py)

## Classes
### `Config`  ·  implements/extends Config
- def: [`axlearn/common/checkpointer_orbax.py:344`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L344)
- doc: Configures OrbaxCheckpointer.
- signature: `class Config(BaseCheckpointer.Config):`
- members:
  - `async_timeout_secs` — [`L356`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L356)
  - `enable_single_replica_ckpt_restoring` — [`L362`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L362)
  - `keep_last_n` — [`L348`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L348)
  - `keep_period` — [`L351`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L351)
  - `max_concurrent_restore_gb` — [`L358`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L358)
  - `max_concurrent_save_gb` — [`L357`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L357)
  - `read_latest_checkpoint_from_tracker_file` — [`L370`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L370)
  - `replica_axis_index` — [`L364`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L364)
  - `skip_uncommitted_checkpoint` — [`L368`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L368)
  - `validation_type` — [`L353`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L353)
- uses (calls/refs, reference-scoped): [`config_class`](config.md#config_class), [`CheckpointValidationType`](checkpointer.md#CheckpointValidationType), [`BaseCheckpointer`](checkpointer.md#BaseCheckpointer), [`Config`](checkpointer.md#BaseCheckpointer.Config), [`EXACT`](checkpointer.md#CheckpointValidationType.EXACT)
- used by: [`__call__`](state_builder.md#OrbaxStateBuilder.__call__), [`_restore_args`](checkpointer_orbax.md#OrbaxCheckpointer._restore_args), [`restore`](checkpointer_orbax.md#OrbaxCheckpointer.restore), [`__init__`](checkpointer_orbax.md#OrbaxCheckpointer.__init__), [`_manager`](checkpointer_orbax.md#OrbaxCheckpointer._manager), [`save_fn_with_summaries`](checkpointer_orbax.md#OrbaxCheckpointer.save_fn_with_summaries)

### `OrbaxCheckpointer`  ·  implements/extends BaseCheckpointer
- def: [`axlearn/common/checkpointer_orbax.py:329`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L329)
- doc: A checkpointer that uses orbax CheckpointManager.
- signature: `class OrbaxCheckpointer(BaseCheckpointer):`
- members:
  - `checkpoint_paths(cls, base_dir: str)` — [`L373`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L373) — See `BaseCheckpointer.checkpointer_paths`.
  - `checkpoint_steps(cls, base_dir)` — [`L378`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L378) — See `BaseCheckpointer.checkpointer_steps`.
  - `ckpt_dir(self, step: int, dir: Optional[str] = None)` — [`L488`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L488) — Obtains the checkpoint dir for the given step.
  - `restore(self, *, step: Optional[int] = None, state: Union[Nested[Tensor], Nested[TensorSpec]])` — [`L531`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L531) — See `BaseCheckpointer.restore` for details.
  - `save(self, *, step: int, state: Nested[Tensor], evaler_summaries: Optional[Dict[str, Any]] = None)` — [`L494`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L494) — See `BaseCheckpointer.save` for details.
  - `save_fn_with_summaries(step: int, last_saved_step: Optional[int])` — [`L415`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L415)
  - `stop(self, *, has_exception: bool = False)` — [`L614`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L614) — See `BaseCheckpointer.stop` for details.
  - `wait_until_finished(self)` — [`L610`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L610) — See `BaseCheckpointer.wait_until_finished` docstring for details.
- protocol/private: `__init__`[`L382`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L382), `_eval_summaries`[`L407`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L407), `_get_spec`[`L473`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L473), `_manager`[`L439`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L439), `_name_format`[`L409`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L409), `_restore_args`[`L541`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L541)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`flatten_items`](utils.md#flatten_items), [`TensorSpec`](utils.md#TensorSpec), [`instantiate`](config.md#InstantiableConfig.instantiate), [`check_state_structure`](checkpointer.md#check_state_structure), [`dir`](checkpointer.md#BaseCheckpointer.Config.dir), [`BaseCheckpointer`](checkpointer.md#BaseCheckpointer), [`save_policy`](checkpointer.md#BaseCheckpointer.Config.save_policy), [`Config`](checkpointer_orbax.md#OrbaxCheckpointer.Config), [`__init__`](checkpointer.md#BaseCheckpointer.__init__), [`validation_type`](checkpointer_orbax.md#OrbaxCheckpointer.Config.validation_type), [`_GRAIN_INSTALLED`](checkpointer_orbax.md#_GRAIN_INSTALLED), [`item`](checkpointer_orbax.md#_GrainDatasetIteratorHandler.RestoreArgs.item), [`PythonSavable`](checkpointer.md#PythonSavable), [`STEP_NUM_DIGITS`](checkpointer.md#STEP_NUM_DIGITS), [`STEP_PREFIX`](checkpointer.md#STEP_PREFIX), [`_replica_devices`](checkpointer_orbax.md#_replica_devices), [`RestoreArgs`](checkpointer_orbax.md#_GrainDatasetIteratorHandler.RestoreArgs), [`RestoreArgs`](checkpointer_orbax.md#_TfIteratorHandler.RestoreArgs), [`_GrainIterator`](checkpointer_orbax.md#_GrainIterator), [`item`](checkpointer_orbax.md#_TfIteratorHandler.RestoreArgs.item), [`_GrainDatasetIteratorHandler`](checkpointer_orbax.md#_GrainDatasetIteratorHandler), [`_TfIteratorHandler`](checkpointer_orbax.md#_TfIteratorHandler), [`enable_single_replica_ckpt_restoring`](checkpointer_orbax.md#OrbaxCheckpointer.Config.enable_single_replica_ckpt_restoring), [`max_concurrent_restore_gb`](checkpointer_orbax.md#OrbaxCheckpointer.Config.max_concurrent_restore_gb), [`replica_axis_index`](checkpointer_orbax.md#OrbaxCheckpointer.Config.replica_axis_index), [`_CheckpointManagerWithTrackerFile`](checkpointer_orbax.md#_CheckpointManagerWithTrackerFile), [`async_timeout_secs`](checkpointer_orbax.md#OrbaxCheckpointer.Config.async_timeout_secs), [`keep_last_n`](checkpointer_orbax.md#OrbaxCheckpointer.Config.keep_last_n), [`keep_period`](checkpointer_orbax.md#OrbaxCheckpointer.Config.keep_period), [`max_concurrent_save_gb`](checkpointer_orbax.md#OrbaxCheckpointer.Config.max_concurrent_save_gb), [`read_latest_checkpoint_from_tracker_file`](checkpointer_orbax.md#OrbaxCheckpointer.Config.read_latest_checkpoint_from_tracker_file), [`skip_uncommitted_checkpoint`](checkpointer_orbax.md#OrbaxCheckpointer.Config.skip_uncommitted_checkpoint)
- used by: [`__call__`](state_builder.md#OrbaxStateBuilder.__call__), [`BaseCheckpointer`](checkpointer.md#BaseCheckpointer), [`restore`](checkpointer.md#BaseCheckpointer.restore), [`save`](checkpointer.md#BaseCheckpointer.save), [`checkpoint_paths`](checkpointer.md#BaseCheckpointer.checkpoint_paths), [`checkpoint_steps`](checkpointer.md#BaseCheckpointer.checkpoint_steps), [`stop`](checkpointer.md#BaseCheckpointer.stop), [`wait_until_finished`](checkpointer.md#BaseCheckpointer.wait_until_finished)

### `RestoreArgs`  ·  implements/extends RestoreArgs
- def: [`axlearn/common/checkpointer_orbax.py:185`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L185)
- signature: `class RestoreArgs(ocp.type_handlers.RestoreArgs):`
- members:
  - `item` — [`L78`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L78)
  - `item` — [`L186`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L186)
- uses (calls/refs, reference-scoped): [`_GrainIterator`](checkpointer_orbax.md#_GrainIterator)
- used by: [`_restore_args`](checkpointer_orbax.md#OrbaxCheckpointer._restore_args), [`deserialize`](checkpointer_orbax.md#_GrainDatasetIteratorHandler.deserialize), [`deserialize`](checkpointer_orbax.md#_TfIteratorHandler.deserialize)

### `_CheckpointManagerWithTrackerFile`  ·  implements/extends CheckpointManager
- def: [`axlearn/common/checkpointer_orbax.py:253`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L253)
- doc: In some extreme cases, the number of available checkpoints may be quite
- signature: `class _CheckpointManagerWithTrackerFile(ocp.CheckpointManager):`
- members:
  - `_load_checkpoint_infos(self, skip_metadata_read=False)` — [`L276`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L276) — The original version looks up in the root directory and return all
  - `tracker_file_path(self)` — [`L273`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L273)
- protocol/private: `__init__`[`L268`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L268), `_finalize_checkpoint`[`L317`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L317), `_tracker_filename`[`L269`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L269)
- uses (calls/refs, reference-scoped): `_multiprocessing_options`
- used by: [`__init__`](checkpointer_orbax.md#OrbaxCheckpointer.__init__)

### `_GrainDatasetIteratorHandler`  ·  implements/extends TypeHandler
- def: [`axlearn/common/checkpointer_orbax.py:175`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L175)
- doc: Serializes grain dataset iterators.
- signature: `class _GrainDatasetIteratorHandler(ocp.type_handlers.TypeHandler):`
- members:
  - `deserialize(self, infos: Sequence[ocp.type_handlers.ParamInfo], args: Optional[Sequence[RestoreArgs]] = None)` — [`L219`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L219)
  - `metadata(self, infos: Sequence[ocp.type_handlers.ParamInfo])` — [`L243`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L243)
  - `serialize(self, values: Sequence[grain.DatasetIterator], infos: Sequence[ocp.type_handlers.ParamInfo], args: Optional[Sequence[ocp.args.PyTreeSave]])` — [`L199`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L199) — Serializes `values` into corresponding `info.path`s.
  - `typestr(self)` — [`L188`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L188)
- protocol/private: `__init__`[`L178`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L178), `_ckpt_dir`[`L191`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L191), `_executor`[`L180`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L180)
- uses (calls/refs, reference-scoped): [`maybe_restore_python_savables`](checkpointer.md#maybe_restore_python_savables), [`maybe_save_python_savables`](checkpointer.md#maybe_save_python_savables), [`item`](checkpointer_orbax.md#_GrainDatasetIteratorHandler.RestoreArgs.item), `name`, [`RestoreArgs`](checkpointer_orbax.md#_GrainDatasetIteratorHandler.RestoreArgs), [`_GrainIterator`](checkpointer_orbax.md#_GrainIterator)
- used by: [`_restore_args`](checkpointer_orbax.md#OrbaxCheckpointer._restore_args)

### `_TfIteratorHandler`  ·  implements/extends TypeHandler
- def: [`axlearn/common/checkpointer_orbax.py:60`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L60)
- doc: Serializes tf.data.Iterator.
- signature: `class _TfIteratorHandler(ocp.type_handlers.TypeHandler):`
- members:
  - `deserialize(self, infos: Sequence[ocp.type_handlers.ParamInfo], args: Optional[Sequence[RestoreArgs]] = None)` — [`L130`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L130)
  - `metadata(self, infos: Sequence[ocp.type_handlers.ParamInfo])` — [`L155`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L155)
  - `serialize(self, values: Sequence[tf.data.Iterator], infos: Sequence[ocp.type_handlers.ParamInfo], args: Optional[Sequence[ocp.args.PyTreeSave]])` — [`L111`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L111) — Serializes `values` into corresponding `info.path`s.
  - `typestr(self)` — [`L80`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L80)
- protocol/private: `__init__`[`L70`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L70), `_ckpt_dir`[`L83`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L83), `_executor`[`L72`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L72), `_get_or_create_tf_ckpt`[`L89`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L89), `_sync_tf_ckpt_and_check_error`[`L97`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L97), `_tf_ckpt_cache`[`L73`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L73)
- uses (calls/refs, reference-scoped): `name`, [`RestoreArgs`](checkpointer_orbax.md#_TfIteratorHandler.RestoreArgs), [`item`](checkpointer_orbax.md#_TfIteratorHandler.RestoreArgs.item)
- used by: [`_restore_args`](checkpointer_orbax.md#OrbaxCheckpointer._restore_args)

## Functions
- `_find_idx(array: np.ndarray, replica_axis_idx: int)` — [`L623`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L623) — Returns the index along given dimension that the current host belongs to.
- `_replica_devices(device_array: np.ndarray, replica_axis_idx: int)` — [`L632`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L632) — Returns the devices from the replica that current host belongs to.

## Module values
- `_GRAIN_INSTALLED` — [`L54`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L54)
- `_GrainIterator` — [`L53`](../../../../../../raw/code/axlearn/axlearn/common/checkpointer_orbax.py#L53)

