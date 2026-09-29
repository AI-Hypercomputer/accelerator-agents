---
title: 'Module: src/maxtext/common/checkpointing.py'
type: catalog
provenance: extracted
module: src/maxtext/common/checkpointing.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.common.checkpointing`/
symbols:
  load_state_if_possible: load_state_if_possible().
  _restore_grain_iterator: _restore_grain_iterator().
  save_checkpoint: save_checkpoint().
  _assert_no_shaped_dtype_struct: _assert_no_shaped_dtype_struct().
  create_orbax_checkpoint_manager: create_orbax_checkpoint_manager().
  _prepare_scaled_down_grain_restore_args: _prepare_scaled_down_grain_restore_args().
  Composite: Composite.
  _populate_pure_dict_from_partial: _populate_pure_dict_from_partial().
  create_orbax_emergency_checkpoint_manager: create_orbax_emergency_checkpoint_manager().
  handle_checkpoint_mismatch: handle_checkpoint_mismatch().
  process_replicator_error_file: process_replicator_error_file().
  GrainCheckpointRestore: GrainCheckpointRestore#
  _restore_emergency_linen_checkpoint_into_nnx: _restore_emergency_linen_checkpoint_into_nnx().
  _load_full_state_from_path: _load_full_state_from_path().
  GrainCheckpointSave: GrainCheckpointSave#
  GrainCheckpointSave.item: GrainCheckpointSave#item.
  EmergencyCheckpointManager: EmergencyCheckpointManager.
  EmergencyReplicatorCheckpointManager: EmergencyReplicatorCheckpointManager.
  maybe_save_checkpoint: maybe_save_checkpoint().
  _load_linen_checkpoint_into_nnx: _load_linen_checkpoint_into_nnx().
  load_state_if_possible.map_to_pspec: load_state_if_possible().map_to_pspec().
  _default_for_sds: _default_for_sds().
  _load_linen_params_into_nnx: _load_linen_params_into_nnx().
  replicator_error_handler: replicator_error_handler().
  _replica_devices: _replica_devices().
  load_params_from_path: load_params_from_path().
  GrainCheckpointHandler: GrainCheckpointHandler#
  GrainCheckpointHandler.save: GrainCheckpointHandler#save().
  GrainCheckpointHandler.restore: GrainCheckpointHandler#restore().
  create_orbax_emergency_replicator_checkpoint_manager: create_orbax_emergency_replicator_checkpoint_manager().
  CheckpointManager: CheckpointManager.
  GrainCheckpointHandler.save_single_process: GrainCheckpointHandler#save_single_process().
  GrainCheckpointHandler.restore_single_process: GrainCheckpointHandler#restore_single_process().
  GrainCheckpointRestore.item: GrainCheckpointRestore#item.
  GrainCheckpointRestore.process_index: GrainCheckpointRestore#process_index.
  GrainCheckpointRestore.process_count: GrainCheckpointRestore#process_count.
  _default_for_sds._make: _default_for_sds()._make().
  CheckpointManagerOptions: CheckpointManagerOptions.
  PyTreeCheckpointHandler: PyTreeCheckpointHandler.
  LocalCheckpointOptions: LocalCheckpointOptions.
  PersistentCheckpointOptions: PersistentCheckpointOptions.
  _rebuild_nnx_with_values: _rebuild_nnx_with_values().
  _load_full_state_from_path.combine_sharding: _load_full_state_from_path().combine_sharding().
  read_replicator_error_file: read_replicator_error_file().
  cleanup_replicator_error_file: cleanup_replicator_error_file().
  print_save_message: print_save_message().
  _find_idx: _find_idx().
  is_structural_or_shape_mismatch: is_structural_or_shape_mismatch().
  setup_checkpoint_logger: setup_checkpoint_logger().
  save_params_to_path: save_params_to_path().
  load_checkpoint_metadata: load_checkpoint_metadata().
---
# Module: [`src/maxtext/common/checkpointing.py`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py)

## Classes
### `GrainCheckpointHandler`  ·  implements/extends CheckpointHandler
- def: [`src/maxtext/common/checkpointing.py:66`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L66)
- doc: A CheckpointHandler that allows specifying process_index and process_count.
- signature: `class GrainCheckpointHandler(PyGrainCheckpointHandler, ocp.CheckpointHandler):`
- members:
  - `restore(self, directory: epath.Path, item: Optional[Any] = None, args: Any = None)` — [`L111`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L111) — Restores the given iterator from the checkpoint in `directory`.
  - `restore_single_process(item, process_index, process_count)` — [`L137`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L137)
  - `save(self, directory: epath.Path, item: Optional[Any] = None, args: Any = None)` — [`L69`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L69) — Saves the given iterator to the checkpoint in `directory`.
  - `save_single_process(item, process_index, process_count)` — [`L96`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L96)
- used by: [`create_orbax_checkpoint_manager`](checkpointing.md#create_orbax_checkpoint_manager), [`GrainCheckpointSave`](checkpointing.md#GrainCheckpointSave), [`item`](checkpointing.md#GrainCheckpointSave.item)

### `GrainCheckpointRestore`  ·  implements/extends CheckpointArgs
- def: [`src/maxtext/common/checkpointing.py:168`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L168)
- signature: `class GrainCheckpointRestore(ocp.args.CheckpointArgs):`
- members:
  - `item` — [`L169`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L169)
  - `process_count` — [`L171`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L171)
  - `process_index` — [`L170`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L170)
- used by: [`_restore_grain_iterator`](checkpointing.md#_restore_grain_iterator), [`_prepare_scaled_down_grain_restore_args`](checkpointing.md#_prepare_scaled_down_grain_restore_args)

### `GrainCheckpointSave`  ·  implements/extends CheckpointArgs
- def: [`src/maxtext/common/checkpointing.py:162`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L162)
- signature: `class GrainCheckpointSave(ocp.args.CheckpointArgs):`
- members:
  - `item` — [`L163`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L163)
- uses (calls/refs, reference-scoped): [`GrainCheckpointHandler`](checkpointing.md#GrainCheckpointHandler)
- used by: [`save_checkpoint`](checkpointing.md#save_checkpoint)

## Functions
- `_assert_no_shaped_dtype_struct(pytree)` — [`L743`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L743) — Asserts that there are no jax.ShapeDtypeStruct leaves in the restored pytree.
- `_default_for_sds(sds)` — [`L174`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L174) — Returns a deterministic value matching `sds` shape/dtype/sharding.
- `_find_idx(array: np.ndarray, replica_axis_idx: int)` — [`L590`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L590) — Returns the index along given dimension that the current host belongs to.
- `_load_full_state_from_path(path, abstract_unboxed_pre_state, enable_orbax_v1, checkpoint_conversion_fn, source_checkpoint_layout, checkpoint_storage_concurrent_gb, use_ocdbt, use_zarr3)` — [`L313`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L313) — Load full state from checkpoint at specified path.
- `_load_linen_checkpoint_into_nnx(path, abstract_nnx_state, checkpoint_storage_concurrent_gb, use_ocdbt, use_zarr3)` — [`L218`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L218) — Restores a Linen-layout checkpoint into an NNX state (pure_nnx resume).
- `_load_linen_params_into_nnx(path, nnx_params_abstract, checkpoint_storage_concurrent_gb, use_ocdbt, use_zarr3)` — [`L281`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L281) — Weight-only load of a Linen-layout checkpoint into an NNX params state.
- `_make()` — [`L186`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L186)
- `_populate_pure_dict_from_partial(abstract_pure, partial_concrete)` — [`L198`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L198) — Fills `abstract_pure` with values from `partial_concrete` (by path), defaulting the rest.
- `_prepare_scaled_down_grain_restore_args(data_iterator: list, process_count_jax: int, process_count_stored: int, directory: epath.Path)` — [`L616`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L616) — Prepares the restore arguments for a scaled-up (list) data iterator.
- `_rebuild_nnx_with_values(abstract_nnx_state, concrete_weights)` — [`L269`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L269) — Fills each Variable in `abstract_nnx_state` with the matching restored array.
- `_replica_devices(device_array: np.ndarray, replica_axis_idx: int)` — [`L599`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L599) — Returns the devices from the replica that current host belongs to.
- `_restore_emergency_linen_checkpoint_into_nnx(checkpoint_manager, step, abstract_nnx_state, map_to_pspec)` — [`L248`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L248) — Restores an emergency Linen-layout checkpoint into an NNX state.
- `_restore_grain_iterator(checkpoint_manager, step: int, data_iterator, checkpoint_args, expansion_factor_real_data: int)` — [`L649`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L649) — Handles the complex logic for restoring a Grain data iterator checkpoint.
- `cleanup_replicator_error_file(error_file: str)` — [`L575`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L575) — Clean up replicator errors file.
- `combine_sharding(sds, shardings)` — [`L354`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L354)
- `create_orbax_checkpoint_manager(checkpoint_dir: str, enable_checkpointing: bool, use_async: bool, save_interval_steps: int, dataset_type: None | str = None, orbax_logger: Any = None, use_ocdbt: bool = True, use_zarr3: bool = True, enable_continuous_checkpointing: bool = False, max_num_checkpoints_to_keep: int = 10, checkpoint_storage_concurrent_gb: int = 96, enable_single_controller: bool = False, colocated_python_checkpointing: bool = False, enable_single_replica_ckpt_restoring: bool = False, enable_autocheckpoint: bool = False, todelete_subdir: str | None = None, todelete_full_path: str | None = None)` — [`L395`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L395) — Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled.
- `create_orbax_emergency_checkpoint_manager(local_checkpoint_dir: str, persistent_checkpoint_dir: str, global_mesh: jax.sharding.Mesh, abstract_state: Any, local_save_interval_steps: int, persistent_save_interval_steps: int, orbax_logger: Any = None)` — [`L480`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L480) — Returns an emergency checkpoint manager.
- `create_orbax_emergency_replicator_checkpoint_manager(local_checkpoint_dir: str, save_interval_steps: int, global_mesh: jax.sharding.Mesh, colocated_python_checkpointing: bool = False)` — [`L518`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L518) — Returns an emergency replicator checkpoint manager.
- `handle_checkpoint_mismatch(context_name: str, path: str)` — [`L766`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L766) — Context manager to intercept PyTree/shape mismatches and raise descriptive errors.
- `is_structural_or_shape_mismatch(e: Exception)` — [`L724`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L724) — Helper to check if an exception is likely a PyTree structure or shape mismatch.
- `load_checkpoint_metadata(checkpoint_dir_path: str)` — [`L1065`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L1065) — Loads custom metadata from an Orbax checkpoint.
- `load_params_from_path(load_parameters_from_path, abstract_unboxed_params, checkpoint_storage_concurrent_gb, use_ocdbt=True, use_zarr3=True)` — [`L1009`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L1009) — Load decode params from checkpoint at specified path.
- `load_state_if_possible(checkpoint_manager: CheckpointManager | None, data_iterator: MultiHostDataLoadIterator | list[MultiHostDataLoadIterator] | None, load_parameters_from_path: str, load_full_state_from_path: str, checkpoint_storage_concurrent_gb: int, abstract_unboxed_pre_state: train_state.TrainState | nnx.State, enable_single_replica_ckpt_restoring: bool | None = False, dataset_type: str | None = "tfds", step: int = -1, use_ocdbt=True, use_zarr3=True, enable_orbax_v1=False, checkpoint_conversion_fn=None, source_checkpoint_layout="orbax", expansion_factor_real_data: int = -1, maxtext_config: Any | None = None)` — [`L781`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L781) — Loads TrainState as possible from the inputs.
- `map_to_pspec(data)` — [`L832`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L832)
- `maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator, step=None)` — [`L1085`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L1085) — Save checkpoint if checkpointing is enabled.
- `print_save_message(step, async_checkpointing)` — [`L583`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L583)
- `process_replicator_error_file(error_file: str)` — [`L555`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L555) — Handles replicator errors by reading, logging, cleaning the error file.
- `read_replicator_error_file(error_file: str)` — [`L566`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L566) — Read replicator errors file.
- `replicator_error_handler(config: Any)` — [`L541`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L541) — Replicator error handler to handle errors in replicator service.
- `save_checkpoint(checkpoint_manager, step, state, config=None, data_iterator=None, force=False)` — [`L1154`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L1154) — Wrapper for saving checkpoint.
- `save_params_to_path(checkpoint_dir, params, use_ocdbt=True, use_zarr3=True)` — [`L1056`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L1056) — Save decode params in checkpoint at specified path.
- `setup_checkpoint_logger(config)` — [`L990`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L990) — Setup checkpoint logger.

## Module values
- `CheckpointManager` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L56)
- `CheckpointManagerOptions` — [`L57`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L57)
- `Composite` — [`L58`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L58)
- `EmergencyCheckpointManager` — [`L60`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L60)
- `EmergencyReplicatorCheckpointManager` — [`L63`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L63)
- `LocalCheckpointOptions` — [`L61`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L61)
- `PersistentCheckpointOptions` — [`L62`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L62)
- `PyTreeCheckpointHandler` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/common/checkpointing.py#L59)

