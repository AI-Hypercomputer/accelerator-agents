---
title: 'Module: axlearn/common/utils.py'
type: catalog
provenance: extracted
module: axlearn/common/utils.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.utils`/
symbols:
  Tensor: Tensor.
  Nested: Nested.
  NestedTensor: NestedTensor.
  TensorSpec.shape: TensorSpec#shape.
  flatten_items: flatten_items().
  TensorSpec.mesh_axes: TensorSpec#mesh_axes.
  TensorSpec.dtype: TensorSpec#dtype.
  shapes: shapes().
  TensorSpec: TensorSpec#
  with_sharding_constraint: with_sharding_constraint().
  VDict: VDict#
  tree_paths: tree_paths().
  maybe_shard: maybe_shard().
  data_partition_type_to_spec: data_partition_type_to_spec().
  vectorized_tree_map: vectorized_tree_map().
  host_to_global_array: host_to_global_array().
  split_prng_key: split_prng_key().
  validate_contains_paths: validate_contains_paths().
  NestedTree: NestedTree.
  create_device_mesh: create_device_mesh().
  canonicalize_per_param_dtype: canonicalize_per_param_dtype().
  NestedTensorSpec: NestedTensorSpec.
  combine_remat_policies: combine_remat_policies().
  prune_tree: prune_tree().
  as_tensor: as_tensor().
  complete_partition_spec_tree: complete_partition_spec_tree().
  HybridMeshShape.ici_mesh_shape: HybridMeshShape#ici_mesh_shape.
  HybridMeshShape.dcn_mesh_shape: HybridMeshShape#dcn_mesh_shape.
  host_to_global_specs: host_to_global_specs().
  get_recursively: get_recursively().
  prune_empty: prune_empty().
  PartitionSpecType: PartitionSpecType.
  MeshShape: MeshShape.
  DataPartitionType: DataPartitionType#
  cast_floats: cast_floats().
  save_and_offload_only_these_names_regex: save_and_offload_only_these_names_regex().
  host_to_global_array.make_array: host_to_global_array().make_array().
  per_param_dtype_by_path.fn: per_param_dtype_by_path().fn().
  register_per_param_settings: register_per_param_settings().
  sequence_mask: sequence_mask().
  get_data_dir: get_data_dir().
  build_standard_mesh: build_standard_mesh().
  global_to_host_array: global_to_host_array().
  match_regex_rules: match_regex_rules().
  replace_layer_config_recursively: replace_layer_config_recursively().
  NestedPartitionSpec: NestedPartitionSpec.
  set_recursively: set_recursively().
  input_partition_spec: input_partition_spec().
  RematType: RematType.
  numeric_checks: numeric_checks().
  expand_vdicts.fn: expand_vdicts().fn().
  PHYSICAL_TO_LOGICAL_DISPATCH_KEY: PHYSICAL_TO_LOGICAL_DISPATCH_KEY.
  cast_floats_per_param: cast_floats_per_param().
  get_or_none: get_or_none().
  create_hybrid_device_mesh: create_hybrid_device_mesh().
  replace_layer_config_recursively.exit_fn: replace_layer_config_recursively().exit_fn().
  expand_vdicts: expand_vdicts().
  check_numerics: check_numerics().
  StackedKeyArray: StackedKeyArray#
  as_numpy_array: as_numpy_array().
  PerParamFn.__call__: PerParamFn#__call__().
  SavePattern: SavePattern.
  dispatch_input_batch: dispatch_input_batch().
  dispatch_input_batch.traverse_and_dispatch: dispatch_input_batch().traverse_and_dispatch().
  PerParamFn: PerParamFn#
  set_data_dir: set_data_dir().
  find_cycles._find_cycles: find_cycles()._find_cycles().
  complete_partition_spec_tree.replace_none_with_proxy: complete_partition_spec_tree().replace_none_with_proxy().
  tree_merge: tree_merge().
  _global_data_dir_stack: _global_data_dir_stack.
  default_remat_combine_fn: default_remat_combine_fn().
  get_current_abstract_or_physical_mesh: get_current_abstract_or_physical_mesh().
  host_to_global_device_array: host_to_global_device_array().
  cast_floats_per_param.cast_per_param: cast_floats_per_param().cast_per_param().
  infer_mesh_shape: infer_mesh_shape().
  HybridMeshShape: HybridMeshShape#
  RematPolicy: RematPolicy#
  push_data_dir: push_data_dir().
  pop_data_dir: pop_data_dir().
  TensorSpec.sharding: TensorSpec#sharding().
  extended_checkpoint_policies: extended_checkpoint_policies.
  own_fields: own_fields().
  per_param_dtype_by_path: per_param_dtype_by_path().
  T: T.
  _supported_float_dtypes: _supported_float_dtypes.
  MemoryKind: MemoryKind.
  offload_dots_saveable: offload_dots_saveable().
  vectorized_tree_map.vectorized_fn: vectorized_tree_map().vectorized_fn().
  global_to_host_array.get_local_array: global_to_host_array().get_local_array().
  cast_floats.cast: cast_floats().cast().
  find_cycles: find_cycles().
  _enable_numeric_checks: _enable_numeric_checks.
  HybridMeshShape.__post_init__: HybridMeshShape#__post_init__().
  HybridMeshShape.__len__: HybridMeshShape#__len__().
  runtime_checks: runtime_checks().
  StackedKeyArray.keys: StackedKeyArray#keys.
  check_param_shape_alignment: check_param_shape_alignment().
  DataDirStack.stack: DataDirStack#stack.
  TensorSpec.memory_kind: TensorSpec#memory_kind.
  combine_remat_policies.convert_to_enum: combine_remat_policies().convert_to_enum().
  runtime_checks.switch: runtime_checks().switch().
  numeric_checks.switch: numeric_checks().switch().
  replicate_to_local_data: replicate_to_local_data().
  count_model_params: count_model_params().
  raise_for_cycles: raise_for_cycles().
  safe_not: safe_not().
  _key_entry_to_str: _key_entry_to_str().
  DataPartitionType.FULL: DataPartitionType#FULL.
  partial_with_fn_metadata: partial_with_fn_metadata().
  DeviceUsage: DeviceUsage#
  get_tpu_dot_precision: get_tpu_dot_precision().
  default_remat_combine_fn.combine_fn: default_remat_combine_fn().combine_fn().
  combine_remat_policies.policy: combine_remat_policies().policy().
  copy_recursively: copy_recursively().
  validate_float_dtype: validate_float_dtype().
  _register_per_param_settings: _register_per_param_settings().
  _maybe_get_special_mesh: _maybe_get_special_mesh().
  _enable_xla_runtime_errors: _enable_xla_runtime_errors.
  _BATCH_AXES: _BATCH_AXES.
  non_empty_leaf_merge_fn: non_empty_leaf_merge_fn().
  thread_stack_traces: thread_stack_traces().
  RematPolicy.__call__: RematPolicy#__call__().
  RematCombineFn.__call__: RematCombineFn#__call__().
  RematCombineFn: RematCombineFn#
  DataPartitionType.REPLICATED: DataPartitionType#REPLICATED.
  DataPartitionType.BATCH: DataPartitionType#BATCH.
  check_jax_type: check_jax_type().
  _NestedT: _NestedT.
  save_and_offload_only_these_names_regex.policy: save_and_offload_only_these_names_regex().policy().
  offload_dots_saveable.policy: offload_dots_saveable().policy().
  _concat: _concat().
  split_prng_key.verify_key_shape: split_prng_key().verify_key_shape().
  split_prng_key.stack_and_reshape: split_prng_key().stack_and_reshape().
  complete_partition_spec_tree.add_leaves: complete_partition_spec_tree().add_leaves().
  _SEQ_AXES: _SEQ_AXES.
  host_to_global_specs.make_array_spec: host_to_global_specs().make_array_spec().
  global_to_host_array.index_to_shard: global_to_host_array().index_to_shard().
  DataDirStack: DataDirStack#
  _reshape_mesh_to_rings: _reshape_mesh_to_rings().
  pytree_children: pytree_children().
  own_fields.get_base_keys: own_fields().get_base_keys().
  VDict.__repr__: VDict#__repr__().
  VDict.tree_flatten_with_keys: VDict#tree_flatten_with_keys().
  VDict.tree_unflatten: VDict#tree_unflatten().
  DeviceUsage.device_id: DeviceUsage#device_id.
  DeviceUsage.device_duty_cycle_percent: DeviceUsage#device_duty_cycle_percent.
  DeviceUsage.device_utilization: DeviceUsage#device_utilization.
  DeviceUsage.hbm_memory_usage_bytes: DeviceUsage#hbm_memory_usage_bytes.
  DeviceUsage.hbm_memory_total_bytes: DeviceUsage#hbm_memory_total_bytes.
  DeviceUsage.hbm_memory_bandwidth_utilization: DeviceUsage#hbm_memory_bandwidth_utilization.
---
# Module: [`axlearn/common/utils.py`](../../../../../../raw/code/axlearn/axlearn/common/utils.py)

## Classes
### `DataDirStack`  ·  implements/extends local
- def: [`axlearn/common/utils.py:1514`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1514)
- doc: See `install_context_stack` on how to ensure thread-safety of the global stack.
- signature: `class DataDirStack(threading.local):`
- members:
  - `stack` — [`L1517`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1517)
- used by: [`get_data_dir`](utils.md#get_data_dir), [`_global_data_dir_stack`](utils.md#_global_data_dir_stack), [`pop_data_dir`](utils.md#pop_data_dir), [`push_data_dir`](utils.md#push_data_dir)

### `DataPartitionType`  ·  implements/extends Enum
- def: [`axlearn/common/utils.py:808`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L808)
- signature: `class DataPartitionType(Enum):`
- members:
  - `BATCH` — [`L814`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L814)
  - `FULL` — [`L810`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L810)
  - `REPLICATED` — [`L812`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L812)
- used by: [`data_partition_type_to_spec`](utils.md#data_partition_type_to_spec), [`host_to_global_array`](utils.md#host_to_global_array), [`make_array`](utils.md#host_to_global_array.make_array), [`global_to_host_array`](utils.md#global_to_host_array), [`batch_partition_spec`](inference_output.md#BaseOutputWriter.Config.batch_partition_spec)

### `DeviceUsage`
- def: [`axlearn/common/utils.py:2012`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2012)
- doc: Usage measurements for a device.
- signature: `class DeviceUsage:`
- members:
  - `device_duty_cycle_percent` — [`L2016`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2016)
  - `device_id` — [`L2015`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2015)
  - `device_utilization` — [`L2017`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2017)
  - `hbm_memory_bandwidth_utilization` — [`L2020`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2020)
  - `hbm_memory_total_bytes` — [`L2019`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2019)
  - `hbm_memory_usage_bytes` — [`L2018`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2018)
- used by: [`_check_host_and_log_metrics`](monitoring/device_monitor.md#DeviceMonitor._check_host_and_log_metrics), [`collect_metrics`](monitoring/device_monitor.md#DeviceMonitorClient.collect_metrics), [`is_host_idle`](monitoring/device_monitor.md#DeviceMonitorClient.is_host_idle)

### `HybridMeshShape`
- def: [`axlearn/common/utils.py:95`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L95)
- doc: A mesh shape for hybrid (i.e., ICI and DCN) parallelism.
- signature: `class HybridMeshShape:`
- members:
  - `dcn_mesh_shape` — [`L106`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L106)
  - `ici_mesh_shape` — [`L105`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L105)
- protocol/private: `__len__`[`L114`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L114), `__post_init__`[`L108`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L108)
- uses (calls/refs, reference-scoped): [`MeshShape`](utils.md#MeshShape)
- used by: [`_get_compiled_train_step_fn`](trainer.md#SpmdTrainer._get_compiled_train_step_fn), [`select_mesh_config`](trainer.md#select_mesh_config), [`create_device_mesh`](utils.md#create_device_mesh), [`mesh_shape`](trainer.md#SpmdTrainer.Config.mesh_shape), [`create_hybrid_device_mesh`](utils.md#create_hybrid_device_mesh)

### `PerParamFn`  ·  implements/extends Protocol
- def: [`axlearn/common/utils.py:1176`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1176)
- doc: A callable that operates on each parameter.
- signature: `class PerParamFn(Protocol[T]):`
- members:
  - `__call__(self, params: Union[Nested[Tensor], Nested[TensorSpec]])` — [`L1179`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1179) — This protocol requires a callable that accepts either a nested Tensor or
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`TensorSpec`](utils.md#TensorSpec), [`T`](utils.md#T)
- used by: [`canonicalize_per_param_dtype`](utils.md#canonicalize_per_param_dtype), [`per_param_dtype_by_path`](utils.md#per_param_dtype_by_path), [`train_dtype`](trainer.md#SpmdTrainer.Config.train_dtype)

### `RematCombineFn`  ·  implements/extends Protocol
- def: [`axlearn/common/utils.py:222`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L222)
- signature: `class RematCombineFn(Protocol):`
- members:
  - `__call__(self, p1: RematType, p2: RematType, *, prim: Primitive, args: tuple[Any], kwargs: dict[str, Any])` — [`L223`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L223) — Protocol for remat policy combine function.
- uses (calls/refs, reference-scoped): [`RematType`](utils.md#RematType)
- used by: [`combine_remat_policies`](utils.md#combine_remat_policies), [`default_remat_combine_fn`](utils.md#default_remat_combine_fn)

### `RematPolicy`  ·  implements/extends Protocol
- def: [`axlearn/common/utils.py:159`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L159)
- signature: `class RematPolicy(Protocol):`
- protocol/private: `__call__`[`L160`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L160)
- uses (calls/refs, reference-scoped): [`RematType`](utils.md#RematType)
- used by: [`combine_remat_policies`](utils.md#combine_remat_policies), [`save_and_offload_only_these_names_regex`](utils.md#save_and_offload_only_these_names_regex), [`_save_and_offload_only_these_names_regex`](attention.md#_save_and_offload_only_these_names_regex), [`save_or_offload_flash_attention_policy`](flash_attention/remat.md#save_or_offload_flash_attention_policy), [`offload_dots_saveable`](utils.md#offload_dots_saveable)

### `StackedKeyArray`  ·  implements/extends NamedTuple
- def: [`axlearn/common/utils.py:535`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L535)
- signature: `class StackedKeyArray(NamedTuple):`
- members:
  - `keys` — [`L536`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L536)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor)
- used by: [`_run`](repeat.md#Repeat._run), [`initialize_parameters_recursively`](pipeline.md#Pipeline.initialize_parameters_recursively), [`split_prng_key`](utils.md#split_prng_key), [`initialize_parameters_recursively`](attention.md#FusedQKVLinear.initialize_parameters_recursively), [`initialize_parameters_recursively`](repeat.md#Repeat.initialize_parameters_recursively)

### `TensorSpec`
- def: [`axlearn/common/utils.py:127`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L127)
- doc: Specification of a Tensor.
- signature: `class TensorSpec:`
- members:
  - `sharding(self)` — [`L139`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L139)
  - `dtype` — [`L134`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L134)
  - `memory_kind` — [`L136`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L136)
  - `mesh_axes` — [`L135`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L135)
  - `shape` — [`L133`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L133)
- uses (calls/refs, reference-scoped): [`ParameterSpec`](base_layer.md#ParameterSpec), `env`, [`MemoryKind`](utils.md#MemoryKind)
- used by: [`ParameterSpec`](base_layer.md#ParameterSpec), [`initialize_parameters_recursively`](base_layer.md#BaseLayer.initialize_parameters_recursively), `test_all_mask`, [`OptStateSpec`](optimizer_base.md#OptStateSpec), [`_get_spec`](checkpointer.md#TensorStoreStateStorage._get_spec), [`create_parameter_specs_recursively`](base_layer.md#BaseLayer.create_parameter_specs_recursively), [`_create_layer_parameter_specs`](quantized_dot_general/layers.md#QuantizedDotGeneral._create_layer_parameter_specs), [`_create_layer_parameter_specs`](rattention/rattention.md#ResidualLinearAttention._create_layer_parameter_specs), [`adastar_optimizer`](optimizers.md#adastar_optimizer), [`test_conditional_update_transformation`](update_transformation_test.md#ConditionalUpdateTransformationTest.test_conditional_update_transformation), [`scale_by_schedule`](optimizers.md#scale_by_schedule), [`partition_fn`](optimizers.md#skip_and_clip_by_global_norm.partition_fn), [`create_parameter_specs_recursively`](attention.md#FusedQKVLinear.create_parameter_specs_recursively), [`target_to_source`](state_builder.md#EmaParamsConverter.target_to_source), [`_create_layer_parameter_specs`](attention.md#MultiheadAttention._create_layer_parameter_specs), [`_restore_args`](checkpointer_orbax.md#OrbaxCheckpointer._restore_args), [`_trainer_state_specs`](trainer.md#SpmdTrainer._trainer_state_specs), [`partition_fn`](optimizers.md#ema.partition_fn), [`restore`](checkpointer_orbax.md#OrbaxCheckpointer.restore), [`get_vx_spec`](factorized_rms.md#scale_by_factored_rms.get_vx_spec), [`copy_partition`](optimizers.md#copy_partition), [`partition_fn`](optimizers.md#param_ema.partition_fn), [`partition_fn`](optimizers.md#scale_by_lion.partition_fn), [`create_state_partition_specs`](update_transformation.md#ConditionalUpdateTransformation.create_state_partition_specs), [`TrainerState`](trainer.md#TrainerState), [`_initialize_parameter`](base_layer.md#BaseLayer._initialize_parameter), [`_partition`](optimizers.md#adastar_optimizer.partition_fn._partition), [`_compute_fan_axes`](base_layer.md#BaseLayer._compute_fan_axes), [`test_param_specs`](update_transformation_test.md#UpdatesTest.test_param_specs), [`partition_fn`](optimizers.md#adastar_optimizer.partition_fn), [`param_specs`](update_transformation.md#Updates.param_specs), [`_create_layer_parameter_specs`](attention.md#MultiheadAttentionXL._create_layer_parameter_specs), [`_create_layer_parameter_specs`](layers.md#BatchNorm._create_layer_parameter_specs), [`_create_layer_parameter_specs`](layers.md#MovingAverage._create_layer_parameter_specs), [`model`](trainer.md#TrainerState.model), [`partition_fn`](factorized_rms.md#scale_by_factored_rms.partition_fn), [`partition_fn`](optimizers.md#adam_partition.partition_fn), [`_create_layer_parameter_specs`](attention.md#BaseMultiheadLinear._create_layer_parameter_specs), [`_create_layer_parameter_specs`](attention.md#TransformerAttentionLayer._create_layer_parameter_specs), [`_create_layer_parameter_specs`](attention.md#TransformerFeedForwardLayer._create_layer_parameter_specs)  (+33 more; 1 test-only)

### `VDict`  ·  implements/extends dict
- def: [`axlearn/common/utils.py:433`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L433)
- doc: A dict with Tensor leaf nodes whose values should be vectorized.
- signature: `class VDict(dict):`
- members:
  - `tree_flatten_with_keys(self)` — [`L439`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L439)
  - `tree_unflatten(cls, keys, values)` — [`L453`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L453)
- protocol/private: `__repr__`[`L436`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L436)
- used by: `test_transformer_extend_step`, [`create_parameter_specs_recursively`](attention.md#FusedQKVLinear.create_parameter_specs_recursively), `_convert_from_stacked_params`, [`create_parameter_specs_recursively`](pipeline.md#Pipeline.create_parameter_specs_recursively), [`initialize_parameters_recursively`](pipeline.md#Pipeline.initialize_parameters_recursively), `test_transformer_prefill_states`, [`vectorized_tree_map`](utils.md#vectorized_tree_map), [`create_parameter_specs_recursively`](repeat.md#Repeat.create_parameter_specs_recursively), [`test_param_specs`](update_transformation_test.md#UpdatesTest.test_param_specs), `_convert_to_qkv_linear`, [`extend_step`](attention.md#RepeatedTransformerLayer.extend_step), [`fn`](utils.md#expand_vdicts.fn), [`expand_vdicts`](utils.md#expand_vdicts), [`init_states`](attention.md#RepeatedTransformerLayer.init_states), [`mock_params`](update_transformation_test.md#mock_params), [`init`](pipeline.md#Pipeline.init), [`init`](repeat.md#Repeat.init), [`vectorized_fn`](utils.md#vectorized_tree_map.vectorized_fn), [`init`](attention.md#FusedQKVLinear.init)

## Functions
- `_concat(*, prefix: str, suffix: str, separator: str)` — [`L373`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L373)
- `_find_cycles(tree: Nested, *, key_path: KeyPath, seen: list[int])` — [`L1976`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1976)
- `_key_entry_to_str(key_entry: KeyEntry)` — [`L377`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L377)
- `_maybe_get_special_mesh(mesh_shape: MeshShape, *, devices: np.ndarray)` — [`L1667`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1667) — Checks if any of the special mesh shapes are applicable.
- `_register_per_param_settings(settings: NestedTree, *, description: str, path: Optional[str] = None)` — [`L1591`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1591)
- `_reshape_mesh_to_rings(a: np.ndarray, *, shape: tuple[int, int])` — [`L1625`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1625) — Reshapes device mesh to rings for 64x4 or 32x8 mesh shape.
- `add_leaves(i, x)` — [`L710`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L710)
- `as_numpy_array(x: Any)` — [`L612`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L612) — Converts `x` to numpy ndarray recursively.
- `as_tensor(x: Any)` — [`L587`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L587) — Converts `x` to Tensor recursively.
- `build_standard_mesh(mesh_shape: MeshShape, *, devices: np.ndarray)` — [`L1686`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1686)
- `canonicalize_per_param_dtype(param_dtype: Union[jnp.dtype, ConfigOr[PerParamFn[jnp.dtype]]])` — [`L1288`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1288) — Canonicalize the input `param_dtype` to a consistent format of
- `cast(x: Union[Tensor, TensorSpec])` — [`L1164`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1164)
- `cast_floats(in_tree: Union[NestedTensor, NestedTensorSpec], to_dtype: Optional[jnp.dtype])` — [`L1139`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1139) — Maps valid float arrays found in the inputs to the requested dtype in {float32, bfloat16}.
- `cast_floats_per_param(in_tree: Union[NestedTensor, NestedTensorSpec], per_param_dtype: Nested[jnp.dtype])` — [`L1247`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1247) — Cast each parameter in a tree to a specified dtype.
- `cast_per_param(x: Union[Tensor, TensorSpec], to_dtype: jnp.dtype)` — [`L1266`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1266)
- `check_jax_type(*, args: Optional[Sequence] = None, kwargs: Optional[dict] = None, pretty_named_args: Optional[dict] = None, msg: Optional[str] = None)` — [`L1371`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1371) — Checks that the supplied arguments are valid JAX types and raise ValueError if not.
- `check_numerics(x: Tensor, msg_fmt: str = "", **msg_kwargs)` — [`L359`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L359) — Checks that all elements in `x` are finite.
- `check_param_shape_alignment(source_tree: NestedTensor, target_tree: NestedTensor)` — [`L1336`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1336) — Check param shape alignment between two parameter trees.
- `combine_fn(p1: RematType, p2: RematType, *, prim: Primitive, args: tuple[Any], kwargs: dict[str, Any])` — [`L261`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L261)
- `combine_remat_policies(policy_1: ConfigOr[RematPolicy], policy_2: ConfigOr[RematPolicy], *, combine_fn: ConfigOr[RematCombineFn] = default_remat_combine_fn())` — [`L289`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L289) — Returns a remat policy that combines the two policies with `combine_fn`.
- `complete_partition_spec_tree(treedef: jax.tree_util.PyTreeDef, partition_spec_tree: NestedTree)` — [`L670`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L670) — Adapted from flatten_axes(), but with a simplified API and more error logging and messages.
- `convert_to_enum(p: Union[RematType, bool])` — [`L311`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L311)
- `copy_recursively(*, source: NestedTensor, target: NestedTensor, path: Union[str, Sequence[str]], separator: str = "/")` — [`L1100`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1100) — Sets target[path] = source[path].
- `count_model_params(tree: NestedTensor)` — [`L1331`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1331) — Count the number of parameters in a model.
- `create_device_mesh(mesh_shape: Union[MeshShape, HybridMeshShape], *, devices: Optional[Sequence[Any]] = None)` — [`L1767`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1767) — Constructs a device mesh.
- `create_hybrid_device_mesh(mesh_shape: HybridMeshShape, *, devices: Sequence[Any], process_is_granule: bool = False)` — [`L1714`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1714) — Extends the method to have an option to fall back to naive mesh.
- `data_partition_type_to_spec(partition: Union[DataPartitionType, Nested[PartitionSpec]])` — [`L821`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L821) — Returns a PartitionSpec for the given partition type.
- `default_remat_combine_fn(preferred_remat_type: Optional[RematType] = None)` — [`L246`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L246) — The default remat policy combine function.
- `dispatch_input_batch(input_batch: NestedTensor, *, batch_axis_names: Union[str, Sequence[str]] = "data")` — [`L763`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L763) — Constrains all leaf values in the input batch, then (optionally) dispatches examples
- `exit_fn(_, child)` — [`L2142`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2142)
- `expand_vdicts(tree: NestedTensor)` — [`L483`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L483) — Expands each VDict in `tree` to a list.
- `find_cycles(tree: Nested)` — [`L1960`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1960) — Find a cycle in pytree `tree` if one exists.
- `flatten_items(tree: Nested[Tensor], separator: str = "/", is_leaf: Optional[Callable[[Any], bool]] = None)` — [`L421`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L421) — Flattens `tree` and returns a list of (path, value) pairs.
- `fn(value: Union[Tensor, VDict])` — [`L496`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L496)
- `fn(tree: Union[Nested[Tensor], Nested[TensorSpec]])` — [`L1233`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1233)
- `get_base_keys(base: type)` — [`L2092`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2092)
- `get_current_abstract_or_physical_mesh()` — [`L650`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L650) — Returns the current abstract mesh if it's set, or the physical mesh otherwise.
- `get_data_dir()` — [`L1559`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1559)
- `get_local_array(value: Tensor)` — [`L980`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L980)
- `get_or_none(x: Optional[dict], key: Any)` — [`L1565`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1565)
- `get_recursively(x: NestedTensor, path: Union[str, Sequence[str]], separator: Optional[str] = "/")` — [`L1029`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1029) — Recursively indexes through the nested tensor.
- `get_tpu_dot_precision(dtype)` — [`L2163`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2163) — Get the suitable DotAlgorithmPreset for the given dtype.
- `global_to_host_array(global_arrays: Nested[Tensor], *, partition: Optional[DataPartitionType] = DataPartitionType.FULL)` — [`L941`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L941) — Extracts host addressable data from each Tensor in `global_arrays`.
- `host_to_global_array(host_arrays: Nested[Union[np.ndarray, Tensor]], *, partition: Union[Nested[PartitionSpec], DataPartitionType] = DataPartitionType.FULL)` — [`L842`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L842) — Converts the given host device arrays to global device arrays.
- `host_to_global_device_array(*args, **kwargs)` — [`L929`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L929) — A deprecated alias for `host_to_global_array`.
- `host_to_global_specs(host_arrays: Nested[jax.ShapeDtypeStruct], *, partition: PartitionSpec)` — [`L903`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L903) — Converts the given host-local specs to global array specs.
- `index_to_shard(shards: list[jax.Shard], global_shape: Sequence[int])` — [`L962`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L962) — Returns a mapping from (sorted) indices to shards.
- `infer_mesh_shape(mesh_shape: MeshShape, *, num_devices: Optional[int] = None)` — [`L1891`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1891) — Infer the value for -1 from len(jax.devices()) and other dims if there is -1 in mesh shape.
- `input_partition_spec()` — [`L743`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L743) — Returns partition spec for the input batch.
- `make_array(x: np.ndarray, partition_spec: PartitionSpec)` — [`L878`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L878)
- `make_array_spec(x: jax.ShapeDtypeStruct, partition_spec: PartitionSpec)` — [`L917`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L917)
- `match_regex_rules(x: str, *, rules: Sequence[tuple[str, T]], default_value: Optional[T] = None)` — [`L1572`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1572) — Matches the given string against a sequence of regex-based rules.
- `maybe_shard(x: NestedTensor, partition_spec: Optional[PartitionSpec])` — [`L644`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L644)
- `non_empty_leaf_merge_fn(primary: Any, secondary: Any)` — [`L1461`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1461) — This function chooses the non-empty leaf. If both leaves are non-empty, an error
- `numeric_checks(enabled: bool = True)` — [`L346`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L346)
- `offload_dots_saveable(offload_src: str, offload_dst: str)` — [`L199`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L199) — Extract from offload_dot_with_no_batch_dims and remove no-batch-dims limit.
- `own_fields(cfg: ConfigBase)` — [`L2084`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2084) — Returns fields that are defined by `cfg`, rather than any of its ancestors.
- `partial_with_fn_metadata(fn, *args, **kwargs)` — [`L1421`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1421) — Wraps a function with functools.partial, retaining original function's metadata.
- `per_param_dtype_by_path(default_dtype: Optional[jnp.dtype] = None, *, update_rules: Optional[Sequence[tuple[str, Optional[jnp.dtype]]]] = None)` — [`L1191`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1191) — Returns a function that assigns a dtype to each parameter based on the provided update
- `policy(prim, *_, **params)` — [`L186`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L186)
- `policy(prim, *_, **params)` — [`L214`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L214)
- `policy(prim, *args, **kwargs)` — [`L316`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L316)
- `pop_data_dir()` — [`L1527`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1527)
- `prune_empty(in_tree: Nested[Tensor])` — [`L2068`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2068) — Returns a shallow copy of the input tree with empty subtrees pruned.
- `prune_tree(in_tree: NestedTensor, should_prune: Callable[[str, NestedTensor], bool], *, prefix: str = "", separator: str = "/")` — [`L1427`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1427) — Returns a shallow copy of the input tree with subtrees pruned based on `should_prune`.
- `push_data_dir(data_dir: Optional[str])` — [`L1523`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1523)
- `pytree_children(node: Any)` — [`L1935`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1935) — Generate the (key, value) pairs for the immediate children of a pytree `node`.
- `raise_for_cycles(tree: Any)` — [`L1998`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1998) — Raise an informative error message if `tree` contains cycles.
- `register_per_param_settings(settings: NestedTree, *, description: str, path: Optional[str] = None)` — [`L1597`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1597) — Registers per-parameter setting.
- `replace_layer_config_recursively(cfg: ConfigBase, *, target_cls: Configurable, source_config: ConfigBase, exclude_keys: Optional[Sequence[str]] = None)` — [`L2099`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2099) — Replaces the target_cls's config with the source_config.
- `replace_none_with_proxy(tree)` — [`L695`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L695)
- `replicate_to_local_data(x: NestedTensor)` — [`L658`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L658) — Replicates and converts Tensors in `x` to local DeviceArrays.
- `runtime_checks(enabled: bool = True)` — [`L332`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L332)
- `safe_not(mask: Tensor)` — [`L2042`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2042) — Inverts a boolean mask.
- `save_and_offload_only_these_names_regex(*, names_which_can_be_saved: SavePattern, names_which_can_be_offloaded: SavePattern, offload_src: str, offload_dst: str)` — [`L163`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L163) — Adapted from jax source code to support regex.
- `sequence_mask(*, lengths: Tensor, max_len: int, dtype: jnp.dtype = jnp.bool)` — [`L2023`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2023) — Computes a mask over sequence positions for each given length.
- `set_data_dir(data_dir: Optional[str])` — [`L1532`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1532) — Sets the environment variable DATA_DIR to the given `data_dir`.
- `set_recursively(x: NestedTensor, *, value: Tensor, path: Union[str, Sequence[str]], separator: Optional[str] = "/")` — [`L1062`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1062) — Sets x[path...] = value, where path can be a multi-part index.
- `shapes(nested_tensor: NestedTensor)` — [`L367`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L367) — Returns a tree of the same structure as `nested_tensor` but with corresponding shapes instead
- `split_prng_key(prng_key: Union[StackedKeyArray, Tensor], num_keys: Union[int, Sequence[int]])` — [`L539`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L539) — Splits prng_key to keys iteratively and return the stacked keys.
- `stack_and_reshape(*keys)` — [`L573`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L573)
- `switch(value)` — [`L335`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L335)
- `switch(value)` — [`L349`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L349)
- `thread_stack_traces()` — [`L1921`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1921) — Retrieves the current python stack traces.
- `traverse_and_dispatch(data: NestedTensor)` — [`L795`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L795)
- `tree_merge(primary: Nested[Any], *, secondary: Nested[Any], leaf_merge_fn: Callable[[Any, Any], Any])` — [`L1484`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1484) — Merge `secondary` into `primary`. The result contains deep copies of subtrees from both.
- `tree_paths(tree: NestedTree, separator: str = "/", is_leaf: Optional[Callable[[Any], bool]] = None)` — [`L396`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L396) — Returns a tree of the same structure as `nested_tensor` but with corresponding paths instead
- `validate_contains_paths(x: Nested[Tensor], paths: Sequence[str])` — [`L2056`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L2056) — Raises ValueError if any of the given `paths` are not present in `x`.
- `validate_float_dtype(dtype: jnp.dtype)` — [`L1408`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1408) — Validates if the provided dtype is both a float and amongst the set supported.
- `vectorized_fn(*nodes)` — [`L470`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L470)
- `vectorized_tree_map(fn, tree, *rest)` — [`L467`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L467) — Similar to jax.tree.map(), but vectorizes `fn` on VDict's.
- `verify_key_shape(x)` — [`L559`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L559)
- `with_sharding_constraint(x: Tensor, shardings)` — [`L637`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L637)

## Module values
- `MemoryKind` — [`L123`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L123)
- `MeshShape` — [`L85`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L85)
- `Nested` — [`L73`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L73)
- `NestedPartitionSpec` — [`L78`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L78)
- `NestedTensor` — [`L77`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L77)
- `NestedTensorSpec` — [`L148`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L148)
- `NestedTree` — [`L76`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L76)
- `PHYSICAL_TO_LOGICAL_DISPATCH_KEY` — [`L760`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L760)
- `PartitionSpecType` — [`L79`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L79)
- `RematType` — [`L149`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L149)
- `SavePattern` — [`L150`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L150)
- `T` — [`L81`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L81)
- `Tensor` — [`L75`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L75)
- `_BATCH_AXES` — [`L817`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L817)
- `_NestedT` — [`L72`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L72)
- `_SEQ_AXES` — [`L818`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L818)
- `_enable_numeric_checks` — [`L87`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L87)
- `_enable_xla_runtime_errors` — [`L88`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L88)
- `_global_data_dir_stack` — [`L1520`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L1520)
- `_supported_float_dtypes` — [`L91`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L91)
- `extended_checkpoint_policies` — [`L324`](../../../../../../raw/code/axlearn/axlearn/common/utils.py#L324)

