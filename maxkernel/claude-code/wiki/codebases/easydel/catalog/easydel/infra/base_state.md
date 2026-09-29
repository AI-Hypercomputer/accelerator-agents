---
title: 'Module: easydel/infra/base_state.py'
type: catalog
provenance: extracted
module: easydel/infra/base_state.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.base_state`/
symbols:
  EasyDeLState: EasyDeLState#
  EasyDeLState.model: EasyDeLState#model().
  EasyDeLState.graphstate: EasyDeLState#graphstate.
  EasyDeLState.load_state: EasyDeLState#load_state().
  EasyDeLState.step: EasyDeLState#step.
  EasyDeLState.merge: EasyDeLState#merge().
  EasyDeLState.load_optimizer: EasyDeLState#load_optimizer().
  logger: logger.
  EasyDeLState.shard_state: EasyDeLState#shard_state().
  EasyDeLState.init_tx: EasyDeLState#init_tx().
  EasyDeLState.shardings: EasyDeLState#shardings().
  EasyDeLState.graphother: EasyDeLState#graphother.
  EasyDeLState.gather_optimizer_state: EasyDeLState#gather_optimizer_state().
  EasyDeLState.save_state: EasyDeLState#save_state().
  EasyDeLState.gather_model: EasyDeLState#gather_model().
  EasyDeLState.opt_state: EasyDeLState#opt_state.
  EasyDeLState.shard_optimizer_state: EasyDeLState#shard_optimizer_state().
  EasyDeLState.shard_model: EasyDeLState#shard_model().
  _sanitize_partition_specs_for_shape_tree: _sanitize_partition_specs_for_shape_tree().
  EasyDeLState.load_state._BaseModuleLoader._model_task: EasyDeLState#load_state()._BaseModuleLoader#_model_task.
  EasyDeLState.save_optimizer: EasyDeLState#save_optimizer().
  EasyDeLState.graphdef: EasyDeLState#graphdef.
  EasyDeLState.new_method: EasyDeLState#new_method().
  _has_saved_optimizer_state: _has_saved_optimizer_state().
  EasyDeLState.apply_gradients: EasyDeLState#apply_gradients().
  EasyDeLState.apply_sharding_on_tree: EasyDeLState#apply_sharding_on_tree().
  EasyDeLState._apply_gather_on_tree: EasyDeLState#_apply_gather_on_tree().
  _has_resume_model: _has_resume_model().
  EasyDeLState.apply_fn: EasyDeLState#apply_fn.
  EasyDeLState.size: EasyDeLState#size().
  EasyDeLState.gather_state: EasyDeLState#gather_state().
  EasyDeLState.tx: EasyDeLState#tx.
  _get_checkpoint_step: _get_checkpoint_step().
  EasyDeLState._load_tensorstore: EasyDeLState#_load_tensorstore().
  EasyDeLState.__repr__: EasyDeLState#__repr__().
  _read_checkpoint_metadata: _read_checkpoint_metadata().
  EasyDeLState.create: EasyDeLState#create().
  EasyDeLState.mesh: EasyDeLState#mesh().
  _sanitize_partition_specs_for_shape_tree._sanitize: _sanitize_partition_specs_for_shape_tree()._sanitize().
  EasyDeLState.load_state._BaseModuleLoader: EasyDeLState#load_state()._BaseModuleLoader#
  RESUME_MODEL_SUBDIR: RESUME_MODEL_SUBDIR.
  OPTIMIZER_NAME: OPTIMIZER_NAME.
  OPTIMIZER_STRUCT_NAME: OPTIMIZER_STRUCT_NAME.
  TX_STRUCT_JSON: TX_STRUCT_JSON.
  EasyDeLState.make: EasyDeLState#make().
  EasyDeLState.calculate_size: EasyDeLState#calculate_size().
  AM: AM.
  _is_optimizer_template_incompatibility: _is_optimizer_template_incompatibility().
  EasyDeLState.esurge_cache_scope_key: EasyDeLState#esurge_cache_scope_key.
  EasyDeLState.merge_to_state: EasyDeLState#merge_to_state().
  WEIGHTS_NAME: WEIGHTS_NAME.
  EasyDeLState.shard_with_shape: EasyDeLState#shard_with_shape().
  EasyDeLState.__str__: EasyDeLState#__str__.
---
# Module: [`easydel/infra/base_state.py`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py)

## Classes
### `EasyDeLState`  ·  implements/extends PyTreeNode
- def: [`easydel/infra/base_state.py:253`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L253) — documented in [easydel-infra-base_module](../../../concepts/easydel-infra-base_module.md)
- doc: Complete state container for EasyDeL models during training or inference.
- signature: `class EasyDeLState(struct.PyTreeNode):`
- members:
  - `__repr__(self)` — [`L1917`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1917) — Return a string representation of the EasyDeLState.
  - `apply_gradients(self: Self, *, grads)` — [`L353`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L353) — Apply gradients to update parameters and optimizer state.
  - `apply_sharding_on_tree(tree)` — [`L1654`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1654) — Apply sharding functions to a pytree.
  - `calculate_size(pytree)` — [`L932`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L932) — Calculate total size of JAX arrays in a pytree.
  - `create(cls, *, step: int | None = None, graphdef: nn.GraphDef | None = None, graphstate: nn.GraphState | None = None, graphother: nn.GraphState | None = None, model: nn.Module | None = None, tx: optax.GradientTransformation | None = None, opt_state: optax.OptState | None = None, init_opt_state: bool = False)` — [`L422`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L422) — Create a new EasyDeLState instance.
  - `gather_model(self, partition_rules: PartitionLike = None, mesh: Mesh | None = None)` — [`L1730`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1730) — Gather model parameters from distributed devices.
  - `gather_optimizer_state(self: Self, partition_rules: PartitionLike = None)` — [`L738`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L738) — Gather the optimizer state from distributed devices to a single device.
  - `gather_state(self)` — [`L1690`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1690) — Gather the entire state from distributed devices.
  - `init_tx(self: Self, tx: optax.GradientTransformation, partition_rules: PartitionLike = None)` — [`L575`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L575) — Initialize the optimizer state with automatic sharding support.
  - `load_optimizer(self: Self, load_directory: str | ePathLike, checkpointer: Checkpointer | None = None, tx_template: None | optax.GradientTransformation = None)` — [`L943`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L943) — Load optimizer state from saved checkpoint files.
  - `load_state(cls, load_directory: str | os.PathLike, device: jax.Device | None = "cpu", dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.bfloat16, precision: jax.lax.Precision | None = None, sharding_axis_dims: collections.abc.Sequence[int] = (1, -1, 1, 1, 1), sharding_dcn_axis_dims: collections.abc.Sequence[int] | None = None, sharding_axis_names: collections.abc.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"), partition_axis: PartitionAxis | None = None, shard_fns: collections.abc.Mapping[tuple, tp.Callable] | dict | None = None, backend: EasyDeLBackends | None = None, platform: EasyDeLPlatforms | None = None, config_kwargs: EasyDeLBaseConfigDict | None = None, model_task: TaskType = TaskType.AUTO_BIND, auto_shard_model: bool = True, partition_rules: tuple[tuple[str, PartitionSpec], ...] | None = None, quantization_config: QuantizationConfig | None = None, apply_quantization: bool = False, verbose: bool = True, tx_template: optax.GradientTransformation | None = None, **kwargs)` — [`L1323`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1323) — Load an EasyDeLState from a saved checkpoint directory. — documented in [easydel-infra-base_state](../../../concepts/easydel-infra-base_state.md)
  - `make(graphstate)` — [`L633`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L633)
  - `merge(self: Self, tree)` — [`L790`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L790) — Merge a parameter tree with the graph definition to reconstruct the model.
  - `merge_to_state(self: Self, tree)` — [`L832`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L832) — Create a new state with updated parameters.
  - `mesh(self)` — [`L1865`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1865) — Get the JAX device mesh from the model.
  - `model(self)` — [`L862`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L862) — Reconstruct and return the full EasyDeL model module. — documented in [easydel-infra-base_state](../../../concepts/easydel-infra-base_state.md)
  - `new_method(tx_template)` — [`L1022`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1022) — Load using modern TensorStore format.
  - `save_optimizer(self, save_directory: str | ePathLike, float_dtype: jnp.dtype | None = None, checkpointer: Checkpointer | None = None, step: int | None = None)` — [`L1104`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1104) — Save the optimizer state to a directory.
  - `save_state(self, save_directory: str | os.PathLike | ePathLike, float_dtype: jnp.dtype | None = None, save_optimizer: bool = True, merge_lora_before_save: bool = False, step: int | None = None)` — [`L1197`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1197) — Save the complete EasyDeLState to a directory.
  - `shard_model(self: Self, partition_rules: PartitionLike = None, mesh: Mesh | None = None)` — [`L1797`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1797) — Shard model parameters based on partition rules.
  - `shard_optimizer_state(self, opt_state: tp.Any | None = None, partition_rules: PartitionLike = None)` — [`L668`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L668) — Apply sharding to the optimizer state based on partition rules.
  - `shard_state(self, partition_rules: PartitionLike = None, mesh: Mesh = None)` — [`L1592`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1592) — Shard the entire state based on partition rules.
  - `shard_with_shape(self: Self, shape)` — [`L1547`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1547) — Apply sharding constraints based on a reference shape pytree.
  - `shardings(self)` — [`L1884`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1884) — Get the sharding annotations for all state components.
  - `size(self)` — [`L900`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L900) — Calculate the total memory size of the state in bytes.
  - `apply_fn` — [`L347`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L347)
  - `esurge_cache_scope_key` — [`L348`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L348)
  - `graphdef` — [`L340`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L340)
  - `graphother` — [`L343`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L343)
  - `graphstate` — [`L342`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L342) — documented in [easydel-infra-base_state](../../../concepts/easydel-infra-base_state.md)
  - `opt_state` — [`L346`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L346)
  - `step` — [`L339`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L339)
  - `tx` — [`L345`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L345)
- protocol/private: `__str__`[`L1929`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1929), `_apply_gather_on_tree`[`L1775`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1775), `_load_tensorstore`[`L1027`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1027)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](base_module.md#EasyDeLBaseModule), [`TaskType`](factory.md#TaskType), [`QuantizationConfig`](../layers/quantization/_configs.md#QuantizationConfig), [`mesh`](base_module.md#EasyDeLBaseModule.mesh), [`from_pretrained`](../modules/auto/auto_configuration.md#AutoEasyDeLConfig.from_pretrained), [`_get_partition_rules`](base_module.md#EasyDeLBaseModule._get_partition_rules), [`EasyDeLBackends`](etils.md#EasyDeLBackends), [`EasyDeLBaseConfigDict`](base_config.md#EasyDeLBaseConfigDict), [`EasyDeLPlatforms`](etils.md#EasyDeLPlatforms), [`logger`](base_state.md#logger), [`deepcopy_model`](../utils/traversals.md#deepcopy_model), [`_sanitize_partition_specs_for_shape_tree`](base_state.md#_sanitize_partition_specs_for_shape_tree), [`save_pretrained`](mixins/bridge.md#EasyBridgeMixin.save_pretrained), [`AutoEasyDeLConfig`](../modules/auto/auto_configuration.md#AutoEasyDeLConfig), [`PartitionLike`](base_module.md#PartitionLike), [`_get_mesh`](base_module.md#EasyDeLBaseModule._get_mesh), [`_has_saved_optimizer_state`](base_state.md#_has_saved_optimizer_state), [`bind_model_task`](../modules/auto/auto_configuration.md#AutoEasyDeLConfig.bind_model_task), [`materialize_meta_leaves`](utils.md#materialize_meta_leaves), [`AUTO_BIND`](factory.md#TaskType.AUTO_BIND), [`is_remote_path`](../utils/helpers.md#is_remote_path), [`_has_resume_model`](base_state.md#_has_resume_model), [`_get_checkpoint_step`](base_state.md#_get_checkpoint_step), `architectures`, [`RESUME_MODEL_SUBDIR`](base_state.md#RESUME_MODEL_SUBDIR), [`OPTIMIZER_NAME`](base_state.md#OPTIMIZER_NAME), [`OPTIMIZER_STRUCT_NAME`](base_state.md#OPTIMIZER_STRUCT_NAME), [`TX_STRUCT_JSON`](base_state.md#TX_STRUCT_JSON), [`AM`](base_state.md#AM), [`_is_optimizer_template_incompatibility`](base_state.md#_is_optimizer_template_incompatibility)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`generate_unified`](../trainers/base_trainer.md#BaseTrainer.generate_unified), [`_preprocess_batch_input`](../trainers/group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](../trainers/group_filtered_policy_optimization/gfpo_trainer.md#GFPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](../trainers/proximal_policy_optimization_trainer/ppo_trainer.md#PPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](../trainers/self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer._preprocess_batch_input), [`_train_epoch`](../trainers/trainer/trainer.md#Trainer._train_epoch), [`_preprocess_batch_input`](../trainers/agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer._preprocess_batch_input), [`configure_functions`](../trainers/distillation_trainer/distillation_trainer.md#DistillationTrainer.configure_functions), [`configure_functions`](../trainers/direct_preference_optimization_trainer/dpo_trainer.md#DPOTrainer.configure_functions), [`configure_functions`](../trainers/generalized_knowledge_distillation_trainer/gkd_trainer.md#GKDTrainer.configure_functions), [`_eval_dataset_steps_auto_clamped`](../trainers/base_trainer.md#BaseTrainer._eval_dataset_steps_auto_clamped), [`_tpu_preemption_sync_available`](../trainers/base_trainer.md#BaseTrainer._tpu_preemption_sync_available), [`_preprocess_batch_input`](../trainers/xpo_trainer/xpo_trainer.md#XPOTrainer._preprocess_batch_input), [`configure_functions`](../trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.md#OnPolicyDistillationTrainer.configure_functions), [`model_state`](../trainers/base_trainer.md#BaseTrainer.model_state), [`_eval_epoch`](../trainers/trainer/trainer.md#Trainer._eval_epoch), [`_preprocess_batch_input`](../trainers/nash_md_trainer/nash_md_trainer.md#NashMDTrainer._preprocess_batch_input), [`configure_functions`](../trainers/sparse_distillation_trainer/sparse_distillation_trainer.md#SparseDistillationTrainer.configure_functions), [`concatenated_forward`](../trainers/kto_trainer/kto_trainer.md#KTOTrainer.concatenated_forward), [`configure_functions`](../trainers/xpo_trainer/xpo_trainer.md#XPOTrainer.configure_functions), [`maybe_benchmark`](../trainers/base_trainer.md#BaseTrainer.maybe_benchmark), [`update_state_respectfully`](../trainers/training_utils.md#update_state_respectfully), [`maybe_generate`](../trainers/base_trainer.md#BaseTrainer.maybe_generate), [`__init__`](../trainers/base_trainer.md#BaseTrainer.__init__), [`concatenated_forward`](../trainers/binary_classifier_optimization_trainer/bco_trainer.md#BCOTrainer.concatenated_forward), [`minibatch_call`](../trainers/training_utils.md#minibatch_call), [`_configure_state`](../trainers/base_trainer.md#BaseTrainer._configure_state), [`_run_training_loop`](../trainers/trainer/trainer.md#Trainer._run_training_loop), [`build_trainer`](elarge/model.md#eLargeModel.build_trainer), [`train`](../trainers/ray_scaler/distributed_trainer.md#RayDistributedTrainer.train), [`_create_grain_dataloader`](../trainers/base_trainer.md#BaseTrainer._create_grain_dataloader), [`to_state`](base_module.md#EasyDeLBaseModule.to_state), [`_score_auxiliary_rewards`](../trainers/agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer._score_auxiliary_rewards), [`_preprocess_batch_input`](../trainers/sparse_distillation_trainer/sparse_distillation_trainer.md#SparseDistillationTrainer._preprocess_batch_input), [`_preprocess_batch_input`](../trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.md#OnPolicyDistillationTrainer._preprocess_batch_input), [`_preprocess_batch_input`](../trainers/seq_kd_trainer/seq_kd_trainer.md#SeqKDTrainer._preprocess_batch_input), [`__init__`](../trainers/group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.__init__), [`_log_training_generations_to_wandb`](../trainers/base_trainer.md#BaseTrainer._log_training_generations_to_wandb), [`__init__`](../trainers/proximal_policy_optimization_trainer/ppo_trainer.md#PPOTrainer.__init__)  (+194 more)

### `_BaseModuleLoader`  ·  implements/extends EasyDeLBaseModule
- def: [`easydel/infra/base_state.py:1503`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1503)
- doc: Internal loader class for binding model task.
- signature: `class _BaseModuleLoader(EasyDeLBaseModule):`
- protocol/private: `_model_task`[`L1506`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L1506)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](base_module.md#EasyDeLBaseModule), [`from_pretrained`](mixins/bridge.md#EasyBridgeMixin.from_pretrained), [`load_optimizer`](base_state.md#EasyDeLState.load_optimizer), [`logger`](base_state.md#logger), [`shard_state`](base_state.md#EasyDeLState.shard_state), [`create`](base_state.md#EasyDeLState.create)
- used by: [`EasyDeLBaseModule`](base_module.md#EasyDeLBaseModule)

## Functions
- `_get_checkpoint_step(load_directory: str | os.PathLike | ePathLike)` — [`L188`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L188) — Return the recorded checkpoint step from ``metadata.json`` when available.
- `_has_resume_model(load_directory: str | os.PathLike | ePathLike)` — [`L172`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L172) — Return whether ``load_state`` should load model weights from ``_resume_model``.
- `_has_saved_optimizer_state(load_directory: str | os.PathLike | ePathLike)` — [`L148`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L148) — Return whether a checkpoint directory contains resumable optimizer artifacts.
- `_is_optimizer_template_incompatibility(exc: Exception)` — [`L201`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L201) — Return whether an optimizer restore error signals a template mismatch.
- `_read_checkpoint_metadata(load_directory: str | os.PathLike | ePathLike)` — [`L135`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L135) — Best-effort read of the checkpoint discovery metadata.
- `_sanitize(spec: tp.Any, shape_obj: tp.Any)` — [`L217`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L217)
- `_sanitize_partition_specs_for_shape_tree(partition_specs: tp.Any, shape_tree: tp.Any, mesh: Mesh)` — [`L209`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L209) — Sanitize a partition-spec tree against concrete tensor shapes.

## Module values
- `AM` — [`L114`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L114)
- `OPTIMIZER_NAME` — [`L120`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L120)
- `OPTIMIZER_STRUCT_NAME` — [`L123`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L123)
- `RESUME_MODEL_SUBDIR` — [`L129`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L129)
- `TX_STRUCT_JSON` — [`L126`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L126)
- `WEIGHTS_NAME` — [`L117`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L117)
- `logger` — [`L132`](../../../../../../raw/code/EasyDeL/easydel/infra/base_state.py#L132)

