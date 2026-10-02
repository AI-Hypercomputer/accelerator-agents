---
title: 'Module: easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.proximal_policy_optimization_trainer.ppo_trainer`/
symbols:
  PPOTrainer._preprocess_batch_input: PPOTrainer#_preprocess_batch_input().
  PPOTrainer.configure_functions: PPOTrainer#configure_functions().
  PPOTrainer.arguments: PPOTrainer#arguments.
  PPOTrainer.__init__: PPOTrainer#__init__().
  PPOTrainer.compute_rollout_logps_values: PPOTrainer#compute_rollout_logps_values.
  PPOTrainer.reward_funcs: PPOTrainer#reward_funcs.
  PPOTrainer.apply_fn: PPOTrainer#apply_fn().
  PPOTrainer._compute_refmodel_logps: PPOTrainer#_compute_refmodel_logps().
  PPOTrainer._compute_rollout_logps_values: PPOTrainer#_compute_rollout_logps_values().
  PPOTrainer.ref_state: PPOTrainer#ref_state.
  PPOTrainer._get_preprocess_transform: PPOTrainer#_get_preprocess_transform().
  RewardFunc: RewardFunc.
  PPOTrainer._compute_gae: PPOTrainer#_compute_gae().
  PPOTrainer.create_grain_collect_function: PPOTrainer#create_grain_collect_function().
  PPOTrainer.create_tfds_collect_function: PPOTrainer#create_tfds_collect_function().
  PPOTrainer.processing_class: PPOTrainer#processing_class.
  PPOTrainer._is_pretokenized: PPOTrainer#_is_pretokenized().
  PPOTrainer.reward_processing_classes: PPOTrainer#reward_processing_classes.
  PPOTrainer.compute_refmodel_logps: PPOTrainer#compute_refmodel_logps.
  PPOTrainer: PPOTrainer#
  PPOTrainer.reward_weights: PPOTrainer#reward_weights.
  PPOTrainer.num_generations: PPOTrainer#num_generations.
  PPOTrainer.data_tokenize_fn: PPOTrainer#data_tokenize_fn.
  PPOTrainer.padding_value: PPOTrainer#padding_value.
  PPOTrainer._masked_whiten: PPOTrainer#_masked_whiten().
  PPOTrainer.reward_func_names: PPOTrainer#reward_func_names.
  PPOTrainer.scan_fn: PPOTrainer#scan_fn().
  logger: logger.
---
# Module: [`easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py)

## Classes
### `PPOTrainer`  ·  implements/extends Trainer
- def: [`easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py:81`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L81)
- doc: Proximal Policy Optimization trainer for RLHF.
- signature: `class PPOTrainer(Trainer):`
- members:
  - `__init__(self, arguments: PPOConfig, model: EasyDeLBaseModule | EasyDeLState | None, reward_funcs: RewardFunc | list[RewardFunc], train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None, eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None, processing_class: ProcessingClassType | None = None, reward_processing_classes: ProcessingClassType | None = None, data_tokenize_fn: tp.Callable | None = None)` — [`L130`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L130) — Initialize the PPO trainer.
  - `_compute_gae(self, rewards: jax.Array, values: jax.Array, mask: jax.Array)` — [`L556`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L556) — Compute Generalized Advantage Estimation (GAE).
  - `_get_preprocess_transform(self)` — [`L270`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L270) — Get the preprocessing transform for ShardedDataSource.
  - `_is_pretokenized(self)` — [`L285`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L285) — Check if the dataset already has tokenized fields.
  - `_masked_whiten(self, x: jax.Array, mask: jax.Array, *, shift_mean: bool)` — [`L536`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L536) — Normalize array values with optional mean shifting.
  - `_preprocess_batch_input(self, state: EasyDeLState, batch: dict[str, jax.Array], is_train: bool)` — [`L607`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L607) — Preprocess a batch for PPO training. — documented in [easydel-trainers-base_trainer](../../../../concepts/easydel-trainers-base_trainer.md)
  - `apply_fn(gd, gs, gt, batch)` — [`L221`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L221)
  - `configure_functions(self)` — [`L341`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L341) — Configure and JIT-compile training and evaluation step functions.
  - `create_grain_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L299`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L299) — Create a data collator for Grain data loading.
  - `create_tfds_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L320`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L320) — Create a data collator for TFDS data loading.
  - `scan_fn(adv_next, inputs)` — [`L588`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L588)
  - `arguments` — [`L128`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L128)
  - `compute_refmodel_logps` — [`L451`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L451)
  - `compute_rollout_logps_values` — [`L514`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L514)
  - `data_tokenize_fn` — [`L259`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L259)
  - `num_generations` — [`L254`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L254)
  - `padding_value` — [`L191`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L191)
  - `processing_class` — [`L187`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L187)
  - `ref_state` — [`L180`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L180)
  - `reward_func_names` — [`L252`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L252)
  - `reward_funcs` — [`L195`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L195)
  - `reward_processing_classes` — [`L255`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L255)
  - `reward_weights` — [`L248`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L248)
- protocol/private: `_compute_refmodel_logps`[`L412`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L412), `_compute_rollout_logps_values`[`L463`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L463)
- uses (calls/refs, reference-scoped): [`config`](../../infra/base_module.md#EasyDeLBaseModule.config), [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`generate_unified`](../base_trainer.md#BaseTrainer.generate_unified), [`model`](../../infra/base_state.md#EasyDeLState.model), [`Trainer`](../trainer/trainer.md#Trainer), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`_all_gather`](../base_trainer.md#BaseTrainer._all_gather), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`open_shard`](../../data/core/protocols.md#ShardedDataSource.open_shard), `pad_token_id`, [`model_state`](../base_trainer.md#BaseTrainer.model_state), [`shard_names`](../../data/core/protocols.md#ShardedDataSource.shard_names), [`_train_source`](../base_trainer.md#BaseTrainer._train_source), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`capture_time`](../../utils/helpers.md#capture_time), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`__init__`](../base_trainer.md#BaseTrainer.__init__), [`graphstate`](../../infra/base_state.md#EasyDeLState.graphstate), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`to_state`](../../infra/base_module.md#EasyDeLBaseModule.to_state), [`model`](../base_trainer.md#BaseTrainer.model), [`_log_training_generations_to_wandb`](../base_trainer.md#BaseTrainer._log_training_generations_to_wandb), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`resolve_straight_through_emulator`](../training_utils.md#resolve_straight_through_emulator), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`compute_token_logps_and_entropies_chunked`](../_logprob_utils.md#compute_token_logps_and_entropies_chunked), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`filter_kwargs_for_callable`](../training_utils.md#filter_kwargs_for_callable), [`quantization_mode`](../training_configurations.md#TrainingArguments.quantization_mode), [`quantization_bits`](../training_configurations.md#TrainingArguments.quantization_bits), [`quantization_group_size`](../training_configurations.md#TrainingArguments.quantization_group_size), [`get_streaming_checkpointer`](../training_configurations.md#TrainingArguments.get_streaming_checkpointer), [`ppo_step`](_fn.md#ppo_step), [`sanitize_model_call_kwargs`](../training_utils.md#sanitize_model_call_kwargs), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`GRPOPreprocessTransform`](../prompt_transforms.md#GRPOPreprocessTransform)  (+58 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`configure_functions`](../trainer/trainer.md#Trainer.configure_functions), [`Trainer`](../trainer/trainer.md#Trainer), [`_is_pretokenized`](../base_trainer.md#BaseTrainer._is_pretokenized), [`_get_preprocess_transform`](../base_trainer.md#BaseTrainer._get_preprocess_transform), [`create_grain_collect_function`](../trainer/trainer.md#Trainer.create_grain_collect_function), [`create_tfds_collect_function`](../trainer/trainer.md#Trainer.create_tfds_collect_function), [`_preprocess_batch_input`](../base_trainer.md#BaseTrainer._preprocess_batch_input)

## Module values
- `RewardFunc` — [`L77`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L77)
- `logger` — [`L76`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_trainer.py#L76)

