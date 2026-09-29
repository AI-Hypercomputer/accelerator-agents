---
title: 'Module: easydel/trainers/xpo_trainer/xpo_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/xpo_trainer/xpo_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.xpo_trainer.xpo_trainer`/
symbols:
  XPOTrainer._preprocess_batch_input: XPOTrainer#_preprocess_batch_input().
  XPOTrainer.configure_functions: XPOTrainer#configure_functions().
  XPOTrainer._score_rewards: XPOTrainer#_score_rewards().
  XPOTrainer.arguments: XPOTrainer#arguments.
  XPOTrainer.on_step_end: XPOTrainer#on_step_end().
  XPOTrainer._schedule_value: XPOTrainer#_schedule_value().
  XPOTrainer: XPOTrainer#
  XPOTrainer.__init__: XPOTrainer#__init__().
  XPOTrainer.ref_state: XPOTrainer#ref_state.
  XPOTrainer._get_reward_processing_classes: XPOTrainer#_get_reward_processing_classes().
  _ensure_state: _ensure_state().
  XPOTrainer.loss_type_id: XPOTrainer#loss_type_id.
  XPOTrainer._current_beta_value: XPOTrainer#_current_beta_value().
  XPOTrainer._current_alpha_value: XPOTrainer#_current_alpha_value().
  XPOTrainer._gather_scalar: XPOTrainer#_gather_scalar().
  XPOTrainer._beta_schedule: XPOTrainer#_beta_schedule.
  XPOTrainer._alpha_schedule: XPOTrainer#_alpha_schedule.
---
# Module: [`easydel/trainers/xpo_trainer/xpo_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py)

## Classes
### `XPOTrainer`  ·  implements/extends GRPOTrainer
- def: [`easydel/trainers/xpo_trainer/xpo_trainer.py:55`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L55)
- doc: Trainer for Exploratory Preference Optimization (XPO).
- signature: `class XPOTrainer(GRPOTrainer):`
- members:
  - `__init__(self, arguments: XPOConfig, model: EasyDeLBaseModule | EasyDeLState, reward_funcs: collections.abc.Sequence[tp.Callable] | tp.Callable, *, reference_model: EasyDeLBaseModule | EasyDeLState | None = None, train_dataset: Dataset | IterableDataset | None = None, eval_dataset: Dataset | dict[str, Dataset] | IterableDataset | None = None, processing_class: ProcessingClassType, reward_processing_classes: ProcessingClassType | list[ProcessingClassType] | None = None)` — [`L70`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L70) — Initialize the XPO trainer.
  - `_current_alpha_value(self)` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L160) — Get the current alpha (exploratory weight) value based on training progress.
  - `_current_beta_value(self)` — [`L152`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L152) — Get the current beta (KL penalty scaling) value based on training progress.
  - `_gather_scalar(self, value: float, batch_size: int)` — [`L338`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L338) — Create a scalar array and gather across all devices.
  - `_get_reward_processing_classes(self)` — [`L118`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L118) — Normalize reward processing classes to a list aligned to reward functions.
  - `_preprocess_batch_input(self, state: EasyDeLState, batch: dict[str, jax.Array], is_train: bool)` — [`L351`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L351) — Preprocess a batch by generating completions and computing rewards.
  - `_schedule_value(self, schedule: tp.Any, default: float)` — [`L131`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L131) — Resolve a scheduled value based on current training progress.
  - `_score_rewards(self, prompt_ids: jax.Array, prompt_mask: jax.Array, completion_ids: jax.Array, completion_mask: jax.Array, *, prompt_texts: list[str] | None, completion_texts: list[str] | None, raw_text: list[str] | None = None, reasoning: list[str | None] | None = None, tool_calls: list[tp.Any | None] | None = None)` — [`L235`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L235) — Score completions using configured reward functions.
  - `configure_functions(self)` — [`L168`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L168) — Configure and compile the training and evaluation step functions.
  - `on_step_end(self, state: EasyDeLState, metrics: MetricsType, step: int)` — [`L561`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L561) — Hook called at the end of each training step.
  - `arguments` — [`L68`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L68)
  - `loss_type_id` — [`L100`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L100)
  - `ref_state` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L96)
- protocol/private: `_alpha_schedule`[`L99`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L99), `_beta_schedule`[`L98`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L98)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`generate_unified`](../base_trainer.md#BaseTrainer.generate_unified), [`Registry`](../../utils/registery.md#Registry), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`_all_gather`](../base_trainer.md#BaseTrainer._all_gather), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`model_state`](../base_trainer.md#BaseTrainer.model_state), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`capture_time`](../../utils/helpers.md#capture_time), [`register`](../../utils/registery.md#Registry.register), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`graphstate`](../../infra/base_state.md#EasyDeLState.graphstate), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`model`](../base_trainer.md#BaseTrainer.model), [`GRPOTrainer`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer), [`step`](../../infra/base_state.md#EasyDeLState.step), [`__init__`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.__init__), [`_log_training_generations_to_wandb`](../base_trainer.md#BaseTrainer._log_training_generations_to_wandb), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`resolve_straight_through_emulator`](../training_utils.md#resolve_straight_through_emulator), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`xpo_step`](_fn.md#xpo_step), [`reward_funcs`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.reward_funcs), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`quantization_mode`](../training_configurations.md#TrainingArguments.quantization_mode), [`quantization_bits`](../training_configurations.md#TrainingArguments.quantization_bits), [`quantization_group_size`](../training_configurations.md#TrainingArguments.quantization_group_size), [`get_streaming_checkpointer`](../training_configurations.md#TrainingArguments.get_streaming_checkpointer), [`processing_class`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.processing_class), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`MetricsType`](../training_configurations.md#MetricsType), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`_train_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._train_shared_fn_extra_args), [`_coerce_generation_texts`](../base_trainer.md#BaseTrainer._coerce_generation_texts), [`_eval_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_extra_args), [`deepcopy_model`](../../utils/traversals.md#deepcopy_model), [`shardings`](../../infra/base_state.md#EasyDeLState.shardings), [`graphother`](../../infra/base_state.md#EasyDeLState.graphother)  (+30 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_preprocess_batch_input`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer._preprocess_batch_input), [`configure_functions`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.configure_functions), [`GRPOTrainer`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer), [`on_step_end`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.on_step_end)

## Functions
- `_ensure_state(model: EasyDeLBaseModule | EasyDeLState)` — [`L42`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/xpo_trainer.py#L42) — Convert a model to EasyDeLState if it isn't already.

