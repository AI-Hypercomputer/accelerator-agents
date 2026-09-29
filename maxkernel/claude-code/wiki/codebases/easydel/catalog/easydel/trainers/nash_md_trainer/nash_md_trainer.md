---
title: 'Module: easydel/trainers/nash_md_trainer/nash_md_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/nash_md_trainer/nash_md_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.nash_md_trainer.nash_md_trainer`/
symbols:
  NashMDTrainer._preprocess_batch_input: NashMDTrainer#_preprocess_batch_input().
  NashMDTrainer.configure_functions: NashMDTrainer#configure_functions().
  NashMDTrainer.compute_refmodel_logps: NashMDTrainer#compute_refmodel_logps.
  NashMDTrainer._score_rewards: NashMDTrainer#_score_rewards().
  NashMDTrainer.arguments: NashMDTrainer#arguments.
  NashMDTrainer.__init__: NashMDTrainer#__init__().
  NashMDTrainer._schedule_value: NashMDTrainer#_schedule_value().
  NashMDTrainer._compute_model_logps: NashMDTrainer#_compute_model_logps().
  NashMDTrainer: NashMDTrainer#
  _ensure_state: _ensure_state().
  NashMDTrainer._current_beta_value: NashMDTrainer#_current_beta_value().
  NashMDTrainer._current_mixture_coef: NashMDTrainer#_current_mixture_coef().
  NashMDTrainer.ref_state: NashMDTrainer#ref_state.
  NashMDTrainer._beta_schedule: NashMDTrainer#_beta_schedule.
  NashMDTrainer._mixture_schedule: NashMDTrainer#_mixture_schedule.
  NashMDTrainer.missing_eos_penalty: NashMDTrainer#missing_eos_penalty.
  NashMDTrainer._train_shared_fn_extra_args: NashMDTrainer#_train_shared_fn_extra_args().
  NashMDTrainer._eval_shared_fn_extra_args: NashMDTrainer#_eval_shared_fn_extra_args().
  logger: logger.
  NashMDTrainer.num_generations: NashMDTrainer#num_generations.
---
# Module: [`easydel/trainers/nash_md_trainer/nash_md_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py)

## Classes
### `NashMDTrainer`  ·  implements/extends GRPOTrainer
- def: [`easydel/trainers/nash_md_trainer/nash_md_trainer.py:68`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L68)
- doc: Nash Mirror Descent trainer for preference optimization.
- signature: `class NashMDTrainer(GRPOTrainer):`
- members:
  - `_current_beta_value(self)` — [`L147`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L147) — Get current beta value from schedule.
  - `_current_mixture_coef(self)` — [`L155`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L155) — Get current mixture coefficient from schedule.
  - `_preprocess_batch_input(self, state: EasyDeLState, batch: dict[str, jax.Array], is_train: bool)` — [`L341`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L341) — Generate model and mixture completions, compute rewards and probabilities.
  - `_schedule_value(self, schedule: tp.Any, default: float)` — [`L124`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L124) — Get current value from schedule based on training epoch.
  - `_score_rewards(self, prompts: list[str], completions: list[str], *, raw_text: list[str] | None = None, reasoning: list[str | None] | None = None, tool_calls: list[tp.Any | None] | None = None, batch: dict[str, tp.Any] | None = None)` — [`L266`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L266) — Compute reward scores for prompt-completion pairs.
  - `configure_functions(self)` — [`L171`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L171) — Configure JIT-compiled training and evaluation functions.
  - `arguments` — [`L86`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L86)
  - `compute_refmodel_logps` — [`L244`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L244)
  - `missing_eos_penalty` — [`L117`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L117)
  - `num_generations` — [`L118`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L118)
  - `ref_state` — [`L113`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L113)
- protocol/private: `__init__`[`L88`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L88), `_beta_schedule`[`L115`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L115), `_compute_model_logps`[`L221`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L221), `_eval_shared_fn_extra_args`[`L168`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L168), `_mixture_schedule`[`L116`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L116), `_train_shared_fn_extra_args`[`L164`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L164)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`generate_unified`](../base_trainer.md#BaseTrainer.generate_unified), [`Registry`](../../utils/registery.md#Registry), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`_all_gather`](../base_trainer.md#BaseTrainer._all_gather), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`model_state`](../base_trainer.md#BaseTrainer.model_state), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`capture_time`](../../utils/helpers.md#capture_time), [`register`](../../utils/registery.md#Registry.register), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`graphstate`](../../infra/base_state.md#EasyDeLState.graphstate), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`model`](../base_trainer.md#BaseTrainer.model), [`GRPOTrainer`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer), [`step`](../../infra/base_state.md#EasyDeLState.step), [`__init__`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.__init__), [`_log_training_generations_to_wandb`](../base_trainer.md#BaseTrainer._log_training_generations_to_wandb), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`resolve_straight_through_emulator`](../training_utils.md#resolve_straight_through_emulator), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`get_per_token_logps`](../group_relative_policy_optimization/_fn.md#get_per_token_logps), [`reward_funcs`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.reward_funcs), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`quantization_mode`](../training_configurations.md#TrainingArguments.quantization_mode), [`quantization_bits`](../training_configurations.md#TrainingArguments.quantization_bits), [`quantization_group_size`](../training_configurations.md#TrainingArguments.quantization_group_size), [`get_streaming_checkpointer`](../training_configurations.md#TrainingArguments.get_streaming_checkpointer), [`nash_md_step`](_fn.md#nash_md_step), [`processing_class`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.processing_class), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`_coerce_generation_texts`](../base_trainer.md#BaseTrainer._coerce_generation_texts), [`shardings`](../../infra/base_state.md#EasyDeLState.shardings), [`graphother`](../../infra/base_state.md#EasyDeLState.graphother), [`max_training_steps`](../trainer_protocol.md#BaseTrainerProtocol.max_training_steps), [`max_prompt_length`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig.max_prompt_length), [`text`](../base_trainer.md#GenerationResults.text)  (+25 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_preprocess_batch_input`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer._preprocess_batch_input), [`configure_functions`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.configure_functions), [`GRPOTrainer`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer), [`_train_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._train_shared_fn_extra_args), [`_eval_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_extra_args)

## Functions
- `_ensure_state(model: EasyDeLBaseModule | EasyDeLState | None)` — [`L51`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L51) — Convert model to EasyDeLState if needed.

## Module values
- `logger` — [`L45`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/nash_md_trainer.py#L45)

