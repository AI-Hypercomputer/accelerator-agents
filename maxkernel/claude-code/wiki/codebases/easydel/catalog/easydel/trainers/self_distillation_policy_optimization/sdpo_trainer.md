---
title: 'Module: easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.self_distillation_policy_optimization.sdpo_trainer`/
symbols:
  SDPOTrainer._preprocess_batch_input: SDPOTrainer#_preprocess_batch_input().
  SDPOTrainer.configure_functions: SDPOTrainer#configure_functions().
  SDPOTrainer.arguments: SDPOTrainer#arguments.
  SDPOTrainer._configure_teacher_context: SDPOTrainer#_configure_teacher_context().
  SDPOTrainer.compute_refmodel_logps: SDPOTrainer#compute_refmodel_logps.
  SDPOTrainer.on_step_end: SDPOTrainer#on_step_end().
  SDPOTrainer.__init__: SDPOTrainer#__init__().
  SDPOTrainer._compute_refmodel_logps: SDPOTrainer#_compute_refmodel_logps().
  _build_feedback_separator: _build_feedback_separator().
  SDPOTrainer.teacher_prompt_length: SDPOTrainer#teacher_prompt_length.
  SDPOTrainer: SDPOTrainer#
  SDPOTrainer._effective_feedback_length: SDPOTrainer#_effective_feedback_length.
  RewardFunc: RewardFunc.
  SDPOTrainer._get_rich_feedback: SDPOTrainer#_get_rich_feedback().
  SDPOTrainer._tokenize_feedback_separators: SDPOTrainer#_tokenize_feedback_separators().
  SDPOTrainer._ensure_non_empty_prompts: SDPOTrainer#_ensure_non_empty_prompts().
  SDPOTrainer.ref_state: SDPOTrainer#ref_state.
  SDPOTrainer._get_self_feedback: SDPOTrainer#_get_self_feedback().
  logger: logger.
  SDPOTrainer.feedback_func: SDPOTrainer#feedback_func.
  FeedbackFunc: FeedbackFunc.
  _FEEDBACK_CORRECT: _FEEDBACK_CORRECT.
  _FEEDBACK_TEMPLATE_SOLUTION: _FEEDBACK_TEMPLATE_SOLUTION.
  _FEEDBACK_TEMPLATE_ENV: _FEEDBACK_TEMPLATE_ENV.
  _FEEDBACK_TEMPLATE_SOLVE: _FEEDBACK_TEMPLATE_SOLVE.
  SDPOTrainer._resolve_model_context_window: SDPOTrainer#_resolve_model_context_window().
---
# Module: [`easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py)

## Classes
### `SDPOTrainer`  ·  implements/extends GRPOTrainer
- def: [`easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py:119`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L119)
- doc: Self-Distillation Policy Optimization trainer.
- signature: `class SDPOTrainer(GRPOTrainer):`
- members:
  - `_configure_teacher_context(self)` — [`L215`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L215) — Cap feedback tokens so prompt+feedback+completion fits model context.
  - `_ensure_non_empty_prompts(self, prompt_ids: jax.Array, prompt_mask: jax.Array)` — [`L448`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L448) — Ensure each prompt row has at least one visible token.
  - `_get_rich_feedback(self, completion_prompts: list, completions: list, rewards: jax.Array)` — [`L385`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L385) — Get feedback strings from the user-supplied ``feedback_func``.
  - `_get_self_feedback(self, completions: list[str], rewards: jax.Array, generation_factor: int)` — [`L338`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L338) — Derive feedback without a rich-feedback function.
  - `_preprocess_batch_input(self, state: EasyDeLState, batch: dict[str, jax.Array], is_train: bool)` — [`L476`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L476) — Build the SDPO batch from raw prompts.
  - `_resolve_model_context_window(config: tp.Any)` — [`L198`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L198) — Best-effort extraction of model context window from config.
  - `_tokenize_feedback_separators(self, feedback_texts: list[str])` — [`L421`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L421) — Tokenise feedback separator strings and pad to effective feedback length.
  - `configure_functions(self)` — [`L249`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L249) — Configure and JIT-compile the SDPO training / evaluation steps.
  - `on_step_end(self, state: EasyDeLState, metrics: MetricsType, step: int)` — [`L743`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L743) — Post-step hook - syncs reference model when requested.
  - `arguments` — [`L164`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L164)
  - `compute_refmodel_logps` — [`L313`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L313)
  - `feedback_func` — [`L181`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L181)
  - `ref_state` — [`L761`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L761)
  - `teacher_prompt_length` — [`L183`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L183)
- protocol/private: `__init__`[`L166`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L166), `_compute_refmodel_logps`[`L296`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L296), `_effective_feedback_length`[`L182`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L182)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`generate_unified`](../base_trainer.md#BaseTrainer.generate_unified), [`model`](../../infra/base_state.md#EasyDeLState.model), [`Registry`](../../utils/registery.md#Registry), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`_all_gather`](../base_trainer.md#BaseTrainer._all_gather), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`model_state`](../base_trainer.md#BaseTrainer.model_state), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`capture_time`](../../utils/helpers.md#capture_time), [`register`](../../utils/registery.md#Registry.register), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`graphstate`](../../infra/base_state.md#EasyDeLState.graphstate), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`model`](../base_trainer.md#BaseTrainer.model), [`GRPOTrainer`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer), [`__init__`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.__init__), [`_log_training_generations_to_wandb`](../base_trainer.md#BaseTrainer._log_training_generations_to_wandb), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`resolve_straight_through_emulator`](../training_utils.md#resolve_straight_through_emulator), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`get_per_token_logps`](../group_relative_policy_optimization/_fn.md#get_per_token_logps), [`reward_funcs`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.reward_funcs), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`quantization_mode`](../training_configurations.md#TrainingArguments.quantization_mode), [`quantization_bits`](../training_configurations.md#TrainingArguments.quantization_bits), [`quantization_group_size`](../training_configurations.md#TrainingArguments.quantization_group_size), [`get_streaming_checkpointer`](../training_configurations.md#TrainingArguments.get_streaming_checkpointer), [`sdpo_step`](_fn.md#sdpo_step), [`processing_class`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.processing_class), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`MetricsType`](../training_configurations.md#MetricsType), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`apply_chat_template`](../prompt_utils.md#apply_chat_template), [`_coerce_generation_texts`](../base_trainer.md#BaseTrainer._coerce_generation_texts), [`deepcopy_model`](../../utils/traversals.md#deepcopy_model), [`shardings`](../../infra/base_state.md#EasyDeLState.shardings)  (+49 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_preprocess_batch_input`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer._preprocess_batch_input), [`configure_functions`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.configure_functions), [`GRPOTrainer`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer), [`on_step_end`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.on_step_end)

## Functions
- `_build_feedback_separator(*, is_successful: bool, env_feedback: str, correct_solution: str | None)` — [`L88`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L88) — Return the text block inserted between the prompt and the completion.

## Module values
- `FeedbackFunc` — [`L76`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L76)
- `RewardFunc` — [`L80`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L80)
- `_FEEDBACK_CORRECT` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L82)
- `_FEEDBACK_TEMPLATE_ENV` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L84)
- `_FEEDBACK_TEMPLATE_SOLUTION` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L83)
- `_FEEDBACK_TEMPLATE_SOLVE` — [`L85`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L85)
- `logger` — [`L74`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/sdpo_trainer.py#L74)

