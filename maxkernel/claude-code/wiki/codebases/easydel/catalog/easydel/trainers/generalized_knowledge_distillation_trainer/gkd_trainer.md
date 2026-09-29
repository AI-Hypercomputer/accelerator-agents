---
title: 'Module: easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.generalized_knowledge_distillation_trainer.gkd_trainer`/
symbols:
  GKDTrainer.configure_functions: GKDTrainer#configure_functions().
  GKDTrainer.arguments: GKDTrainer#arguments.
  GKDTrainer.teacher_state: GKDTrainer#teacher_state.
  GKDTrainer.generation_config: GKDTrainer#generation_config.
  GKDTrainer._preprocess_batch_input: GKDTrainer#_preprocess_batch_input().
  GKDTrainer._apply_generation: GKDTrainer#_apply_generation().
  GKDTrainer.__init__: GKDTrainer#__init__().
  GKDTrainer.gkd_generate_function: GKDTrainer#gkd_generate_function.
  GKDTrainer.pad_token_id: GKDTrainer#pad_token_id.
  GKDTrainer._build_batch_from_sequences: GKDTrainer#_build_batch_from_sequences().
  logger: logger.
  GKDTrainer._should_use_student_sampling: GKDTrainer#_should_use_student_sampling().
  GKDTrainer: GKDTrainer#
  GKDTrainer.lmbda: GKDTrainer#lmbda.
  GKDTrainer.seq_kd: GKDTrainer#seq_kd.
  GKDTrainer._extract_prompts: GKDTrainer#_extract_prompts().
  GKDTrainer._warned_missing_prompt: GKDTrainer#_warned_missing_prompt.
  GKDTrainer._on_policy_rng: GKDTrainer#_on_policy_rng.
---
# Module: [`easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py)

## Classes
### `GKDTrainer`  ·  implements/extends SFTTrainer
- def: [`easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py:50`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L50)
- doc: Generalized Knowledge Distillation trainer with optional on-policy sampling.
- signature: `class GKDTrainer(SFTTrainer):`
- members:
  - `_apply_generation(self, generator_state: EasyDeLState, batch: dict[str, jax.Array], *, source: str)` — [`L263`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L263) — Generate completions using the specified model.
  - `_build_batch_from_sequences(self, original_batch: dict[str, jax.Array], sequences: jax.Array, *, prompt_seq_len: int)` — [`L360`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L360) — Construct training batch from generated sequences.
  - `_extract_prompts(self, batch: dict[str, jax.Array])` — [`L306`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L306) — Extract prompt token IDs and masks from batch.
  - `_preprocess_batch_input(self, state: EasyDeLState, batch: dict[str, jax.Array], is_train: bool)` — [`L215`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L215) — Optionally generate on-policy samples before training.
  - `_should_use_student_sampling(self)` — [`L253`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L253) — Determine whether to use student on-policy sampling for this batch.
  - `configure_functions(self)` — [`L142`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L142) — Configure JIT-compiled training and evaluation functions.
  - `arguments` — [`L68`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L68)
  - `generation_config` — [`L125`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L125)
  - `gkd_generate_function` — [`L93`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L93)
  - `lmbda` — [`L89`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L89)
  - `pad_token_id` — [`L121`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L121)
  - `seq_kd` — [`L90`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L90)
  - `teacher_state` — [`L102`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L102)
- protocol/private: `__init__`[`L70`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L70), `_on_policy_rng`[`L91`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L91), `_warned_missing_prompt`[`L92`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L92)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`model`](../../infra/base_state.md#EasyDeLState.model), [`flops_per_token`](../../infra/base_module.md#EasyDeLBaseModule.flops_per_token), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`model_state`](../base_trainer.md#BaseTrainer.model_state), [`mesh`](../../infra/base_module.md#EasyDeLBaseModule.mesh), [`state_shardings`](../trainer_protocol.md#BaseTrainerProtocol.state_shardings), [`TrainerConfigureFunctionOutput`](../trainer_protocol.md#TrainerConfigureFunctionOutput), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`to_state`](../../infra/base_module.md#EasyDeLBaseModule.to_state), [`model`](../base_trainer.md#BaseTrainer.model), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`resolve_straight_through_emulator`](../training_utils.md#resolve_straight_through_emulator), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`create_generate_function`](../base_trainer.md#BaseTrainer.create_generate_function), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`quantization_mode`](../training_configurations.md#TrainingArguments.quantization_mode), [`quantization_bits`](../training_configurations.md#TrainingArguments.quantization_bits), [`quantization_group_size`](../training_configurations.md#TrainingArguments.quantization_group_size), [`get_streaming_checkpointer`](../training_configurations.md#TrainingArguments.get_streaming_checkpointer), [`gkd_step`](_fn.md#gkd_step), [`__init__`](../supervised_fine_tuning_trainer/sft_trainer.md#SFTTrainer.__init__), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`_train_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._train_shared_fn_extra_args), [`_eval_shared_fn_extra_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_extra_args), [`deepcopy_model`](../../utils/traversals.md#deepcopy_model), [`shardings`](../../infra/base_state.md#EasyDeLState.shardings), [`ensure_checkpoint_path`](../training_configurations.md#TrainingArguments.ensure_checkpoint_path), [`straight_through_emulator`](../training_configurations.md#TrainingArguments.straight_through_emulator), [`tensor_straight_through`](../training_configurations.md#TrainingArguments.tensor_straight_through), [`_purify_batch`](../base_trainer.md#BaseTrainer._purify_batch), [`GKDConfig`](gkd_config.md#GKDConfig), [`SFTTrainer`](../supervised_fine_tuning_trainer/sft_trainer.md#SFTTrainer), [`_extra_forward_flops_per_token`](../trainer_protocol.md#BaseTrainerProtocol._extra_forward_flops_per_token), [`sharded_evaluation_step_function`](../trainer_protocol.md#BaseTrainerProtocol.sharded_evaluation_step_function), [`sharded_training_step_function`](../trainer_protocol.md#BaseTrainerProtocol.sharded_training_step_function), [`logger`](gkd_trainer.md#logger)  (+8 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`configure_functions`](../trainer/trainer.md#Trainer.configure_functions), [`SFTTrainer`](../supervised_fine_tuning_trainer/sft_trainer.md#SFTTrainer), [`_preprocess_batch_input`](../supervised_fine_tuning_trainer/sft_trainer.md#SFTTrainer._preprocess_batch_input)

## Module values
- `logger` — [`L46`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/gkd_trainer.py#L46)

