---
title: 'Module: easydel/trainers/group_filtered_policy_optimization/gfpo_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/group_filtered_policy_optimization/gfpo_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.group_filtered_policy_optimization.gfpo_trainer`/
symbols:
  GFPOTrainer._preprocess_batch_input: GFPOTrainer#_preprocess_batch_input().
  GFPOTrainer.configure_functions: GFPOTrainer#configure_functions().
  GFPOTrainer.arguments: GFPOTrainer#arguments.
  GFPOTrainer.__init__: GFPOTrainer#__init__().
  GFPOTrainer._default_filter_func: GFPOTrainer#_default_filter_func().
  RewardFunc: RewardFunc.
  GroupFilterFunc: GroupFilterFunc.
  GFPOTrainer._filter_completions: GFPOTrainer#_filter_completions().
  GFPOTrainer: GFPOTrainer#
  GFPOTrainer.group_filter_func: GFPOTrainer#group_filter_func.
---
# Module: [`easydel/trainers/group_filtered_policy_optimization/gfpo_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_trainer.py)

## Classes
### `GFPOTrainer`  ·  implements/extends GRPOTrainer
- def: [`easydel/trainers/group_filtered_policy_optimization/gfpo_trainer.py:52`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_trainer.py#L52)
- doc: Group Filtered Policy Optimization trainer for RLHF.
- signature: `class GFPOTrainer(GRPOTrainer):`
- members:
  - `__init__(self, arguments: GFPOConfig, model: EasyDeLBaseModule | EasyDeLState | None, reward_funcs: RewardFunc | list[RewardFunc], train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None, eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None, processing_class: ProcessingClassType = None, reward_processing_classes: ProcessingClassType = None, data_tokenize_fn: tp.Callable | None = None, group_filter_func: GroupFilterFunc | None = None)` — [`L99`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_trainer.py#L99) — Initialize the GFPO trainer.
  - `_default_filter_func(self, completion_ids: jax.Array, rewards: jax.Array, completion_mask: jax.Array)` — [`L207`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_trainer.py#L207) — Default filter function based on length and reward-per-token efficiency.
  - `_filter_completions(self, prompt_ids: jax.Array, prompt_mask: jax.Array, completion_ids: jax.Array, completion_mask: jax.Array, ref_per_token_logps: jax.Array, rewards: jax.Array, rewards_per_func: jax.Array, completion_prompts: list[str], num_generations: int, num_remains: int)` — [`L245`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_trainer.py#L245) — Filter completions to keep top-K samples per group.
  - `_preprocess_batch_input(self, state: EasyDeLState, batch: dict[str, jax.Array], is_train: bool)` — [`L323`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_trainer.py#L323) — Preprocess batch with GFPO filtering.
  - `configure_functions(self)` — [`L153`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_trainer.py#L153) — Configure training functions with filtered generation count.
  - `arguments` — [`L97`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_trainer.py#L97)
  - `group_filter_func` — [`L140`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_trainer.py#L140)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`generate_unified`](../base_trainer.md#BaseTrainer.generate_unified), [`model`](../../infra/base_state.md#EasyDeLState.model), [`configure_functions`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.configure_functions), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`_all_gather`](../base_trainer.md#BaseTrainer._all_gather), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`capture_time`](../../utils/helpers.md#capture_time), [`graphstate`](../../infra/base_state.md#EasyDeLState.graphstate), [`step_partition_spec`](../training_configurations.md#TrainingArguments.step_partition_spec), [`GRPOTrainer`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer), [`__init__`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.__init__), [`_log_training_generations_to_wandb`](../base_trainer.md#BaseTrainer._log_training_generations_to_wandb), [`loss_config`](../training_configurations.md#TrainingArguments.loss_config), [`gradient_accumulation_steps`](../training_configurations.md#TrainingArguments.gradient_accumulation_steps), [`ref_state`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.ref_state), [`reward_funcs`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.reward_funcs), [`scheduler`](../trainer_protocol.md#BaseTrainerProtocol.scheduler), [`processing_class`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.processing_class), [`_train_shared_fn_static_args`](../base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`_eval_shared_fn_static_args`](../base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`compute_refmodel_logps`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.compute_refmodel_logps), [`apply_chat_template`](../prompt_utils.md#apply_chat_template), [`_coerce_generation_texts`](../base_trainer.md#BaseTrainer._coerce_generation_texts), [`graphother`](../../infra/base_state.md#EasyDeLState.graphother), [`normalize_generation_model_kwargs`](../training_utils.md#normalize_generation_model_kwargs), [`ref_logps_chunk_size`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.ref_logps_chunk_size), [`text`](../base_trainer.md#GenerationResults.text), [`_eos_token_id`](../base_trainer.md#BaseTrainer._eos_token_id), [`scale_rewards`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.scale_rewards), [`GFPOConfig`](gfpo_config.md#GFPOConfig), [`_pad_token_id`](../base_trainer.md#BaseTrainer._pad_token_id), [`_decode_prompt_batch`](../base_trainer.md#BaseTrainer._decode_prompt_batch), [`_purify_batch`](../base_trainer.md#BaseTrainer._purify_batch), [`completion_ids`](../base_trainer.md#GenerationResults.completion_ids), [`eval_is_conversational`](../base_trainer.md#BaseTrainer.eval_is_conversational), [`reasoning`](../base_trainer.md#GenerationResults.reasoning), [`tool_calls`](../base_trainer.md#GenerationResults.tool_calls)  (+40 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_preprocess_batch_input`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer._preprocess_batch_input), [`configure_functions`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.configure_functions), [`GRPOTrainer`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer)

## Module values
- `GroupFilterFunc` — [`L48`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_trainer.py#L48)
- `RewardFunc` — [`L47`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_filtered_policy_optimization/gfpo_trainer.py#L47)

