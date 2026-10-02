---
title: 'Module: easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.agentic_moshpit.agentic_moshpit_trainer`/
symbols:
  AgenticMoshPitTrainer._preprocess_batch_input: AgenticMoshPitTrainer#_preprocess_batch_input().
  AgenticMoshPitTrainer._score_auxiliary_rewards: AgenticMoshPitTrainer#_score_auxiliary_rewards().
  AgenticMoshPitTrainer.generate_fn: AgenticMoshPitTrainer#generate_fn().
  AgenticMoshPitTrainer.arguments: AgenticMoshPitTrainer#arguments.
  AgenticMoshPitTrainer.wrapped_env_factory: AgenticMoshPitTrainer#wrapped_env_factory().
  AgenticMoshPitTrainer.__init__: AgenticMoshPitTrainer#__init__().
  AgenticMoshPitTrainer._tools: AgenticMoshPitTrainer#_tools.
  AgenticMoshPitTrainer._tool_call_parser: AgenticMoshPitTrainer#_tool_call_parser.
  AgenticMoshPitTrainer._rollout_manager: AgenticMoshPitTrainer#_rollout_manager.
  AgenticMoshPitTrainer._wrap_env_with_tools: AgenticMoshPitTrainer#_wrap_env_with_tools().
  AgenticMoshPitTrainer._log_trajectories_to_wandb: AgenticMoshPitTrainer#_log_trajectories_to_wandb().
  AgenticMoshPitTrainer: AgenticMoshPitTrainer#
  AgenticMoshPitTrainer.create_tfds_collect_function: AgenticMoshPitTrainer#create_tfds_collect_function().
  AgenticMoshPitTrainer.create_grain_collect_function: AgenticMoshPitTrainer#create_grain_collect_function().
  AgenticMoshPitTrainer._apply_auxiliary_rewards_to_trajectories: AgenticMoshPitTrainer#_apply_auxiliary_rewards_to_trajectories().
  RewardFunc: RewardFunc.
  AgenticMoshPitTrainer._strip_thinking: AgenticMoshPitTrainer#_strip_thinking.
  AgenticMoshPitTrainer._make_generate_fn: AgenticMoshPitTrainer#_make_generate_fn().
  _InfiniteRolloutDataset._row: _InfiniteRolloutDataset#_row.
  _InfiniteRolloutDataset.__getitem__: _InfiniteRolloutDataset#__getitem__().
  _InfiniteRolloutDataset._len: _InfiniteRolloutDataset#_len.
  _InfiniteRolloutDataset.__iter__: _InfiniteRolloutDataset#__iter__().
  _InfiniteRolloutDataset.__len__: _InfiniteRolloutDataset#__len__().
  _InfiniteRolloutDataset.column_names: _InfiniteRolloutDataset#column_names().
  _InfiniteRolloutDataset.num_rows: _InfiniteRolloutDataset#num_rows().
  AgenticMoshPitTrainer._rollout_step: AgenticMoshPitTrainer#_rollout_step.
  logger: logger.
  _InfiniteRolloutDataset: _InfiniteRolloutDataset#
  AgenticMoshPitTrainer.env_factory: AgenticMoshPitTrainer#env_factory.
  AgenticMoshPitTrainer._env_reward_placeholder: AgenticMoshPitTrainer#_env_reward_placeholder().
  AgenticMoshPitTrainer._is_env_reward_placeholder: AgenticMoshPitTrainer#_is_env_reward_placeholder().
  _InfiniteRolloutDataset.__init__: _InfiniteRolloutDataset#__init__().
  _InfiniteRolloutDataset._batch_size: _InfiniteRolloutDataset#_batch_size.
---
# Module: [`easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py)

## Classes
### `AgenticMoshPitTrainer`  ·  implements/extends GRPOTrainer
- def: [`easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py:126`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L126)
- doc: Agentic MoshPit Trainer for multi-turn RL training with environments.
- signature: `class AgenticMoshPitTrainer(GRPOTrainer):`
- members:
  - `_env_reward_placeholder(prompts, completions, **kwargs)` — [`L258`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L258) — Placeholder reward function — actual rewards come from environments.
  - `_log_trajectories_to_wandb(self, trajectories: list, state: EasyDeLState)` — [`L715`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L715) — Log full trajectory conversations to Weights & Biases.
  - `_make_generate_fn(self, state: EasyDeLState)` — [`L262`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L262) — Create a batched generation function for rollouts.
  - `_preprocess_batch_input(self, state: EasyDeLState, batch: dict[str, jax.Array], is_train: bool)` — [`L502`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L502) — Run agentic rollouts and prepare training batch.
  - `_wrap_env_with_tools(self, env: AgenticEnvironment)` — [`L491`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L491) — Wrap environment with tool support if tools are configured.
  - `create_grain_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L799`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L799) — Create a Grain data collator for agentic training.
  - `create_tfds_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end")` — [`L781`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L781) — Create a data collator for agentic training.
  - `generate_fn(prompts: list[str], *, temperature: float | None = None, top_p: float | None = None, top_k: int | None = None, num_return_sequences: int | None = None, strip_thinking: bool = False)` — [`L281`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L281)
  - `wrapped_env_factory()` — [`L530`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L530)
  - `arguments` — [`L183`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L183)
  - `env_factory` — [`L212`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L212)
- protocol/private: `__init__`[`L185`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L185), `_apply_auxiliary_rewards_to_trajectories`[`L465`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L465), `_is_env_reward_placeholder`[`L338`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L338), `_rollout_manager`[`L249`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L249), `_rollout_step`[`L213`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L213), `_score_auxiliary_rewards`[`L341`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L341), `_strip_thinking`[`L206`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L206), `_tool_call_parser`[`L225`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L225), `_tools`[`L215`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L215)
- uses (calls/refs, reference-scoped): [`config`](../../infra/base_module.md#EasyDeLBaseModule.config), [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`generate_unified`](../base_trainer.md#BaseTrainer.generate_unified), [`run_grouped_episodes`](env_manager.md#RolloutManager.run_grouped_episodes), [`model`](../../infra/base_state.md#EasyDeLState.model), [`Registry`](../../utils/registery.md#Registry), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`_all_gather`](../base_trainer.md#BaseTrainer._all_gather), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`capture_time`](../../utils/helpers.md#capture_time), [`register`](../../utils/registery.md#Registry.register), [`graphstate`](../../infra/base_state.md#EasyDeLState.graphstate), [`GRPOTrainer`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer), [`step`](../../infra/base_state.md#EasyDeLState.step), [`__init__`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.__init__), [`_log_training_generations_to_wandb`](../base_trainer.md#BaseTrainer._log_training_generations_to_wandb), [`ref_state`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.ref_state), [`reward_funcs`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.reward_funcs), [`processing_class`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.processing_class), [`Tool`](tools.md#Tool), [`compute_refmodel_logps`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.compute_refmodel_logps), [`apply_chat_template`](../prompt_utils.md#apply_chat_template), [`_coerce_generation_texts`](../base_trainer.md#BaseTrainer._coerce_generation_texts), [`AgenticEnvironment`](environment.md#AgenticEnvironment), [`graphother`](../../infra/base_state.md#EasyDeLState.graphother), [`normalize_generation_model_kwargs`](../training_utils.md#normalize_generation_model_kwargs), [`ref_logps_chunk_size`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.ref_logps_chunk_size), [`max_prompt_length`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig.max_prompt_length), [`text`](../base_trainer.md#GenerationResults.text), [`turn_record_to_message`](env_manager.md#turn_record_to_message), [`can_log_metrics`](../training_configurations.md#TrainingArguments.can_log_metrics), [`scale_rewards`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.scale_rewards), [`AgenticMoshPitConfig`](agentic_moshpit_config.md#AgenticMoshPitConfig), [`collate_trajectories`](env_manager.md#RolloutManager.collate_trajectories), [`pause_esurge`](../../infra/mixins/generation.md#EasyGenerationMixin.pause_esurge), [`reasoning`](../base_trainer.md#GenerationResults.reasoning), [`tool_calls`](../base_trainer.md#GenerationResults.tool_calls), [`graphdef`](../../infra/base_state.md#EasyDeLState.graphdef)  (+55 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_preprocess_batch_input`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer._preprocess_batch_input), [`GRPOTrainer`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer), [`create_grain_collect_function`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.create_grain_collect_function), [`create_tfds_collect_function`](../group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.create_tfds_collect_function)

### `_InfiniteRolloutDataset`
- def: [`easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py:83`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L83)
- doc: Infinite dummy dataset for environment-driven training.
- signature: `class _InfiniteRolloutDataset:`
- members:
  - `column_names(self)` — [`L117`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L117)
  - `num_rows(self)` — [`L121`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L121)
- protocol/private: `__getitem__`[`L108`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L108), `__init__`[`L96`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L96), `__iter__`[`L101`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L101), `__len__`[`L105`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L105), `_batch_size`[`L97`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L97), `_len`[`L99`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L99), `_row`[`L98`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L98)
- used by: [`_tool_call_parser`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._tool_call_parser)

## Module values
- `RewardFunc` — [`L80`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L80)
- `logger` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_trainer.py#L78)

