---
title: 'Module: easydel/trainers/agentic_moshpit/agentic_moshpit_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/agentic_moshpit/agentic_moshpit_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.agentic_moshpit.agentic_moshpit_config`/AgenticMoshPitConfig#
symbols:
  AgenticMoshPitConfig: ''
  AgenticMoshPitConfig.__post_init__: __post_init__().
  AgenticMoshPitConfig.advantage_estimator: advantage_estimator.
  AgenticMoshPitConfig.group_size: group_size.
  AgenticMoshPitConfig.tool_schemas: tool_schemas.
  AgenticMoshPitConfig.num_env_groups: num_env_groups.
  AgenticMoshPitConfig.reward_mode: reward_mode.
  AgenticMoshPitConfig.step_reward_gamma: step_reward_gamma.
  AgenticMoshPitConfig.tool_names: tool_names.
  AgenticMoshPitConfig.max_steps: max_steps.
  AgenticMoshPitConfig.episode_reward_weight: episode_reward_weight.
  AgenticMoshPitConfig.step_reward_weight: step_reward_weight.
  AgenticMoshPitConfig.system_prompt: system_prompt.
  AgenticMoshPitConfig.tool_caller: tool_caller.
  AgenticMoshPitConfig.max_tool_calls_per_step: max_tool_calls_per_step.
  AgenticMoshPitConfig.reasoning_parser: reasoning_parser.
  AgenticMoshPitConfig.trainer_prefix: trainer_prefix.
  AgenticMoshPitConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/agentic_moshpit/agentic_moshpit_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py)

## Classes
### `AgenticMoshPitConfig`  ·  implements/extends GRPOConfig
- def: [`easydel/trainers/agentic_moshpit/agentic_moshpit_config.py:34`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L34)
- doc: Configuration class for Agentic MoshPit training.
- signature: `class AgenticMoshPitConfig(GRPOConfig):`
- members:
  - `advantage_estimator` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L108)
  - `episode_reward_weight` — [`L116`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L116)
  - `group_size` — [`L95`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L95)
  - `max_steps` — [`L91`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L91)
  - `max_tool_calls_per_step` — [`L147`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L147)
  - `num_env_groups` — [`L99`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L99)
  - `reasoning_parser` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L151)
  - `reward_mode` — [`L104`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L104)
  - `step_reward_gamma` — [`L112`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L112)
  - `step_reward_weight` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L120)
  - `system_prompt` — [`L125`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L125)
  - `tool_caller` — [`L137`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L137)
  - `tool_names` — [`L129`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L129)
  - `tool_schemas` — [`L133`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L133)
  - `trainer_prefix` — [`L86`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L86)
- protocol/private: `__hash__`[`L187`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L187), `__post_init__`[`L163`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/agentic_moshpit_config.py#L163)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`GRPOConfig`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`__post_init__`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig.__post_init__), [`num_generations`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig.num_generations), [`num_return_sequences`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig.num_return_sequences)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_preprocess_batch_input`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._preprocess_batch_input), [`GRPOConfig`](../group_relative_policy_optimization/grpo_config.md#GRPOConfig), [`_score_auxiliary_rewards`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._score_auxiliary_rewards), [`arguments`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer.arguments), [`__init__`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer.__init__), [`_tool_call_parser`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._tool_call_parser), [`_tools`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._tools), [`_rollout_manager`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._rollout_manager), [`_wrap_env_with_tools`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._wrap_env_with_tools), [`_apply_auxiliary_rewards_to_trajectories`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._apply_auxiliary_rewards_to_trajectories), [`_strip_thinking`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._strip_thinking)

