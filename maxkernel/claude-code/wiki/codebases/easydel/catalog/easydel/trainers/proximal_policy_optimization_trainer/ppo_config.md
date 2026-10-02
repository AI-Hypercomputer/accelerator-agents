---
title: 'Module: easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.proximal_policy_optimization_trainer.ppo_config`/PPOConfig#
symbols:
  PPOConfig.__post_init__: __post_init__().
  PPOConfig: ''
  PPOConfig.logprob_vocab_chunk_size: logprob_vocab_chunk_size.
  PPOConfig.max_prompt_length: max_prompt_length.
  PPOConfig.entropy_coef: entropy_coef.
  PPOConfig.reward_weights: reward_weights.
  PPOConfig.num_generations: num_generations.
  PPOConfig.max_completion_length: max_completion_length.
  PPOConfig.cliprange: cliprange.
  PPOConfig.vf_coef: vf_coef.
  PPOConfig.cliprange_value: cliprange_value.
  PPOConfig.missing_eos_penalty: missing_eos_penalty.
  PPOConfig.num_return_sequences: num_return_sequences.
  PPOConfig.kl_coef: kl_coef.
  PPOConfig.kl_estimator: kl_estimator.
  PPOConfig.gamma: gamma.
  PPOConfig.lam: lam.
  PPOConfig.whiten_rewards: whiten_rewards.
  PPOConfig.whiten_advantages: whiten_advantages.
  PPOConfig.skip_apply_chat_template: skip_apply_chat_template.
  PPOConfig.temperature: temperature.
  PPOConfig.mask_truncated_completions: mask_truncated_completions.
  PPOConfig.trainer_prefix: trainer_prefix.
  PPOConfig.remove_unused_columns: remove_unused_columns.
  PPOConfig.dataset_num_proc: dataset_num_proc.
  PPOConfig.learning_rate: learning_rate.
  PPOConfig.num_ppo_epochs: num_ppo_epochs.
  PPOConfig.tools: tools.
  PPOConfig.top_p: top_p.
  PPOConfig.top_k: top_k.
  PPOConfig.presence_penalty: presence_penalty.
  PPOConfig.frequency_penalty: frequency_penalty.
  PPOConfig.min_p: min_p.
  PPOConfig.repetition_penalty: repetition_penalty.
  PPOConfig.generation_kwargs: generation_kwargs.
  PPOConfig.chat_template_kwargs: chat_template_kwargs.
  PPOConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py)

## Classes
### `PPOConfig`  ·  implements/extends TrainingArguments
- def: [`easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py:38`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L38)
- doc: Configuration class for Proximal Policy Optimization (PPO) training.
- signature: `class PPOConfig(TrainingArguments):`
- members:
  - `chat_template_kwargs` — [`L176`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L176)
  - `cliprange` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L84)
  - `cliprange_value` — [`L92`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L92)
  - `dataset_num_proc` — [`L64`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L64)
  - `entropy_coef` — [`L112`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L112)
  - `frequency_penalty` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L160)
  - `gamma` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L96)
  - `generation_kwargs` — [`L172`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L172)
  - `kl_coef` — [`L76`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L76)
  - `kl_estimator` — [`L80`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L80)
  - `lam` — [`L100`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L100)
  - `learning_rate` — [`L68`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L68)
  - `logprob_vocab_chunk_size` — [`L184`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L184)
  - `mask_truncated_completions` — [`L180`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L180)
  - `max_completion_length` — [`L60`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L60)
  - `max_prompt_length` — [`L56`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L56)
  - `min_p` — [`L164`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L164)
  - `missing_eos_penalty` — [`L118`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L118)
  - `num_generations` — [`L140`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L140)
  - `num_ppo_epochs` — [`L72`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L72)
  - `num_return_sequences` — [`L136`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L136)
  - `presence_penalty` — [`L156`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L156)
  - `remove_unused_columns` — [`L52`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L52)
  - `repetition_penalty` — [`L168`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L168)
  - `reward_weights` — [`L126`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L126)
  - `skip_apply_chat_template` — [`L132`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L132)
  - `temperature` — [`L144`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L144)
  - `tools` — [`L122`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L122)
  - `top_k` — [`L152`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L152)
  - `top_p` — [`L148`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L148)
  - `trainer_prefix` — [`L48`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L48)
  - `vf_coef` — [`L88`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L88)
  - `whiten_advantages` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L108)
  - `whiten_rewards` — [`L104`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L104)
- protocol/private: `__hash__`[`L229`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L229), `__post_init__`[`L194`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/ppo_config.py#L194)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`__post_init__`](../training_configurations.md#TrainingArguments.__post_init__), [`_handle_deprecated_max_sequence_length`](../training_configurations.md#TrainingArguments._handle_deprecated_max_sequence_length), [`generation_temperature`](../training_configurations.md#TrainingArguments.generation_temperature)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_preprocess_batch_input`](ppo_trainer.md#PPOTrainer._preprocess_batch_input), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`configure_functions`](ppo_trainer.md#PPOTrainer.configure_functions), [`arguments`](ppo_trainer.md#PPOTrainer.arguments), [`__init__`](ppo_trainer.md#PPOTrainer.__init__), [`_compute_refmodel_logps`](ppo_trainer.md#PPOTrainer._compute_refmodel_logps), [`_compute_rollout_logps_values`](ppo_trainer.md#PPOTrainer._compute_rollout_logps_values), [`_get_preprocess_transform`](ppo_trainer.md#PPOTrainer._get_preprocess_transform), [`_compute_gae`](ppo_trainer.md#PPOTrainer._compute_gae), [`create_grain_collect_function`](ppo_trainer.md#PPOTrainer.create_grain_collect_function), [`create_tfds_collect_function`](ppo_trainer.md#PPOTrainer.create_tfds_collect_function), [`reward_weights`](ppo_trainer.md#PPOTrainer.reward_weights), [`num_generations`](ppo_trainer.md#PPOTrainer.num_generations)

