---
title: 'Module: easydel/trainers/group_relative_policy_optimization/grpo_config.py'
type: catalog
provenance: extracted
module: easydel/trainers/group_relative_policy_optimization/grpo_config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.group_relative_policy_optimization.grpo_config`/GRPOConfig#
symbols:
  GRPOConfig: ''
  GRPOConfig.__post_init__: __post_init__().
  GRPOConfig.max_prompt_length: max_prompt_length.
  GRPOConfig.logprob_vocab_chunk_size: logprob_vocab_chunk_size.
  GRPOConfig.max_completion_length: max_completion_length.
  GRPOConfig.num_generations: num_generations.
  GRPOConfig.completion_chunk_size: completion_chunk_size.
  GRPOConfig.max_loss_completion_tokens: max_loss_completion_tokens.
  GRPOConfig.reward_weights: reward_weights.
  GRPOConfig.scale_rewards: scale_rewards.
  GRPOConfig.mask_truncated_completions: mask_truncated_completions.
  GRPOConfig.beta: beta.
  GRPOConfig.num_return_sequences: num_return_sequences.
  GRPOConfig.ref_logps_chunk_size: ref_logps_chunk_size.
  GRPOConfig.epsilon_high: epsilon_high.
  GRPOConfig.sync_ref_model: sync_ref_model.
  GRPOConfig.ref_model_sync_steps: ref_model_sync_steps.
  GRPOConfig.loss_type: loss_type.
  GRPOConfig.epsilon: epsilon.
  GRPOConfig.ref_model_mixup_alpha: ref_model_mixup_alpha.
  GRPOConfig.delta: delta.
  GRPOConfig.importance_sampling_level: importance_sampling_level.
  GRPOConfig.skip_apply_chat_template: skip_apply_chat_template.
  GRPOConfig.temperature: temperature.
  GRPOConfig.top_entropy_quantile: top_entropy_quantile.
  GRPOConfig.trainer_prefix: trainer_prefix.
  GRPOConfig.remove_unused_columns: remove_unused_columns.
  GRPOConfig.dataset_num_proc: dataset_num_proc.
  GRPOConfig.learning_rate: learning_rate.
  GRPOConfig.num_iterations: num_iterations.
  GRPOConfig.tools: tools.
  GRPOConfig.top_p: top_p.
  GRPOConfig.top_k: top_k.
  GRPOConfig.presence_penalty: presence_penalty.
  GRPOConfig.frequency_penalty: frequency_penalty.
  GRPOConfig.min_p: min_p.
  GRPOConfig.repetition_penalty: repetition_penalty.
  GRPOConfig.generation_kwargs: generation_kwargs.
  GRPOConfig.chat_template_kwargs: chat_template_kwargs.
  GRPOConfig.__hash__: __hash__.
---
# Module: [`easydel/trainers/group_relative_policy_optimization/grpo_config.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py)

## Classes
### `GRPOConfig`  ·  implements/extends TrainingArguments
- def: [`easydel/trainers/group_relative_policy_optimization/grpo_config.py:25`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L25)
- doc: Configuration class for Group Relative Policy Optimization training.
- signature: `class GRPOConfig(TrainingArguments):`
- members:
  - `__post_init__(self, max_sequence_length: int | None, quantization_block: int | None)` — [`L222`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L222) — Post initialization to set dependent parameters.
  - `beta` — [`L66`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L66)
  - `chat_template_kwargs` — [`L173`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L173)
  - `completion_chunk_size` — [`L194`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L194)
  - `dataset_num_proc` — [`L58`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L58)
  - `delta` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L78)
  - `epsilon` — [`L70`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L70)
  - `epsilon_high` — [`L74`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L74)
  - `frequency_penalty` — [`L157`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L157)
  - `generation_kwargs` — [`L169`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L169)
  - `importance_sampling_level` — [`L104`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L104)
  - `learning_rate` — [`L62`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L62)
  - `logprob_vocab_chunk_size` — [`L212`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L212)
  - `loss_type` — [`L100`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L100)
  - `mask_truncated_completions` — [`L177`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L177)
  - `max_completion_length` — [`L54`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L54)
  - `max_loss_completion_tokens` — [`L203`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L203)
  - `max_prompt_length` — [`L50`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L50)
  - `min_p` — [`L161`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L161)
  - `num_generations` — [`L137`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L137)
  - `num_iterations` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L96)
  - `num_return_sequences` — [`L128`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L128)
  - `presence_penalty` — [`L153`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L153)
  - `ref_logps_chunk_size` — [`L185`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L185)
  - `ref_model_mixup_alpha` — [`L88`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L88)
  - `ref_model_sync_steps` — [`L92`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L92)
  - `remove_unused_columns` — [`L46`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L46)
  - `repetition_penalty` — [`L165`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L165)
  - `reward_weights` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L108)
  - `scale_rewards` — [`L114`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L114)
  - `skip_apply_chat_template` — [`L124`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L124)
  - `sync_ref_model` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L84)
  - `temperature` — [`L141`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L141)
  - `tools` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L120)
  - `top_entropy_quantile` — [`L181`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L181)
  - `top_k` — [`L149`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L149)
  - `top_p` — [`L145`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L145)
  - `trainer_prefix` — [`L42`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L42)
- protocol/private: `__hash__`[`L289`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/grpo_config.py#L289)
- uses (calls/refs, reference-scoped): [`Registry`](../../utils/registery.md#Registry), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`max_length`](../training_configurations.md#TrainingArguments.max_length), [`hash_fn`](../../utils/compiling_utils.md#hash_fn), [`register`](../../utils/registery.md#Registry.register), [`__post_init__`](../training_configurations.md#TrainingArguments.__post_init__), [`RLVRConfig`](../rlvr_trainer/rlvr_config.md#RLVRConfig), [`AgenticMoshPitConfig`](../agentic_moshpit/agentic_moshpit_config.md#AgenticMoshPitConfig), [`GFPOConfig`](../group_filtered_policy_optimization/gfpo_config.md#GFPOConfig), [`GSPOConfig`](../group_sequence_policy_optimization/gspo_config.md#GSPOConfig), [`SDPOConfig`](../self_distillation_policy_optimization/sdpo_config.md#SDPOConfig), [`NashMDConfig`](../nash_md_trainer/nash_md_config.md#NashMDConfig), [`XPOConfig`](../xpo_trainer/xpo_config.md#XPOConfig), [`_handle_deprecated_max_sequence_length`](../training_configurations.md#TrainingArguments._handle_deprecated_max_sequence_length), [`generation_temperature`](../training_configurations.md#TrainingArguments.generation_temperature)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_preprocess_batch_input`](grpo_trainer.md#GRPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](../group_filtered_policy_optimization/gfpo_trainer.md#GFPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](../self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer._preprocess_batch_input), [`_preprocess_batch_input`](../agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer._preprocess_batch_input), [`configure_functions`](grpo_trainer.md#GRPOTrainer.configure_functions), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`configure_functions`](../self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer.configure_functions), [`configure_functions`](../xpo_trainer/xpo_trainer.md#XPOTrainer.configure_functions), [`configure_functions`](../nash_md_trainer/nash_md_trainer.md#NashMDTrainer.configure_functions), [`configure_functions`](../group_filtered_policy_optimization/gfpo_trainer.md#GFPOTrainer.configure_functions), [`arguments`](grpo_trainer.md#GRPOTrainer.arguments), [`__init__`](grpo_trainer.md#GRPOTrainer.__init__), [`__init__`](../rlvr_trainer/rlvr_trainer.md#RLVRTrainer.__init__), [`_configure_teacher_context`](../self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer._configure_teacher_context), [`on_step_end`](grpo_trainer.md#GRPOTrainer.on_step_end), [`on_step_end`](../self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer.on_step_end), [`ref_logps_chunk_size`](grpo_trainer.md#GRPOTrainer.ref_logps_chunk_size), [`RLVRConfig`](../rlvr_trainer/rlvr_config.md#RLVRConfig), [`scale_rewards`](grpo_trainer.md#GRPOTrainer.scale_rewards), [`on_step_end`](../xpo_trainer/xpo_trainer.md#XPOTrainer.on_step_end), [`AgenticMoshPitConfig`](../agentic_moshpit/agentic_moshpit_config.md#AgenticMoshPitConfig), [`GFPOConfig`](../group_filtered_policy_optimization/gfpo_config.md#GFPOConfig), [`GSPOConfig`](../group_sequence_policy_optimization/gspo_config.md#GSPOConfig), [`SDPOConfig`](../self_distillation_policy_optimization/sdpo_config.md#SDPOConfig), [`NashMDConfig`](../nash_md_trainer/nash_md_config.md#NashMDConfig), [`XPOConfig`](../xpo_trainer/xpo_config.md#XPOConfig), [`_compute_refmodel_logps`](grpo_trainer.md#GRPOTrainer._compute_refmodel_logps), [`num_generations`](grpo_trainer.md#GRPOTrainer.num_generations), [`__post_init__`](../agentic_moshpit/agentic_moshpit_config.md#AgenticMoshPitConfig.__post_init__), [`__post_init__`](../self_distillation_policy_optimization/sdpo_config.md#SDPOConfig.__post_init__), [`_get_preprocess_transform`](grpo_trainer.md#GRPOTrainer._get_preprocess_transform), [`_compute_model_logps`](../nash_md_trainer/nash_md_trainer.md#NashMDTrainer._compute_model_logps), [`_compute_refmodel_logps`](../self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer._compute_refmodel_logps), [`__post_init__`](../rlvr_trainer/rlvr_config.md#RLVRConfig.__post_init__), [`create_grain_collect_function`](grpo_trainer.md#GRPOTrainer.create_grain_collect_function), [`create_tfds_collect_function`](grpo_trainer.md#GRPOTrainer.create_tfds_collect_function), [`importance_sampling_level`](grpo_trainer.md#GRPOTrainer.importance_sampling_level), [`teacher_prompt_length`](../self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer.teacher_prompt_length), [`__post_init__`](../xpo_trainer/xpo_config.md#XPOConfig.__post_init__)  (+10 more)

