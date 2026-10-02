---
title: 'Module: easydel/infra/loss_utils.py'
type: catalog
provenance: extracted
module: easydel/infra/loss_utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.loss_utils`/
symbols:
  LossMetrics: LossMetrics#
  LossConfig: LossConfig#
  fixed_cross_entropy: fixed_cross_entropy().
  auxiliary_load_balancing_loss_func: auxiliary_load_balancing_loss_func().
  causal_lm_loss_chunked_lm_head: causal_lm_loss_chunked_lm_head().
  _should_chunk_causal_lm_loss: _should_chunk_causal_lm_loss().
  ForSequenceClassificationLoss: ForSequenceClassificationLoss().
  get_factor_and_weight: get_factor_and_weight().
  ForQuestionAnsweringLoss: ForQuestionAnsweringLoss().
  CausalLMLossStrategy.compute: CausalLMLossStrategy#compute().
  LossMetrics.loss: LossMetrics#loss.
  SLNF: SLNF.
  causal_lm_loss_chunked_lm_head._chunk_loss: causal_lm_loss_chunked_lm_head()._chunk_loss().
  ForCausalLMLoss: ForCausalLMLoss().
  LOSS_MAPPING: LOSS_MAPPING.
  convert_special_loss_normalizing_factor_to_enum: convert_special_loss_normalizing_factor_to_enum().
  resolve_loss_strategy: resolve_loss_strategy().
  BaseLossStrategy.compute: BaseLossStrategy#compute().
  FunctionalLossStrategy.compute: FunctionalLossStrategy#compute().
  CausalLMLossStrategy.plan_forward: CausalLMLossStrategy#plan_forward().
  resolve_causal_lm_chunk_token_size: resolve_causal_lm_chunk_token_size().
  BaseLossStrategy: BaseLossStrategy#
  BaseLossStrategy.plan_forward: BaseLossStrategy#plan_forward().
  cross_entropy_blockwise_logits: cross_entropy_blockwise_logits().
  ForTokenClassification: ForTokenClassification().
  LossForwardPlan: LossForwardPlan#
  LossConfig.loss_normalizing_factor: LossConfig#loss_normalizing_factor.
  _apply_sparse_label_smoothing: _apply_sparse_label_smoothing().
  LossConfig.ignore_index: LossConfig#ignore_index.
  LossMetrics.accuracy: LossMetrics#accuracy.
  LossConfig.label_smoothing: LossConfig#label_smoothing.
  LossConfig.compute_dtype: LossConfig#compute_dtype.
  LossMetrics.other_metrics: LossMetrics#other_metrics.
  _logsumexp_chunked: _logsumexp_chunked().
  sparse_cross_entropy_chunked_vocab: sparse_cross_entropy_chunked_vocab().
  sparse_cross_entropy_chunked_tokens: sparse_cross_entropy_chunked_tokens().
  compute_weighted_cross_entropy: compute_weighted_cross_entropy().
  dynamic_cross_entropy_loss: dynamic_cross_entropy_loss().
  LossConfig.shift_tokens: LossConfig#shift_tokens.
  LossConfig.chunk_token_size: LossConfig#chunk_token_size.
  LossMetrics.weight_sum: LossMetrics#weight_sum.
  FACTOR_TYPE: FACTOR_TYPE.
  _sum_weights_per_segment._repeat_last_nonnegative: _sum_weights_per_segment()._repeat_last_nonnegative().
  LossForwardPlan.forward_kwargs: LossForwardPlan#forward_kwargs.
  LossConfig.z_loss: LossConfig#z_loss.
  LossConfig.reduction: LossConfig#reduction.
  LossMetrics.z_loss: LossMetrics#z_loss.
  cross_entropy_blockwise_logits.process_block: cross_entropy_blockwise_logits().process_block().
  FunctionalLossStrategy: FunctionalLossStrategy#
  cross_entropy_blockwise_logits.full_body: cross_entropy_blockwise_logits().full_body().
  sparse_cross_entropy_chunked_tokens.body: sparse_cross_entropy_chunked_tokens().body().
  compute_weighted_cross_entropy_and_accuracy: compute_weighted_cross_entropy_and_accuracy().
  _sum_weights_per_segment: _sum_weights_per_segment().
  causal_lm_loss_chunked_lm_head._accumulate_chunk: causal_lm_loss_chunked_lm_head()._accumulate_chunk().
  CausalLMLossStrategy: CausalLMLossStrategy#
  SpecialLossNormalizingFactor.NO_WEIGHT_NUM_REAL_TARGET_TOKENS: SpecialLossNormalizingFactor#NO_WEIGHT_NUM_REAL_TARGET_TOKENS.
  SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS: SpecialLossNormalizingFactor#NUM_REAL_TARGET_TOKENS.
  LossConfig.chunk_vocab_size: LossConfig#chunk_vocab_size.
  LossConfig.chunk_block_size: LossConfig#chunk_block_size.
  LossMetrics.max_grad_norm: LossMetrics#max_grad_norm.
  LossMetrics.mean_grad_norm: LossMetrics#mean_grad_norm.
  LossMetrics.grad_norms: LossMetrics#grad_norms.
  LossMetrics.execution_time: LossMetrics#execution_time.
  FunctionalLossStrategy.__init__: FunctionalLossStrategy#__init__().
  LossConfig.__str__: LossConfig#__str__.
  LossConfig.__hash__: LossConfig#__hash__.
  _resolve_module_vocab_size: _resolve_module_vocab_size().
  SpecialLossNormalizingFactor.NUM_TOTAL_TARGET_TOKENS: SpecialLossNormalizingFactor#NUM_TOTAL_TARGET_TOKENS.
  SpecialLossNormalizingFactor.AVERAGE_PER_SEQUENCE: SpecialLossNormalizingFactor#AVERAGE_PER_SEQUENCE.
  LossConfig.num_labels: LossConfig#num_labels.
  LossConfig.divide_weight_sum: LossConfig#divide_weight_sum.
  LossConfig.break_on_nan: LossConfig#break_on_nan.
  LossConfig.classification_problem_type: LossConfig#classification_problem_type.
  LossMetrics.chosen_rewards: LossMetrics#chosen_rewards.
  LossMetrics.rejected_rewards: LossMetrics#rejected_rewards.
  cross_entropy_with_logits: cross_entropy_with_logits().
  FunctionalLossStrategy.loss_fn: FunctionalLossStrategy#loss_fn.
  SpecialLossNormalizingFactor: SpecialLossNormalizingFactor#
  LossConfig.problem_type: LossConfig#problem_type.
  LossConfig.num_classification_labels: LossConfig#num_classification_labels.
  LossConfig.__repr__: LossConfig#__repr__().
  LossMetrics.learning_rate: LossMetrics#learning_rate.
  _logsumexp_chunked.max_body: _logsumexp_chunked().max_body().
  _logsumexp_chunked.sum_body: _logsumexp_chunked().sum_body().
  _label_smoothing_params: _label_smoothing_params().
  sigmoid_cross_entropy_with_logits: sigmoid_cross_entropy_with_logits().
  onehot: onehot().
  _cross_entropy_with_logits_fwd: _cross_entropy_with_logits_fwd().
  _cross_entropy_with_logits_bwd: _cross_entropy_with_logits_bwd().
  cross_entropy_loss_and_accuracy: cross_entropy_loss_and_accuracy().
  _sum_weights_per_segment._repeat_last_nonnegative.fn: _sum_weights_per_segment()._repeat_last_nonnegative().fn().
  _supports_chunked_causal_lm_forward: _supports_chunked_causal_lm_forward().
  BaseLossStrategy.__name__: BaseLossStrategy#__name__.
  CausalLMLossStrategy.__name__: CausalLMLossStrategy#__name__.
---
# Module: [`easydel/infra/loss_utils.py`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py)

## Classes
### `BaseLossStrategy`
- def: [`easydel/infra/loss_utils.py:161`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L161)
- doc: Two-stage loss interface: plan the forward pass, then consume outputs.
- signature: `class BaseLossStrategy:`
- members:
  - `compute(self, *, module: tp.Any, outputs: tp.Any, labels: jax.Array | None, loss_config: LossConfig | None, batch: collections.abc.Mapping[str, Array], loss_kwargs: dict[str, tp.Any], paxis: PartitionAxis | None, forward_plan: LossForwardPlan)` — [`L206`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L206) — Compute the loss from model outputs.
  - `plan_forward(self, *, module: tp.Any, labels: jax.Array | None, loss_config: LossConfig | None, batch: collections.abc.Mapping[str, Array], loss_kwargs: dict[str, tp.Any])` — [`L181`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L181) — Return forward-pass overrides for the upcoming model call.
- protocol/private: `__name__`[`L179`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L179)
- uses (calls/refs, reference-scoped): [`LossMetrics`](loss_utils.md#LossMetrics), [`LossConfig`](loss_utils.md#LossConfig), [`compute`](loss_utils.md#CausalLMLossStrategy.compute), [`compute`](loss_utils.md#FunctionalLossStrategy.compute), [`plan_forward`](loss_utils.md#CausalLMLossStrategy.plan_forward), [`LossForwardPlan`](loss_utils.md#LossForwardPlan), [`CausalLMLossStrategy`](loss_utils.md#CausalLMLossStrategy), [`FunctionalLossStrategy`](loss_utils.md#FunctionalLossStrategy)
- used by: [`compute_loss`](base_module.md#EasyDeLBaseModule.compute_loss), [`resolve_loss_strategy`](loss_utils.md#resolve_loss_strategy), [`CausalLMLossStrategy`](loss_utils.md#CausalLMLossStrategy), [`FunctionalLossStrategy`](loss_utils.md#FunctionalLossStrategy)

### `CausalLMLossStrategy`  ·  implements/extends BaseLossStrategy
- def: [`easydel/infra/loss_utils.py:2451`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2451)
- doc: Planning-aware loss strategy for causal language modelling.
- signature: `class CausalLMLossStrategy(BaseLossStrategy):`
- members:
  - `compute(self, *, module: tp.Any, outputs: tp.Any, labels: jax.Array | None, loss_config: LossConfig | None, batch: collections.abc.Mapping[str, Array], loss_kwargs: dict[str, tp.Any], paxis: PartitionAxis | None, forward_plan: LossForwardPlan)` — [`L2503`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2503) — Compute causal-LM loss, dispatching to dense or chunked path. — documented in [easydel-infra-loss_utils](../../../concepts/easydel-infra-loss_utils.md)
  - `plan_forward(self, *, module: tp.Any, labels: jax.Array | None, loss_config: LossConfig | None, batch: collections.abc.Mapping[str, Array], loss_kwargs: dict[str, tp.Any])` — [`L2474`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2474) — Decide whether to request the chunked (headless) forward path.
- protocol/private: `__name__`[`L2472`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2472)
- uses (calls/refs, reference-scoped): [`LossMetrics`](loss_utils.md#LossMetrics), [`LossConfig`](loss_utils.md#LossConfig), [`_should_chunk_causal_lm_loss`](loss_utils.md#_should_chunk_causal_lm_loss), [`causal_lm_loss_chunked_lm_head`](loss_utils.md#causal_lm_loss_chunked_lm_head), [`ForCausalLMLoss`](loss_utils.md#ForCausalLMLoss), [`BaseLossStrategy`](loss_utils.md#BaseLossStrategy), [`LossForwardPlan`](loss_utils.md#LossForwardPlan), [`forward_kwargs`](loss_utils.md#LossForwardPlan.forward_kwargs), [`_resolve_module_vocab_size`](loss_utils.md#_resolve_module_vocab_size)
- used by: [`resolve_loss_strategy`](loss_utils.md#resolve_loss_strategy), [`compute`](loss_utils.md#BaseLossStrategy.compute), [`BaseLossStrategy`](loss_utils.md#BaseLossStrategy), [`plan_forward`](loss_utils.md#BaseLossStrategy.plan_forward)

### `FunctionalLossStrategy`  ·  implements/extends BaseLossStrategy
- def: [`easydel/infra/loss_utils.py:242`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L242)
- doc: Compatibility adapter that wraps a plain loss callable as a strategy.
- signature: `class FunctionalLossStrategy(BaseLossStrategy):`
- members:
  - `__init__(self, loss_fn: tp.Callable[..., LossMetrics])` — [`L261`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L261) — Wrap a plain loss callable as a ``BaseLossStrategy``.
  - `compute(self, *, module: tp.Any, outputs: tp.Any, labels: jax.Array | None, loss_config: LossConfig | None, batch: collections.abc.Mapping[str, Array], loss_kwargs: dict[str, tp.Any], paxis: PartitionAxis | None, forward_plan: LossForwardPlan)` — [`L272`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L272) — Delegate to the wrapped loss callable, ignoring the forward plan.
  - `loss_fn` — [`L269`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L269)
- uses (calls/refs, reference-scoped): [`LossMetrics`](loss_utils.md#LossMetrics), [`LossConfig`](loss_utils.md#LossConfig), [`BaseLossStrategy`](loss_utils.md#BaseLossStrategy), [`LossForwardPlan`](loss_utils.md#LossForwardPlan)
- used by: [`resolve_loss_strategy`](loss_utils.md#resolve_loss_strategy), [`compute`](loss_utils.md#BaseLossStrategy.compute), [`BaseLossStrategy`](loss_utils.md#BaseLossStrategy)

### `LossConfig`
- def: [`easydel/infra/loss_utils.py:346`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L346) — documented in [easydel-infra-loss_utils](../../../concepts/easydel-infra-loss_utils.md)
- members:
  - `__repr__(self)` — [`L424`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L424) — Return a detailed string representation of the configuration.
  - `break_on_nan` — [`L413`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L413)
  - `chunk_block_size` — [`L421`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L421)
  - `chunk_token_size` — [`L420`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L420)
  - `chunk_vocab_size` — [`L419`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L419)
  - `classification_problem_type` — [`L416`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L416)
  - `compute_dtype` — [`L422`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L422)
  - `divide_weight_sum` — [`L411`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L411)
  - `ignore_index` — [`L405`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L405)
  - `label_smoothing` — [`L406`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L406)
  - `loss_normalizing_factor` — [`L408`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L408)
  - `num_classification_labels` — [`L415`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L415)
  - `num_labels` — [`L409`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L409)
  - `problem_type` — [`L410`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L410)
  - `reduction` — [`L414`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L414)
  - `shift_tokens` — [`L412`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L412)
  - `z_loss` — [`L407`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L407)
- protocol/private: `__hash__`[`L435`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L435), `__str__`[`L434`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L434)
- uses (calls/refs, reference-scoped): [`hash_fn`](../utils/compiling_utils.md#hash_fn), [`FACTOR_TYPE`](loss_utils.md#FACTOR_TYPE)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`_ensure_variables`](../trainers/training_configurations.md#TrainingArguments._ensure_variables), [`__all__`](../modules/whisper/modeling_whisper.md#__all__), [`fixed_cross_entropy`](loss_utils.md#fixed_cross_entropy), [`update_state_respectfully`](../trainers/training_utils.md#update_state_respectfully), [`loss_config`](../trainers/training_configurations.md#TrainingArguments.loss_config), [`compute_loss`](base_module.md#EasyDeLBaseModule.compute_loss), [`training_step`](../trainers/direct_preference_optimization_trainer/_fn.md#training_step), [`xpo_step`](../trainers/xpo_trainer/_fn.md#xpo_step), [`training_step`](../trainers/kto_trainer/_fn.md#training_step), [`distillation_step`](../trainers/distillation_trainer/_fn.md#distillation_step), [`gkd_step`](../trainers/generalized_knowledge_distillation_trainer/_fn.md#gkd_step), [`grpo_step`](../trainers/group_relative_policy_optimization/_fn.md#grpo_step), [`nash_md_step`](../trainers/nash_md_trainer/_fn.md#nash_md_step), [`on_policy_distillation_step`](../trainers/on_policy_distillation_trainer/_fn.md#on_policy_distillation_step), [`orpo_step`](../trainers/odds_ratio_preference_optimization_trainer/_fn.md#orpo_step), [`ppo_step`](../trainers/proximal_policy_optimization_trainer/_fn.md#ppo_step), [`sdpo_step`](../trainers/self_distillation_policy_optimization/_fn.md#sdpo_step), [`sparse_distillation_step`](../trainers/sparse_distillation_trainer/_fn.md#sparse_distillation_step), [`training_step`](../trainers/contrastive_preference_optimization_trainer/_fn.md#training_step), [`_should_chunk_causal_lm_loss`](loss_utils.md#_should_chunk_causal_lm_loss), [`apply_training_hooks`](../trainers/base_trainer.md#BaseTrainer.apply_training_hooks), [`causal_lm_loss_chunked_lm_head`](loss_utils.md#causal_lm_loss_chunked_lm_head), [`ForSequenceClassificationLoss`](loss_utils.md#ForSequenceClassificationLoss), [`training_step`](../trainers/binary_classifier_optimization_trainer/_fn.md#training_step), [`training_step`](../trainers/reward_trainer/_fn.md#training_step), [`training_step`](../trainers/trainer/_fn.md#training_step), [`compute_loss`](../modules/whisper/modeling_whisper.md#WhisperForConditionalGeneration.compute_loss), [`evaluation_step`](../trainers/trainer/_fn.md#evaluation_step), [`ForQuestionAnsweringLoss`](loss_utils.md#ForQuestionAnsweringLoss), [`__getattr__`](__init__.md#__getattr__), [`compute`](loss_utils.md#CausalLMLossStrategy.compute), [`evaluation_step`](../trainers/reward_trainer/_fn.md#evaluation_step), [`normalize_trainer_config`](elarge/types/training.md#normalize_trainer_config), [`ForCausalLMLoss`](loss_utils.md#ForCausalLMLoss), [`compute`](loss_utils.md#BaseLossStrategy.compute), [`compute`](loss_utils.md#FunctionalLossStrategy.compute), [`compute_loss`](mixins/protocol.md#BaseModuleProtocol.compute_loss), [`plan_forward`](loss_utils.md#CausalLMLossStrategy.plan_forward), [`resolve_causal_lm_chunk_token_size`](loss_utils.md#resolve_causal_lm_chunk_token_size)  (+3 more)

### `LossForwardPlan`
- def: [`easydel/infra/loss_utils.py:96`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L96)
- doc: Forward-pass requirements declared by a loss strategy.
- signature: `class LossForwardPlan:`
- members:
  - `forward_kwargs` — [`L115`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L115)
- used by: [`compute_loss`](base_module.md#EasyDeLBaseModule.compute_loss), [`compute`](loss_utils.md#CausalLMLossStrategy.compute), [`compute`](loss_utils.md#BaseLossStrategy.compute), [`compute`](loss_utils.md#FunctionalLossStrategy.compute), [`plan_forward`](loss_utils.md#CausalLMLossStrategy.plan_forward), [`plan_forward`](loss_utils.md#BaseLossStrategy.plan_forward)

### `LossMetrics`
- def: [`easydel/infra/loss_utils.py:439`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L439) — documented in [easydel-infra-loss_utils](../../../concepts/easydel-infra-loss_utils.md)
- members:
  - `accuracy` — [`L482`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L482)
  - `chosen_rewards` — [`L487`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L487)
  - `execution_time` — [`L490`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L490)
  - `grad_norms` — [`L486`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L486)
  - `learning_rate` — [`L483`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L483)
  - `loss` — [`L479`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L479)
  - `max_grad_norm` — [`L484`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L484)
  - `mean_grad_norm` — [`L485`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L485)
  - `other_metrics` — [`L489`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L489)
  - `rejected_rewards` — [`L488`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L488)
  - `weight_sum` — [`L481`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L481)
  - `z_loss` — [`L480`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L480)
- used by: [`_train_epoch`](../trainers/trainer/trainer.md#Trainer._train_epoch), [`__all__`](../modules/whisper/modeling_whisper.md#__all__), [`_eval_epoch`](../trainers/trainer/trainer.md#Trainer._eval_epoch), [`fixed_cross_entropy`](loss_utils.md#fixed_cross_entropy), [`update_state_respectfully`](../trainers/training_utils.md#update_state_respectfully), [`minibatch_call`](../trainers/training_utils.md#minibatch_call), [`update_metrics`](../trainers/training_utils.md#update_metrics), [`compute_loss`](base_module.md#EasyDeLBaseModule.compute_loss), [`loss_fn`](../trainers/distillation_trainer/_fn.md#distillation_step.loss_fn), [`training_step`](../trainers/direct_preference_optimization_trainer/_fn.md#training_step), [`xpo_step`](../trainers/xpo_trainer/_fn.md#xpo_step), [`calculate`](../trainers/metrics.md#StepMetrics.calculate), [`_execute_train_step`](../trainers/trainer/trainer.md#Trainer._execute_train_step), [`embedding_training_step`](../trainers/embedding_trainer/_fn.md#embedding_training_step), [`training_step`](../trainers/kto_trainer/_fn.md#training_step), [`distillation_step`](../trainers/distillation_trainer/_fn.md#distillation_step), [`gkd_step`](../trainers/generalized_knowledge_distillation_trainer/_fn.md#gkd_step), [`grpo_step`](../trainers/group_relative_policy_optimization/_fn.md#grpo_step), [`nash_md_step`](../trainers/nash_md_trainer/_fn.md#nash_md_step), [`on_policy_distillation_step`](../trainers/on_policy_distillation_trainer/_fn.md#on_policy_distillation_step), [`orpo_step`](../trainers/odds_ratio_preference_optimization_trainer/_fn.md#orpo_step), [`ppo_step`](../trainers/proximal_policy_optimization_trainer/_fn.md#ppo_step), [`sdpo_step`](../trainers/self_distillation_policy_optimization/_fn.md#sdpo_step), [`sparse_distillation_step`](../trainers/sparse_distillation_trainer/_fn.md#sparse_distillation_step), [`loss_fn`](../trainers/on_policy_distillation_trainer/_fn.md#on_policy_distillation_step.loss_fn), [`training_step`](../trainers/contrastive_preference_optimization_trainer/_fn.md#training_step), [`apply_training_hooks`](../trainers/base_trainer.md#BaseTrainer.apply_training_hooks), [`causal_lm_loss_chunked_lm_head`](loss_utils.md#causal_lm_loss_chunked_lm_head), [`ForSequenceClassificationLoss`](loss_utils.md#ForSequenceClassificationLoss), [`loss_fn`](../trainers/group_relative_policy_optimization/_fn.md#grpo_step.loss_fn), [`training_step`](../trainers/binary_classifier_optimization_trainer/_fn.md#training_step), [`training_step`](../trainers/reward_trainer/_fn.md#training_step), [`training_step`](../trainers/trainer/_fn.md#training_step), [`compute_loss`](../modules/whisper/modeling_whisper.md#WhisperForConditionalGeneration.compute_loss), [`evaluation_step`](../trainers/direct_preference_optimization_trainer/_fn.md#evaluation_step), [`evaluation_step`](../trainers/trainer/_fn.md#evaluation_step), [`ForQuestionAnsweringLoss`](loss_utils.md#ForQuestionAnsweringLoss), [`compute`](loss_utils.md#CausalLMLossStrategy.compute), [`evaluation_step`](../trainers/contrastive_preference_optimization_trainer/_fn.md#evaluation_step), [`evaluation_step`](../trainers/kto_trainer/_fn.md#evaluation_step)  (+32 more)

### `SpecialLossNormalizingFactor`  ·  implements/extends Enum
- def: [`easydel/infra/loss_utils.py:302`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L302)
- doc: Enumeration for dynamic loss normalization strategies.
- signature: `class SpecialLossNormalizingFactor(enum.Enum):`
- members:
  - `AVERAGE_PER_SEQUENCE` — [`L336`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L336)
  - `NO_WEIGHT_NUM_REAL_TARGET_TOKENS` — [`L333`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L333)
  - `NUM_REAL_TARGET_TOKENS` — [`L334`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L334)
  - `NUM_TOTAL_TARGET_TOKENS` — [`L335`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L335)
- used by: [`fixed_cross_entropy`](loss_utils.md#fixed_cross_entropy), [`get_factor_and_weight`](loss_utils.md#get_factor_and_weight), [`SLNF`](loss_utils.md#SLNF), [`convert_special_loss_normalizing_factor_to_enum`](loss_utils.md#convert_special_loss_normalizing_factor_to_enum)

## Functions
- `ForCausalLMLoss(logits: jax.Array | None, labels: jax.Array | None, attention_mask: jax.Array | None = None, config: LossConfig | None = None, paxis: PartitionAxis | None = None, num_items_in_batch: int | None = None, batch: collections.abc.Mapping[str, Array] | None = None, **kwargs: tp.Any)` — [`L2605`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2605) — Compute loss for causal language modeling (next-token prediction). — documented in [easydel-infra-loss_utils](../../../concepts/easydel-infra-loss_utils.md)
- `ForQuestionAnsweringLoss(start_logits: jax.Array | None, end_logits: jax.Array | None, start_positions: jax.Array | None, end_positions: jax.Array | None, config: LossConfig | None = None, paxis: PartitionAxis | None = None, batch: collections.abc.Mapping[str, Array] | None = None, **kwargs: tp.Any)` — [`L2812`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2812) — Compute loss for extractive question answering (SQuAD-style). — documented in [easydel-infra-loss_utils](../../../concepts/easydel-infra-loss_utils.md)
- `ForSequenceClassificationLoss(logits: jax.Array | None, labels: jax.Array | None, attention_mask: jax.Array | None = None, config: LossConfig | None = None, paxis: PartitionAxis | None = None, batch: collections.abc.Mapping[str, Array] | None = None, **kwargs: tp.Any)` — [`L2702`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2702) — Compute loss for sequence classification tasks. — documented in [easydel-infra-loss_utils](../../../concepts/easydel-infra-loss_utils.md)
- `ForTokenClassification(logits: jax.Array | None, labels: jax.Array | None, config: LossConfig | None = None, paxis: PartitionAxis | None = None, batch: collections.abc.Mapping[str, Array] | None = None, **kwargs: tp.Any)` — [`L2885`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2885) — Compute loss for token classification tasks.
- `_accumulate_chunk(i: int, carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array])` — [`L2315`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2315) — Accumulate loss metrics from the i-th chunk into running totals.
- `_apply_sparse_label_smoothing(log_z: jax.Array, target_logit: jax.Array, sum_logits: jax.Array, *, vocab_size: int, label_smoothing: float, dtype: jnp.dtype)` — [`L598`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L598) — Compute label-smoothed cross-entropy in the sparse (index-based) setting.
- `_chunk_loss(chunk_hidden_states, chunk_labels, chunk_attention_mask, chunk_loss_weights)` — [`L2276`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2276) — Compute cross-entropy loss for a single chunk of the sequence. — documented in [easydel-infra-loss_utils](../../../concepts/easydel-infra-loss_utils.md)
- `_cross_entropy_with_logits_bwd(res: tuple[Array, float, Array, Array, Array, Array,], g: tuple[Array, Array])` — [`L1329`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1329) — Backward pass for cross_entropy_with_logits custom VJP.
- `_cross_entropy_with_logits_fwd(logits: Array, targets: Array, z_loss: float = 0)` — [`L1277`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1277) — Forward pass for cross_entropy_with_logits custom VJP.
- `_label_smoothing_params(vocab_size: int, label_smoothing: float, dtype: jnp.dtype)` — [`L560`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L560) — Compute label-smoothing coefficients equivalent to the dense one-hot formulation.
- `_logsumexp_chunked(x: jnp.ndarray, chunk_size: int)` — [`L493`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L493) — Compute logsumexp over the last dimension in chunks for memory efficiency.
- `_repeat_last_nonnegative(xs, reverse=False)` — [`L1645`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1645) — Propagate the last non-zero value through zeros in the array.
- `_resolve_module_vocab_size(module: tp.Any)` — [`L118`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L118) — Resolve vocab size for text and multimodal configs.
- `_should_chunk_causal_lm_loss(*, module: tp.Any, labels: jax.Array | None, loss_config: LossConfig | None, token_chunk_size: int | None = None)` — [`L2367`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2367) — Decide whether the chunked LM-head loss path should be activated. — documented in [easydel-infra-loss_utils](../../../concepts/easydel-infra-loss_utils.md)
- `_sum_weights_per_segment(positions: Array, segment_ids: Array, weights: Array)` — [`L1611`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1611) — Sum weights per segment for packed sequence normalization.
- `_supports_chunked_causal_lm_forward(module: tp.Any)` — [`L2425`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2425) — Check whether *module* exposes the API needed for chunked LM-head loss.
- `auxiliary_load_balancing_loss_func(gate_logits: Array | tuple[Array, ...] | None, num_experts: int | None, top_k: int, attention_mask: Array | None = None, compute_dtype: jnp.dtype = jnp.float32)` — [`L1760`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1760) — Compute auxiliary load balancing loss for Mixture-of-Experts models. — documented in [easydel-infra-loss_utils](../../../concepts/easydel-infra-loss_utils.md)
- `body(i, carry)` — [`L969`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L969) — Process a chunk of tokens and accumulate results.
- `causal_lm_loss_chunked_lm_head(hidden_states: jax.Array | None, labels: jax.Array | None, lm_head_fn: tp.Callable[[jax.Array], jax.Array], *, vocab_size: int, attention_mask: jax.Array | None = None, config: LossConfig | None = None, num_items_in_batch: int | None = None, batch: collections.abc.Mapping[str, Array] | None = None, logit_cap_fn: tp.Callable[[jax.Array], jax.Array] | None = None, token_chunk_size: int | None = None, **kwargs: tp.Any)` — [`L2120`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2120) — Compute causal-LM cross-entropy by projecting hidden states through the — documented in [easydel-infra-loss_utils](../../../concepts/easydel-infra-loss_utils.md)
- `compute_weighted_cross_entropy(logits: Array, targets: Array, weights: Array | None = None, label_smoothing: float = 0, z_loss: float = 0, loss_normalizing_factor: float | None = None, compute_dtype: jnp.dtype = jnp.float32)` — [`L1378`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1378) — Compute weighted cross-entropy loss with label smoothing and z-loss.
- `compute_weighted_cross_entropy_and_accuracy(logits: Array, targets: Array, weights: Array | None = None, label_smoothing: float = 0, z_loss: float = 0, loss_normalizing_factor: float | None = None, compute_dtype: jnp.dtype = jnp.float32)` — [`L1467`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1467) — Compute weighted cross-entropy loss, z-loss, weight sum, and accuracy.
- `convert_special_loss_normalizing_factor_to_enum(x: str)` — [`L1575`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1575) — Convert a string to SpecialLossNormalizingFactor enum. — documented in [easydel-infra-loss_utils](../../../concepts/easydel-infra-loss_utils.md)
- `cross_entropy_blockwise_logits(logits: jax.Array, targets: jax.Array, weights: jax.Array | None = None, *, ignore_index: int = -100, label_smoothing: float = 0, z_loss: float = 0, block_size: int = 8192, dtype: jnp.dtype | None = jnp.float32)` — [`L646`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L646) — Compute blockwise sparse cross-entropy without materializing full softmax.
- `cross_entropy_loss_and_accuracy(source, target, valid=None, compute_dtype: jnp.dtype = jnp.float32)` — [`L1526`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1526) — Compute cross-entropy loss and accuracy with optional masking.
- `cross_entropy_with_logits(logits: Array, targets: Array, z_loss: float)` — [`L1230`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1230) — Compute cross-entropy loss with optional z-loss regularization.
- `dynamic_cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray, weight: jnp.ndarray | None = None, ignore_index: int = -100, reduction: str = "mean", label_smoothing: float = 0, compute_dtype: jnp.dtype = jnp.float32)` — [`L1059`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1059) — Compute cross-entropy loss with flexible reduction and label smoothing.
- `fixed_cross_entropy(source: jax.Array | None, target: jax.Array | None, attention_mask: jax.Array | None = None, config: LossConfig | None = None, num_items_in_batch: int | None = None, batch: collections.abc.Mapping[str, Array] | None = None, **kwargs: tp.Any)` — [`L1895`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1895) — Compute cross-entropy loss with comprehensive configuration options. — documented in [easydel-infra-loss_utils](../../../concepts/easydel-infra-loss_utils.md)
- `fn(prev, x)` — [`L1659`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1659) — Scan step that propagates the previous non-zero value forward.
- `full_body(i, carry)` — [`L779`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L779) — Process a full block in the fori_loop.
- `get_factor_and_weight(loss_normalizing_factor: FACTOR_TYPE, batch: collections.abc.Mapping[str, Array], compute_dtype: jnp.dtype = jnp.float32)` — [`L1689`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1689) — Get loss normalizing factor and weights from batch data. — documented in [easydel-infra-loss_utils](../../../concepts/easydel-infra-loss_utils.md)
- `max_body(i, m)` — [`L530`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L530) — Compute running maximum across chunks.
- `onehot(labels, num_classes, on_value=1, off_value=0)` — [`L1192`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1192) — Create one-hot encoded representations of integer labels.
- `process_block(start: int, size: int, m, log_z, o, sum_logits, best_logit, best_id)` — [`L746`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L746) — Process a single vocabulary block and update accumulators.
- `resolve_causal_lm_chunk_token_size(hidden_states: jax.Array, vocab_size: int, config: LossConfig | None = None)` — [`L2068`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2068) — Choose an optimal token-dimension chunk size for chunked LM-head loss.
- `resolve_loss_strategy(loss_fn: tp.Callable[..., LossMetrics])` — [`L2578`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2578) — Wrap a loss callable in the appropriate ``BaseLossStrategy``.
- `sigmoid_cross_entropy_with_logits(logits: jnp.ndarray, labels: jnp.ndarray, weights: jnp.ndarray | None = None, label_smoothing: float = 0, axis: int | tuple | None = None)` — [`L1137`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L1137) — Compute sigmoid cross-entropy loss for multi-label classification.
- `sparse_cross_entropy_chunked_tokens(logits: jnp.ndarray, targets: jnp.ndarray, weights: jnp.ndarray | None = None, ignore_index: int = -100, label_smoothing: float = 0, z_loss: float = 0, reduction: str = "sum", token_chunk_size: int = 8192, compute_dtype: jnp.dtype = jnp.float32)` — [`L911`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L911) — Compute sparse cross-entropy loss with token sequence chunking.
- `sparse_cross_entropy_chunked_vocab(logits: jnp.ndarray, targets: jnp.ndarray, weights: jnp.ndarray | None = None, ignore_index: int = -100, label_smoothing: float = 0, z_loss: float = 0, reduction: str = "mean", chunk_size: int = 8192, compute_dtype: jnp.dtype = jnp.float32)` — [`L823`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L823) — Compute sparse cross-entropy loss with vocabulary chunking.
- `sum_body(i, s)` — [`L544`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L544) — Compute running sum of shifted exponentials.

## Module values
- `FACTOR_TYPE` — [`L342`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L342)
- `LOSS_MAPPING` — [`L2940`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L2940)
- `SLNF` — [`L339`](../../../../../../raw/code/EasyDeL/easydel/infra/loss_utils.py#L339) — documented in [easydel-infra-loss_utils](../../../concepts/easydel-infra-loss_utils.md)

