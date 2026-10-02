---
title: 'Module: axlearn/common/optimizer_base.py'
type: catalog
provenance: extracted
module: axlearn/common/optimizer_base.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.optimizer_base`/
symbols:
  PartitionedGradientTransformation: PartitionedGradientTransformation#
  OptStateSpec: OptStateSpec.
  NestedOptParam: NestedOptParam.
  OptParam: OptParam#
  PartitionedGradientTransformation.init: PartitionedGradientTransformation#init.
  PartitionedGradientTransformation.update: PartitionedGradientTransformation#update.
  PartitionedGradientTransformation.partition: PartitionedGradientTransformation#partition.
  OptParam.value: OptParam#value.
  TransformPartitionSpecFn: TransformPartitionSpecFn.
  OptParam.weight_decay_scale: OptParam#weight_decay_scale.
  OptParam.factorization_spec: OptParam#factorization_spec.
  TransformInitFn: TransformInitFn.
  OptParam.dtype: OptParam#dtype().
  OptParam.shape: OptParam#shape().
  TransformUpdateFn.__call__: TransformUpdateFn#__call__().
  TransformUpdateFn: TransformUpdateFn#
---
# Module: [`axlearn/common/optimizer_base.py`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py)

## Classes
### `OptParam`
- def: [`axlearn/common/optimizer_base.py:30`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L30)
- doc: A parameter to be optimized by an optimizer.
- signature: `class OptParam:`
- members:
  - `dtype(self)` — [`L38`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L38)
  - `shape(self)` — [`L42`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L42)
  - `factorization_spec` — [`L34`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L34)
  - `value` — [`L33`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L33)
  - `weight_decay_scale` — [`L35`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L35)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`FactorizationSpec`](base_layer.md#FactorizationSpec)
- used by: `_compare_layers`, [`_compute_updated_params`](learner.md#Learner._compute_updated_params), [`forward_and_backward`](learner.md#CompositeLearner.forward_and_backward), [`NestedOptParam`](optimizer_base.md#NestedOptParam), [`forward_and_backward`](learner.md#Learner.forward_and_backward), [`_value_and_grad`](learner.md#_value_and_grad), [`init`](learner.md#Learner.init), [`mock_updates`](update_transformation_test.md#mock_updates), [`init`](learner.md#CompositeLearner.init), [`_opt_params`](trainer.md#SpmdTrainer._opt_params), [`opt_params_from_model_params`](update_transformation_test.md#ConditionalUpdateTransformationTest.opt_params_from_model_params), [`opt_params`](update_transformation.md#Updates.opt_params), [`forward_and_backward`](learner.md#BaseLearner.forward_and_backward), [`init`](update_transformation.md#ConditionalUpdateTransformation.init), [`init`](learner_base.md#LearnerModule.init), [`_init`](optimizers.md#adastar_optimizer.init_fn._init), [`init`](update_transformation.md#OverrideInplaceUpdateTransformation.init), [`init`](update_transformation.md#WrappedPartitionedGradientTransformation.init), [`_update2`](optimizers.md#adastar_optimizer.update2_fn._update2), [`_get_optimizer_model_params`](learner.md#Learner._get_optimizer_model_params), [`should_update_with_optimizers`](learner.md#Learner.should_update_with_optimizers), [`param_values`](update_transformation.md#Updates.param_values), [`should_update_with_optimizers`](learner.md#CompositeLearner.should_update_with_optimizers), [`should_update_with_optimizers`](learner.md#BaseLearner.should_update_with_optimizers), [`maybe_override_scale`](optimizers.md#_weight_decay_scales.maybe_override_scale)  (1 test-only)

### `PartitionedGradientTransformation`  ·  implements/extends NamedTuple
- def: [`axlearn/common/optimizer_base.py:71`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L71)
- doc: An optax-style optimizer with a function to partition the inputs across devices.
- signature: `class PartitionedGradientTransformation(NamedTuple):`
- members:
  - `init` — [`L80`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L80)
  - `partition` — [`L82`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L82)
  - `update` — [`L81`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L81)
- uses (calls/refs, reference-scoped): [`TransformPartitionSpecFn`](optimizer_base.md#TransformPartitionSpecFn), [`TransformInitFn`](optimizer_base.md#TransformInitFn), [`TransformUpdateFn`](optimizer_base.md#TransformUpdateFn)
- used by: [`test_wrapped_partitioned_gradient_transformation`](update_transformation_test.md#UpdateTransformationTest.test_wrapped_partitioned_gradient_transformation), `_compare_layers`, [`_compute_updated_params`](learner.md#Learner._compute_updated_params), [`adastar_optimizer`](optimizers.md#adastar_optimizer), [`scale_by_schedule`](optimizers.md#scale_by_schedule), [`partition_fn`](optimizers.md#skip_and_clip_by_global_norm.partition_fn), [`update_fn`](optimizers.md#adastar_optimizer.update_fn), [`adafactor_optimizer`](optimizers.md#adafactor_optimizer), [`update_fn`](optimizers.md#skip_and_clip_by_global_norm.update_fn), [`init`](learner.md#Learner.init), [`update`](learner.md#CompositeLearner.update), [`scale_by_factored_rms`](factorized_rms.md#scale_by_factored_rms), [`adamw_optimizer`](optimizers.md#adamw_optimizer), [`adamw_decoupled_optimizer`](optimizers.md#adamw_decoupled_optimizer), [`add_decayed_weights`](optimizers.md#add_decayed_weights), [`param_ema`](optimizers.md#param_ema), [`init`](learner.md#CompositeLearner.init), [`offload_optimizer`](optimizers.md#offload_optimizer), [`skip_and_clip_by_global_norm`](optimizers.md#skip_and_clip_by_global_norm), [`chain`](optimizers.md#chain), [`sgd_optimizer`](optimizers.md#sgd_optimizer), [`ema`](optimizers.md#ema), [`transform_update`](update_transformation.md#WrappedPartitionedGradientTransformation.transform_update), [`lion_optimizer`](optimizers.md#lion_optimizer), [`with_partition_fn`](optimizers.md#with_partition_fn), [`init_fn`](optimizers.md#skip_and_clip_by_global_norm.init_fn), [`named_chain`](optimizers.md#named_chain), [`adam_optimizer`](optimizers.md#adam_optimizer), [`create_state_partition_specs`](learner.md#Learner.create_state_partition_specs), [`l2_regularizer`](optimizers.md#l2_regularizer), [`scale_by_lion`](optimizers.md#scale_by_lion), [`create_state_partition_specs`](learner.md#CompositeLearner.create_state_partition_specs), [`scale_update_per_param`](optimizers.md#scale_update_per_param), [`clip_by_global_norm`](optimizers.md#clip_by_global_norm), [`init`](update_transformation.md#OverrideInplaceUpdateTransformation.init), [`init`](update_transformation.md#WrappedPartitionedGradientTransformation.init), [`transformation`](update_transformation.md#WrappedPartitionedGradientTransformation.transformation), [`_no_op`](optimizers.md#_no_op), [`create_state_partition_specs`](update_transformation.md#OverrideInplaceUpdateTransformation.create_state_partition_specs), [`create_state_partition_specs`](update_transformation.md#WrappedPartitionedGradientTransformation.create_state_partition_specs)  (+18 more)

### `TransformUpdateFn`  ·  implements/extends Protocol
- def: [`axlearn/common/optimizer_base.py:54`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L54)
- doc: Similar to optax.TransformUpdateFn, but with two differences:
- signature: `class TransformUpdateFn(typing_extensions.Protocol):`
- protocol/private: `__call__`[`L61`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L61)
- uses (calls/refs, reference-scoped): [`NestedOptParam`](optimizer_base.md#NestedOptParam)
- used by: [`update`](optimizer_base.md#PartitionedGradientTransformation.update)

## Module values
- `NestedOptParam` — [`L47`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L47)
- `OptStateSpec` — [`L67`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L67)
- `TransformInitFn` — [`L51`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L51)
- `TransformPartitionSpecFn` — [`L68`](../../../../../../raw/code/axlearn/axlearn/common/optimizer_base.py#L68)

