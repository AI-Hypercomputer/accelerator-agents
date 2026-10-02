---
title: 'Module: axlearn/common/update_transformation_test.py'
type: catalog
provenance: extracted
module: axlearn/common/update_transformation_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.update_transformation_test`/
symbols:
  UpdateTransformationTest.test_wrapped_partitioned_gradient_transformation: UpdateTransformationTest#test_wrapped_partitioned_gradient_transformation().
  ConditionalUpdateTransformationTest.test_conditional_update_transformation: ConditionalUpdateTransformationTest#test_conditional_update_transformation().
  OverrideInplaceUpdateTransformationTest.test_override_inplace_update_transformation: OverrideInplaceUpdateTransformationTest#test_override_inplace_update_transformation().
  mock_updates: mock_updates().
  UpdatesTest.test_param_specs: UpdatesTest#test_param_specs().
  ConditionalUpdateTransformationTest.get_learner_from_su: ConditionalUpdateTransformationTest#get_learner_from_su().
  ConditionalUpdateTransformationTest.opt_params_from_model_params: ConditionalUpdateTransformationTest#opt_params_from_model_params().
  UpdatesTest.test_mask: UpdatesTest#test_mask().
  ConditionalUpdateTransformationTest.loss_fn: ConditionalUpdateTransformationTest#loss_fn().
  UpdatesTest.test_param_values: UpdatesTest#test_param_values().
  mock_params: mock_params().
  UpdatesTest.expected_result: UpdatesTest#expected_result().
  ConditionalUpdateTransformationTest.check_state_and_model_params_equal: ConditionalUpdateTransformationTest#check_state_and_model_params_equal().
  UpdateTransformationTest: UpdateTransformationTest#
  UpdatesTest: UpdatesTest#
  OverrideInplaceUpdateTransformationTest: OverrideInplaceUpdateTransformationTest#
  ConditionalUpdateTransformationTest: ConditionalUpdateTransformationTest#
  LearnerStep: LearnerStep#
  LearnerStep.state: LearnerStep#state.
  LearnerStep.model_params: LearnerStep#model_params.
---
# Module: [`axlearn/common/update_transformation_test.py`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py)

## Classes
### `ConditionalUpdateTransformationTest`  ·  implements/extends TestCase
- def: [`axlearn/common/update_transformation_test.py:327`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L327)
- doc: Tests for `OverrideInplaceUpdateTransformation`.
- signature: `class ConditionalUpdateTransformationTest(test_utils.TestCase):`
- members:
  - `check_state_and_model_params_equal(step1, step2)` — [`L446`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L446)
  - `get_learner_from_su(should_update_schedule_fn=None)` — [`L332`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L332)
  - `loss_fn(model_params, inputs)` — [`L394`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L394)
  - `opt_params_from_model_params(model_params: NestedTensor, param_specs: NestedTree)` — [`L374`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L374) — Returns a tree of OptParam for Learner.{init,update}.
  - `test_conditional_update_transformation(self, optimizer_type)` — [`L331`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L331)
- uses (calls/refs, reference-scoped): [`default_config`](config.md#Configurable.default_config), [`NestedTensor`](utils.md#NestedTensor), [`instantiate`](config.md#Configurable.Config.instantiate), [`functional`](module.md#functional), [`ParameterSpec`](base_layer.md#ParameterSpec), [`shape`](utils.md#TensorSpec.shape), [`mesh_axes`](utils.md#TensorSpec.mesh_axes), [`config_for_function`](config.md#config_for_function), [`dtype`](utils.md#TensorSpec.dtype), [`NestedOptParam`](optimizer_base.md#NestedOptParam), [`new_output_collection`](module.md#new_output_collection), [`OptParam`](optimizer_base.md#OptParam), [`init`](learner.md#Learner.init), [`adamw_optimizer`](optimizers.md#adamw_optimizer), [`chain`](optimizers.md#chain), [`NestedTree`](utils.md#NestedTree), [`state_updates`](module.md#OutputCollection.state_updates), [`weight_decay_scale`](base_layer.md#ParameterSpec.weight_decay_scale), [`ForwardOutputs`](update_transformation.md#ForwardOutputs), [`factorization`](base_layer.md#ParameterSpec.factorization), [`clip_by_global_norm`](optimizers.md#clip_by_global_norm), [`value`](optimizer_base.md#OptParam.value), [`Learner`](learner.md#Learner), [`weight_decay_scale`](optimizer_base.md#OptParam.weight_decay_scale), [`aux`](update_transformation.md#ForwardOutputs.aux), [`factorization_spec`](optimizer_base.md#OptParam.factorization_spec), [`loss`](update_transformation.md#ForwardOutputs.loss), [`output_collection`](update_transformation.md#ForwardOutputs.output_collection), [`ConditionalUpdateTransformation`](update_transformation.md#ConditionalUpdateTransformation), [`LearnerStep`](update_transformation_test.md#LearnerStep), [`model_params`](update_transformation_test.md#LearnerStep.model_params), [`state`](update_transformation_test.md#LearnerStep.state)  (2 test-only)
- used by: (1 test-only callers)

### `LearnerStep`  ·  implements/extends NamedTuple
- def: [`axlearn/common/update_transformation_test.py:322`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L322)
- signature: `class LearnerStep(NamedTuple):`
- members:
  - `model_params` — [`L324`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L324)
  - `state` — [`L323`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L323)
- used by: [`test_conditional_update_transformation`](update_transformation_test.md#ConditionalUpdateTransformationTest.test_conditional_update_transformation)

### `OverrideInplaceUpdateTransformationTest`  ·  implements/extends TestCase
- def: [`axlearn/common/update_transformation_test.py:284`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L284)
- doc: Tests for `OverrideInplaceUpdateTransformation`.
- signature: `class OverrideInplaceUpdateTransformationTest(test_utils.TestCase):`
- members:
  - `test_override_inplace_update_transformation(self)` — [`L287`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L287)
- uses (calls/refs, reference-scoped): [`default_config`](config.md#Configurable.default_config), [`instantiate`](config.md#Configurable.Config.instantiate), [`functional`](module.md#functional), [`config_for_function`](config.md#config_for_function), [`mock_updates`](update_transformation_test.md#mock_updates), [`tree_paths`](utils.md#tree_paths), [`adamw_optimizer`](optimizers.md#adamw_optimizer), [`param_specs`](update_transformation.md#Updates.param_specs), [`opt_params`](update_transformation.md#Updates.opt_params), [`UpdateTransformation`](update_transformation.md#UpdateTransformation), [`init`](update_transformation.md#OverrideInplaceUpdateTransformation.init), [`create_state_partition_specs`](update_transformation.md#OverrideInplaceUpdateTransformation.create_state_partition_specs), [`OverrideInplaceUpdateTransformation`](update_transformation.md#OverrideInplaceUpdateTransformation)  (1 test-only)
- used by: (1 test-only callers)

### `UpdateTransformationTest`  ·  implements/extends TestCase
- def: [`axlearn/common/update_transformation_test.py:45`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L45)
- doc: Tests related to the `UpdateTransformation` API.
- signature: `class UpdateTransformationTest(test_utils.TestCase):`
- members:
  - `test_wrapped_partitioned_gradient_transformation(self, *, use_config: bool = False, optimizer: PartitionedGradientTransformation)` — [`L110`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L110)
- uses (calls/refs, reference-scoped): [`default_config`](config.md#Configurable.default_config), [`instantiate`](config.md#Configurable.Config.instantiate), [`functional`](module.md#functional), [`PartitionedGradientTransformation`](optimizer_base.md#PartitionedGradientTransformation), [`config_for_function`](config.md#config_for_function), [`maybe_instantiate`](config.md#maybe_instantiate), [`adastar_optimizer`](optimizers.md#adastar_optimizer), [`new_output_collection`](module.md#new_output_collection), [`summaries`](module.md#OutputCollection.summaries), [`init`](optimizer_base.md#PartitionedGradientTransformation.init), [`update`](optimizer_base.md#PartitionedGradientTransformation.update), [`mock_updates`](update_transformation_test.md#mock_updates), [`adamw_optimizer`](optimizers.md#adamw_optimizer), [`adamw_decoupled_optimizer`](optimizers.md#adamw_decoupled_optimizer), [`partition`](optimizer_base.md#PartitionedGradientTransformation.partition), [`set_current_context`](module.md#set_current_context), [`chain`](optimizers.md#chain), [`sgd_optimizer`](optimizers.md#sgd_optimizer), [`Updates`](update_transformation.md#Updates), [`InvocationContext`](module.md#InvocationContext), [`param_specs`](update_transformation.md#Updates.param_specs), [`delta_updates`](update_transformation.md#Updates.delta_updates), [`opt_params`](update_transformation.md#Updates.opt_params), [`state_updates`](module.md#OutputCollection.state_updates), [`UpdateTransformation`](update_transformation.md#UpdateTransformation), [`init`](update_transformation.md#WrappedPartitionedGradientTransformation.init), [`inplace_updates`](update_transformation.md#Updates.inplace_updates), [`create_state_partition_specs`](update_transformation.md#WrappedPartitionedGradientTransformation.create_state_partition_specs), [`WrappedPartitionedGradientTransformation`](update_transformation.md#WrappedPartitionedGradientTransformation), [`get_summaries`](module.md#InvocationContext.get_summaries)  (2 test-only)
- used by: (1 test-only callers)

### `UpdatesTest`  ·  implements/extends TestCase
- def: [`axlearn/common/update_transformation_test.py:209`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L209)
- doc: Tests related to the `Updates` interface.
- signature: `class UpdatesTest(test_utils.TestCase):`
- members:
  - `expected_result(tree: Nested)` — [`L263`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L263)
  - `test_mask(self, fields: Sequence[str], masked_names)` — [`L254`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L254)
  - `test_param_specs(self)` — [`L220`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L220)
  - `test_param_values(self)` — [`L212`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L212)
- uses (calls/refs, reference-scoped): [`Nested`](utils.md#Nested), [`ParameterSpec`](base_layer.md#ParameterSpec), [`shape`](utils.md#TensorSpec.shape), [`dtype`](utils.md#TensorSpec.dtype), [`VDict`](utils.md#VDict), [`FactorizationSpec`](base_layer.md#FactorizationSpec), [`mock_updates`](update_transformation_test.md#mock_updates), [`param_specs`](update_transformation.md#Updates.param_specs), [`delta_updates`](update_transformation.md#Updates.delta_updates), [`opt_params`](update_transformation.md#Updates.opt_params), [`weight_decay_scale`](base_layer.md#ParameterSpec.weight_decay_scale), [`factorization`](base_layer.md#ParameterSpec.factorization), [`inplace_updates`](update_transformation.md#Updates.inplace_updates), [`param_values`](update_transformation.md#Updates.param_values), [`mask`](update_transformation.md#Updates.mask), [`mock_params`](update_transformation_test.md#mock_params)  (2 test-only)
- used by: (1 test-only callers)

## Functions
- `mock_params()` — [`L172`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L172) — Returns mock model params.
- `mock_updates(state_param_none: bool = True)` — [`L183`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation_test.py#L183) — Create an updates object with various semi-reasonable values.

