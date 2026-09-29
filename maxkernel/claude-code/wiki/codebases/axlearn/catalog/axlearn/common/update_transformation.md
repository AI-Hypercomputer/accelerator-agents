---
title: 'Module: axlearn/common/update_transformation.py'
type: catalog
provenance: extracted
module: axlearn/common/update_transformation.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.update_transformation`/
symbols:
  ConditionalUpdateTransformation.transform_update: ConditionalUpdateTransformation#transform_update().
  ConditionalUpdateTransformation.create_state_partition_specs: ConditionalUpdateTransformation#create_state_partition_specs().
  OverrideInplaceUpdateTransformation.transform_update: OverrideInplaceUpdateTransformation#transform_update().
  WrappedPartitionedGradientTransformation.transform_update: WrappedPartitionedGradientTransformation#transform_update().
  Updates: Updates#
  Updates.param_specs: Updates#param_specs().
  ConditionalUpdateTransformation.__init__: ConditionalUpdateTransformation#__init__().
  OverrideInplaceUpdateTransformation._is_passthrough: OverrideInplaceUpdateTransformation#_is_passthrough().
  Updates.opt_params: Updates#opt_params.
  Updates.delta_updates: Updates#delta_updates.
  ForwardFn: ForwardFn#
  UpdateTransformation: UpdateTransformation#
  ConditionalUpdateTransformation.init: ConditionalUpdateTransformation#init().
  ForwardOutputs: ForwardOutputs#
  WrappedPartitionedGradientTransformation.transformation: WrappedPartitionedGradientTransformation#transformation.
  WrappedPartitionedGradientTransformation.init: WrappedPartitionedGradientTransformation#init().
  OverrideInplaceUpdateTransformation.init: OverrideInplaceUpdateTransformation#init().
  Updates.inplace_updates: Updates#inplace_updates.
  WrappedPartitionedGradientTransformation.create_state_partition_specs: WrappedPartitionedGradientTransformation#create_state_partition_specs().
  OverrideInplaceUpdateTransformation.create_state_partition_specs: OverrideInplaceUpdateTransformation#create_state_partition_specs().
  WrappedPartitionedGradientTransformation: WrappedPartitionedGradientTransformation#
  WrappedPartitionedGradientTransformation.Config.transformation: WrappedPartitionedGradientTransformation#Config#transformation.
  Updates.param_values: Updates#param_values().
  Updates.mask: Updates#mask().
  UpdateTransformation.transform_update: UpdateTransformation#transform_update().
  WrappedPartitionedGradientTransformation.Config: WrappedPartitionedGradientTransformation#Config#
  OverrideInplaceUpdateTransformation._keep_passthrough: OverrideInplaceUpdateTransformation#_keep_passthrough().
  BackwardOutputs: BackwardOutputs#
  ConditionalUpdateTransformation.Config: ConditionalUpdateTransformation#Config#
  ConditionalUpdateTransformation.Config.inner: ConditionalUpdateTransformation#Config#inner.
  OverrideInplaceUpdateTransformation.Config: OverrideInplaceUpdateTransformation#Config#
  ConditionalUpdateTransformation.inner: ConditionalUpdateTransformation#inner.
  ConditionalUpdateTransformation._update_schedule: ConditionalUpdateTransformation#_update_schedule.
  ForwardOutputs.aux: ForwardOutputs#aux.
  mask_tree: mask_tree().
  WrappedPartitionedGradientTransformation.__init__: WrappedPartitionedGradientTransformation#__init__().
  ForwardFn.__call__: ForwardFn#__call__().
  ForwardOutputs.loss: ForwardOutputs#loss.
  ForwardOutputs.output_collection: ForwardOutputs#output_collection.
  ForwardPass.model_params: ForwardPass#model_params.
  ForwardBackwardOutputs: ForwardBackwardOutputs#
  OverrideInplaceUpdateTransformation.Config.rules: OverrideInplaceUpdateTransformation#Config#rules.
  _ShouldUpdateState: _ShouldUpdateState#
  Updates.forward_pass: Updates#forward_pass.
  ForwardPass.outputs: ForwardPass#outputs.
  OverrideInplaceUpdateTransformation: OverrideInplaceUpdateTransformation#
  UpdateTransformation.__call__: UpdateTransformation#__call__().
  ConditionalUpdateTransformation: ConditionalUpdateTransformation#
  ForwardBackwardOutputs.forward_outputs: ForwardBackwardOutputs#forward_outputs.
  ForwardBackwardOutputs.backward_outputs: ForwardBackwardOutputs#backward_outputs.
  ForwardPass: ForwardPass#
  ConditionalUpdateTransformation.Config.update_schedule: ConditionalUpdateTransformation#Config#update_schedule.
  ConditionalUpdateTransformation.real_transform: ConditionalUpdateTransformation#real_transform().
  ConditionalUpdateTransformation.stop_transform: ConditionalUpdateTransformation#stop_transform().
  ForwardPass.forward_fn: ForwardPass#forward_fn.
  _ShouldUpdateState.count: _ShouldUpdateState#count.
  BackwardOutputs.updated_params: BackwardOutputs#updated_params.
  ForwardPass.inputs: ForwardPass#inputs.
---
# Module: [`axlearn/common/update_transformation.py`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py)

## Classes
### `BackwardOutputs`
- def: [`axlearn/common/update_transformation.py:351`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L351)
- members:
  - `updated_params` — [`L352`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L352)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested)
- used by: [`forward_and_backward`](learner.md#CompositeLearner.forward_and_backward), [`forward_and_backward`](learner.md#Learner.forward_and_backward), [`backward_outputs`](update_transformation.md#ForwardBackwardOutputs.backward_outputs)

### `ConditionalUpdateTransformation`  ·  implements/extends UpdateTransformation
- def: [`axlearn/common/update_transformation.py:122`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L122)
- doc: A wrapper around a `UpdateTransformation` to conditionally allow or skip
- signature: `class ConditionalUpdateTransformation(UpdateTransformation):`
- members:
  - `create_state_partition_specs(self, model_param_specs: Nested[ParameterSpec])` — [`L155`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L155)
  - `init(self, model_params: Nested[OptParam])` — [`L164`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L164)
  - `real_transform(_)` — [`L189`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L189)
  - `stop_transform(_)` — [`L192`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L192)
  - `transform_update(self, updates: Updates)` — [`L169`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L169)
  - `inner` — [`L150`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L150)
- protocol/private: `__init__`[`L143`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L143), `_update_schedule`[`L151`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L151)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`default_config`](config.md#Configurable.default_config), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`ParameterSpec`](base_layer.md#ParameterSpec), [`shape`](utils.md#TensorSpec.shape), [`mesh_axes`](utils.md#TensorSpec.mesh_axes), [`dtype`](utils.md#TensorSpec.dtype), [`OptStateSpec`](optimizer_base.md#OptStateSpec), [`Config`](module.md#Module.Config), [`__init__`](module.md#Module.__init__), [`OptParam`](optimizer_base.md#OptParam), [`get_invocation_context`](module.md#Module.get_invocation_context), [`Updates`](update_transformation.md#Updates), [`delta_updates`](update_transformation.md#Updates.delta_updates), [`UpdateTransformation`](update_transformation.md#UpdateTransformation), [`create_state_partition_specs`](learner_base.md#LearnerModule.create_state_partition_specs), [`init`](learner_base.md#LearnerModule.init), [`add_state_update`](module.md#Module.add_state_update), [`prune_tree`](utils.md#prune_tree), [`prune_empty`](utils.md#prune_empty), [`inplace_updates`](update_transformation.md#Updates.inplace_updates), [`LearnerModule`](learner_base.md#LearnerModule), [`WrappedPartitionedGradientTransformation`](update_transformation.md#WrappedPartitionedGradientTransformation), [`state`](module.md#Module.state), [`transform_update`](update_transformation.md#UpdateTransformation.transform_update), [`Config`](update_transformation.md#ConditionalUpdateTransformation.Config), [`inner`](update_transformation.md#ConditionalUpdateTransformation.Config.inner), [`get_state_updates`](module.md#InvocationContext.get_state_updates), [`_ShouldUpdateState`](update_transformation.md#_ShouldUpdateState), [`update_schedule`](update_transformation.md#ConditionalUpdateTransformation.Config.update_schedule), [`count`](update_transformation.md#_ShouldUpdateState.count)
- used by: [`get_learner_from_su`](update_transformation_test.md#ConditionalUpdateTransformationTest.get_learner_from_su), [`UpdateTransformation`](update_transformation.md#UpdateTransformation), [`create_state_partition_specs`](learner_base.md#LearnerModule.create_state_partition_specs), [`init`](learner_base.md#LearnerModule.init), [`transform_update`](update_transformation.md#UpdateTransformation.transform_update)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/update_transformation.py:389`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L389)
- doc: Configures `OverrideInplaceUpdateTransformation`.
- signature: `class Config(WrappedPartitionedGradientTransformation.Config):`
- members:
  - `inner` — [`L139`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L139)
  - `rules` — [`L396`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L396)
  - `transformation` — [`L82`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L82)
  - `update_schedule` — [`L141`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L141)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`InstantiableConfig`](config.md#InstantiableConfig), [`PartitionedGradientTransformation`](optimizer_base.md#PartitionedGradientTransformation), [`ConfigOr`](config.md#ConfigOr), [`WrappedPartitionedGradientTransformation`](update_transformation.md#WrappedPartitionedGradientTransformation), [`Config`](update_transformation.md#WrappedPartitionedGradientTransformation.Config)
- used by: [`__init__`](update_transformation.md#ConditionalUpdateTransformation.__init__), [`_is_passthrough`](update_transformation.md#OverrideInplaceUpdateTransformation._is_passthrough), [`transformation`](update_transformation.md#WrappedPartitionedGradientTransformation.transformation), [`_update_schedule`](update_transformation.md#ConditionalUpdateTransformation._update_schedule)

### `ForwardBackwardOutputs`
- def: [`axlearn/common/update_transformation.py:338`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L338)
- signature: `class ForwardBackwardOutputs:`
- members:
  - `backward_outputs` — [`L340`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L340)
  - `forward_outputs` — [`L339`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L339)
- uses (calls/refs, reference-scoped): [`ForwardOutputs`](update_transformation.md#ForwardOutputs), [`BackwardOutputs`](update_transformation.md#BackwardOutputs)
- used by: [`forward_and_backward`](learner.md#CompositeLearner.forward_and_backward), [`forward_and_backward`](learner.md#Learner.forward_and_backward), [`forward_and_backward`](learner.md#BaseLearner.forward_and_backward)

### `ForwardFn`
- def: [`axlearn/common/update_transformation.py:317`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L317)
- members:
  - `__call__(self, *, model_params: Nested[Tensor], inputs: Any)` — [`L320`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L320) — The forward function of a module.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`ForwardOutputs`](update_transformation.md#ForwardOutputs)
- used by: [`forward_and_backward`](learner.md#CompositeLearner.forward_and_backward), [`forward_and_backward`](learner.md#Learner.forward_and_backward), [`_value_and_grad`](learner.md#_value_and_grad), [`forward_and_backward`](learner.md#BaseLearner.forward_and_backward), [`_split_gradients`](learner.md#_split_gradients), [`decorator`](gradient_accumulation.md#with_minibatch_steps.decorator), [`with_minibatch_steps`](gradient_accumulation.md#with_minibatch_steps), [`_as_loss_fn`](learner.md#_as_loss_fn), [`sequential_vmap`](gradient_accumulation.md#with_minibatch_steps.decorator.sequential_vmap), [`ForwardFnTransformation`](learner.md#ForwardFnTransformation), [`forward_fn`](update_transformation.md#ForwardPass.forward_fn)

### `ForwardOutputs`
- def: [`axlearn/common/update_transformation.py:343`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L343)
- members:
  - `aux` — [`L346`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L346)
  - `loss` — [`L345`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L345)
  - `output_collection` — [`L347`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L347)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`OutputCollection`](module.md#OutputCollection)
- used by: [`_train_step`](trainer.md#SpmdTrainer._train_step), [`fwd_helper`](gradient_accumulation.md#with_minibatch_steps.decorator.fwd_helper), [`_forward`](trainer.md#SpmdTrainer._forward), [`forward`](learner.md#_as_loss_fn.forward), [`scan_body`](gradient_accumulation.md#with_minibatch_steps.decorator.fwd_helper.scan_body), [`loss_fn`](update_transformation_test.md#ConditionalUpdateTransformationTest.loss_fn), [`func_fwd`](gradient_accumulation.md#with_minibatch_steps.decorator.func_fwd), [`original_func_positional_args`](gradient_accumulation.md#with_minibatch_steps.decorator.original_func_positional_args), [`func`](gradient_accumulation.md#with_minibatch_steps.decorator.func), [`wrapper`](gradient_accumulation.md#with_minibatch_steps.decorator.sequential_vmap.wrapper), [`filtered_forward`](learner.md#_split_gradients.filtered_forward), [`__call__`](update_transformation.md#ForwardFn.__call__), [`outputs`](update_transformation.md#ForwardPass.outputs), [`forward_outputs`](update_transformation.md#ForwardBackwardOutputs.forward_outputs)

### `ForwardPass`
- def: [`axlearn/common/update_transformation.py:299`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L299)
- members:
  - `forward_fn` — [`L305`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L305)
  - `inputs` — [`L309`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L309)
  - `model_params` — [`L311`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L311)
  - `outputs` — [`L314`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L314)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`ForwardFn`](update_transformation.md#ForwardFn), [`ForwardOutputs`](update_transformation.md#ForwardOutputs)
- used by: [`forward_and_backward`](learner.md#CompositeLearner.forward_and_backward), [`forward_and_backward`](learner.md#Learner.forward_and_backward), [`forward`](learner.md#_as_loss_fn.forward), [`forward_pass`](update_transformation.md#Updates.forward_pass)

### `OverrideInplaceUpdateTransformation`  ·  implements/extends WrappedPartitionedGradientTransformation
- def: [`axlearn/common/update_transformation.py:380`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L380)
- doc: An update transformation that provides rules to override inplace updates.
- signature: `class OverrideInplaceUpdateTransformation(WrappedPartitionedGradientTransformation):`
- members:
  - `_is_passthrough(self, params: Nested[Any])` — [`L398`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L398) — Gets a pytree of bools with True indicating a parameter or gradient is passthrough.
  - `_keep_passthrough(self, params: Nested[Any])` — [`L411`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L411) — Given a pytree of params, keeps only the passthrough params.
  - `create_state_partition_specs(self, model_param_specs: Nested[ParameterSpec])` — [`L415`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L415)
  - `init(self, model_params: Nested[OptParam])` — [`L423`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L423)
  - `transform_update(self, updates: Updates)` — [`L426`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L426)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`ParameterSpec`](base_layer.md#ParameterSpec), [`flatten_items`](utils.md#flatten_items), [`OptParam`](optimizer_base.md#OptParam), [`init`](optimizer_base.md#PartitionedGradientTransformation.init), [`tree_paths`](utils.md#tree_paths), [`partition`](optimizer_base.md#PartitionedGradientTransformation.partition), [`transform_update`](update_transformation.md#WrappedPartitionedGradientTransformation.transform_update), [`Updates`](update_transformation.md#Updates), [`delta_updates`](update_transformation.md#Updates.delta_updates), [`transformation`](update_transformation.md#WrappedPartitionedGradientTransformation.transformation), [`inplace_updates`](update_transformation.md#Updates.inplace_updates), [`match_regex_rules`](utils.md#match_regex_rules), [`WrappedPartitionedGradientTransformation`](update_transformation.md#WrappedPartitionedGradientTransformation), [`mask`](update_transformation.md#Updates.mask), [`tree_merge`](utils.md#tree_merge), [`Config`](update_transformation.md#OverrideInplaceUpdateTransformation.Config), [`mask_tree`](update_transformation.md#mask_tree), [`rules`](update_transformation.md#OverrideInplaceUpdateTransformation.Config.rules), [`non_empty_leaf_merge_fn`](utils.md#non_empty_leaf_merge_fn)
- used by: [`test_override_inplace_update_transformation`](update_transformation_test.md#OverrideInplaceUpdateTransformationTest.test_override_inplace_update_transformation), [`transform_update`](update_transformation.md#WrappedPartitionedGradientTransformation.transform_update), [`init`](update_transformation.md#WrappedPartitionedGradientTransformation.init), [`create_state_partition_specs`](update_transformation.md#WrappedPartitionedGradientTransformation.create_state_partition_specs), [`WrappedPartitionedGradientTransformation`](update_transformation.md#WrappedPartitionedGradientTransformation)

### `UpdateTransformation`  ·  implements/extends LearnerModule
- def: [`axlearn/common/update_transformation.py:54`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L54)
- doc: A Module to transform a model update.
- signature: `class UpdateTransformation(LearnerModule):`
- members:
  - `__call__(self, updates: Updates)` — [`L70`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L70) — Alias for `transform_update()`.
  - `transform_update(self, updates: Updates)` — [`L66`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L66) — Compute the value and grad of `fun`.
- uses (calls/refs, reference-scoped): [`transform_update`](update_transformation.md#ConditionalUpdateTransformation.transform_update), [`transform_update`](update_transformation.md#WrappedPartitionedGradientTransformation.transform_update), [`Updates`](update_transformation.md#Updates), [`LearnerModule`](learner_base.md#LearnerModule), [`WrappedPartitionedGradientTransformation`](update_transformation.md#WrappedPartitionedGradientTransformation), [`ConditionalUpdateTransformation`](update_transformation.md#ConditionalUpdateTransformation)
- used by: [`test_wrapped_partitioned_gradient_transformation`](update_transformation_test.md#UpdateTransformationTest.test_wrapped_partitioned_gradient_transformation), [`transform_update`](update_transformation.md#ConditionalUpdateTransformation.transform_update), [`test_override_inplace_update_transformation`](update_transformation_test.md#OverrideInplaceUpdateTransformationTest.test_override_inplace_update_transformation), [`LearnerModule`](learner_base.md#LearnerModule), [`WrappedPartitionedGradientTransformation`](update_transformation.md#WrappedPartitionedGradientTransformation), [`optimizer`](learner.md#Learner.optimizer), [`Config`](update_transformation.md#WrappedPartitionedGradientTransformation.Config), [`Config`](update_transformation.md#ConditionalUpdateTransformation.Config), [`inner`](update_transformation.md#ConditionalUpdateTransformation.inner), [`ConditionalUpdateTransformation`](update_transformation.md#ConditionalUpdateTransformation)

### `Updates`
- def: [`axlearn/common/update_transformation.py:207`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L207)
- members:
  - `mask(self, keep: Callable[[Nested], Nested[bool]], *, fields: Sequence[Literal["opt_params", "delta_updates", "inplace_updates", "forward_pass"]] = ("opt_params", "delta_updates", "inplace_updates"))` — [`L254`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L254) — Return a copy of this instance where the values of the field have been
  - `param_specs(self)` — [`L240`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L240) — Returns a tree with the same structure as `opt_params` with the metadata of each
  - `param_values(self)` — [`L234`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L234) — Returns a tree with the same structure as `opt_params` with the value of each param.
  - `delta_updates` — [`L224`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L224)
  - `forward_pass` — [`L232`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L232)
  - `inplace_updates` — [`L229`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L229)
  - `opt_params` — [`L220`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L220)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`ParameterSpec`](base_layer.md#ParameterSpec), [`shape`](utils.md#TensorSpec.shape), [`dtype`](utils.md#TensorSpec.dtype), [`OptParam`](optimizer_base.md#OptParam), [`weight_decay_scale`](base_layer.md#ParameterSpec.weight_decay_scale), [`factorization`](base_layer.md#ParameterSpec.factorization), [`mask_tree`](update_transformation.md#mask_tree), [`ForwardPass`](update_transformation.md#ForwardPass)
- used by: [`test_wrapped_partitioned_gradient_transformation`](update_transformation_test.md#UpdateTransformationTest.test_wrapped_partitioned_gradient_transformation), [`forward_and_backward`](learner.md#CompositeLearner.forward_and_backward), [`forward_and_backward`](learner.md#Learner.forward_and_backward), [`transform_update`](update_transformation.md#ConditionalUpdateTransformation.transform_update), [`test_override_inplace_update_transformation`](update_transformation_test.md#OverrideInplaceUpdateTransformationTest.test_override_inplace_update_transformation), [`_value_and_grad`](learner.md#_value_and_grad), [`update`](learner.md#CompositeLearner.update), [`mock_updates`](update_transformation_test.md#mock_updates), [`transform_update`](update_transformation.md#OverrideInplaceUpdateTransformation.transform_update), [`transform_update`](update_transformation.md#WrappedPartitionedGradientTransformation.transform_update), [`update`](learner.md#Learner.update), [`test_param_specs`](update_transformation_test.md#UpdatesTest.test_param_specs), [`test_mask`](update_transformation_test.md#UpdatesTest.test_mask), [`update`](learner.md#BaseLearner.update), [`test_param_values`](update_transformation_test.md#UpdatesTest.test_param_values), [`transform_update`](update_transformation.md#UpdateTransformation.transform_update), [`__call__`](update_transformation.md#UpdateTransformation.__call__), [`real_transform`](update_transformation.md#ConditionalUpdateTransformation.real_transform), [`stop_transform`](update_transformation.md#ConditionalUpdateTransformation.stop_transform)  (1 test-only)

### `WrappedPartitionedGradientTransformation`  ·  implements/extends UpdateTransformation
- def: [`axlearn/common/update_transformation.py:75`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L75)
- doc: An adapter allowing a `PartitionedGradientTransformation` to be used as an
- signature: `class WrappedPartitionedGradientTransformation(UpdateTransformation):`
- members:
  - `create_state_partition_specs(self, model_param_specs: Nested[ParameterSpec])` — [`L94`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L94)
  - `init(self, model_params: Nested[OptParam])` — [`L102`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L102)
  - `transform_update(self, updates: Updates)` — [`L105`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L105) — Run the `PartionedGradientTransformation.update` function to compute updates.
  - `transformation` — [`L86`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L86)
- protocol/private: `__init__`[`L84`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L84)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`Module`](module.md#Module), [`ParameterSpec`](base_layer.md#ParameterSpec), [`PartitionedGradientTransformation`](optimizer_base.md#PartitionedGradientTransformation), [`maybe_instantiate`](config.md#maybe_instantiate), [`__init__`](module.md#Module.__init__), [`OptParam`](optimizer_base.md#OptParam), [`init`](optimizer_base.md#PartitionedGradientTransformation.init), [`update`](optimizer_base.md#PartitionedGradientTransformation.update), [`partition`](optimizer_base.md#PartitionedGradientTransformation.partition), [`transform_update`](update_transformation.md#OverrideInplaceUpdateTransformation.transform_update), [`get_invocation_context`](module.md#Module.get_invocation_context), [`Updates`](update_transformation.md#Updates), [`delta_updates`](update_transformation.md#Updates.delta_updates), [`opt_params`](update_transformation.md#Updates.opt_params), [`UpdateTransformation`](update_transformation.md#UpdateTransformation), [`init`](update_transformation.md#OverrideInplaceUpdateTransformation.init), [`create_state_partition_specs`](update_transformation.md#OverrideInplaceUpdateTransformation.create_state_partition_specs), [`state`](module.md#Module.state), [`transformation`](update_transformation.md#WrappedPartitionedGradientTransformation.Config.transformation), [`set_state_update`](module.md#InvocationContext.set_state_update), [`Config`](update_transformation.md#WrappedPartitionedGradientTransformation.Config), [`OverrideInplaceUpdateTransformation`](update_transformation.md#OverrideInplaceUpdateTransformation)
- used by: [`test_wrapped_partitioned_gradient_transformation`](update_transformation_test.md#UpdateTransformationTest.test_wrapped_partitioned_gradient_transformation), [`transform_update`](update_transformation.md#OverrideInplaceUpdateTransformation.transform_update), [`__init__`](learner.md#Learner.__init__), [`__init__`](update_transformation.md#ConditionalUpdateTransformation.__init__), [`UpdateTransformation`](update_transformation.md#UpdateTransformation), [`create_state_partition_specs`](learner_base.md#LearnerModule.create_state_partition_specs), [`init`](learner_base.md#LearnerModule.init), [`init`](update_transformation.md#OverrideInplaceUpdateTransformation.init), [`create_state_partition_specs`](update_transformation.md#OverrideInplaceUpdateTransformation.create_state_partition_specs), [`transform_update`](update_transformation.md#UpdateTransformation.transform_update), [`Config`](update_transformation.md#OverrideInplaceUpdateTransformation.Config), [`OverrideInplaceUpdateTransformation`](update_transformation.md#OverrideInplaceUpdateTransformation)

### `_ShouldUpdateState`  ·  implements/extends NamedTuple
- def: [`axlearn/common/update_transformation.py:118`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L118)
- signature: `class _ShouldUpdateState(NamedTuple):`
- members:
  - `count` — [`L119`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L119)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor)
- used by: [`transform_update`](update_transformation.md#ConditionalUpdateTransformation.transform_update), [`create_state_partition_specs`](update_transformation.md#ConditionalUpdateTransformation.create_state_partition_specs), [`init`](update_transformation.md#ConditionalUpdateTransformation.init)

## Functions
- `mask_tree(tree: dict, *, keep: dict, mask_value: Any)` — [`L355`](../../../../../../raw/code/axlearn/axlearn/common/update_transformation.py#L355) — Mask out tree leaves that are not transformed by the optimizer.

