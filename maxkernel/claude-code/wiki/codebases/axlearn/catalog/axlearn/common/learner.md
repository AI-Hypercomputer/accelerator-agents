---
title: 'Module: axlearn/common/learner.py'
type: catalog
provenance: extracted
module: axlearn/common/learner.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.learner`/
symbols:
  Learner._compute_updated_params: Learner#_compute_updated_params().
  CompositeLearner.forward_and_backward: CompositeLearner#forward_and_backward().
  Learner.forward_and_backward: Learner#forward_and_backward().
  Learner.init: Learner#init().
  _value_and_grad: _value_and_grad().
  CompositeLearner.update: CompositeLearner#update().
  CompositeLearner.init: CompositeLearner#init().
  Learner.update: Learner#update().
  Learner.__init__: Learner#__init__().
  CompositeLearner.__init__: CompositeLearner#__init__().
  _as_loss_fn.forward: _as_loss_fn().forward().
  Learner.create_state_partition_specs: Learner#create_state_partition_specs().
  BaseLearner.forward_and_backward: BaseLearner#forward_and_backward().
  CompositeLearner.create_state_partition_specs: CompositeLearner#create_state_partition_specs().
  Learner._update_types: Learner#_update_types().
  CompositeLearner._learner_tree: CompositeLearner#_learner_tree().
  Learner.ema: Learner#ema.
  Learner._get_optimizer_model_params: Learner#_get_optimizer_model_params().
  Learner.should_update_with_optimizers: Learner#should_update_with_optimizers().
  BaseLearner.update: BaseLearner#update().
  BaseLearner: BaseLearner#
  CompositeLearner.should_update_with_optimizers: CompositeLearner#should_update_with_optimizers().
  _split_gradients: _split_gradients().
  Learner: Learner#
  Learner.Config: Learner#Config#
  Learner.Config.ema: Learner#Config#ema.
  Learner.optimizer: Learner#optimizer.
  CompositeLearner.ema: CompositeLearner#ema.
  BaseLearner.should_update_with_optimizers: BaseLearner#should_update_with_optimizers().
  Learner._forward_fn_transformation: Learner#_forward_fn_transformation.
  _apply_updates: _apply_updates().
  CompositeLearner.Config.ema: CompositeLearner#Config#ema.
  should_update_with_optimizers: should_update_with_optimizers().
  should_apply_state_updates: should_apply_state_updates().
  Learner.Config.optimizer: Learner#Config#optimizer.
  CompositeLearner.Config: CompositeLearner#Config#
  _split_gradients.filtered_forward: _split_gradients().filtered_forward().
  UpdateType: UpdateType#
  CompositeLearner.should_apply: CompositeLearner#should_apply().
  Learner.Config.forward_fn_transformation: Learner#Config#forward_fn_transformation.
  CompositeLearner.Config.learners: CompositeLearner#Config#learners.
  CompositeLearner.Config.rules: CompositeLearner#Config#rules.
  _as_loss_fn: _as_loss_fn().
  ForwardFnTransformation: ForwardFnTransformation.
  _split_gradients.split_params_fn: _split_gradients().split_params_fn().
  UpdateType.ALL_UPDATES: UpdateType#ALL_UPDATES.
  Learner.Config.update_rules: Learner#Config#update_rules.
  CompositeLearner: CompositeLearner#
  _value_and_grad.split_params_fn: _value_and_grad().split_params_fn().
  UpdateType.OPTIMIZERS: UpdateType#OPTIMIZERS.
  UpdateType.STATE_UPDATES: UpdateType#STATE_UPDATES.
  UpdateType.NO_UPDATE: UpdateType#NO_UPDATE.
  Learner.Config.enable_per_variable_summaries: Learner#Config#enable_per_variable_summaries.
---
# Module: [`axlearn/common/learner.py`](../../../../../../raw/code/axlearn/axlearn/common/learner.py)

## Classes
### `BaseLearner`  ·  implements/extends LearnerModule
- def: [`axlearn/common/learner.py:97`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L97)
- doc: The base class of a learner.
- signature: `class BaseLearner(LearnerModule):`
- members:
  - `forward_and_backward(self, *, fn: ForwardFn, inputs: Nested[Tensor], opt_params: Nested[OptParam])` — [`L112`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L112) — Computes updates to the parameters `opt_params` of loss function `fn`.
  - `should_update_with_optimizers(self, model_params: Nested[OptParam])` — [`L127`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L127) — Returns whether each parameter should be updated with the optimizers (delta updates).
  - `update(self, *, updates: Updates)` — [`L100`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L100) — Computes `model_params` updates from `update`.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`forward_and_backward`](learner.md#CompositeLearner.forward_and_backward), [`forward_and_backward`](learner.md#Learner.forward_and_backward), [`OptParam`](optimizer_base.md#OptParam), [`update`](learner.md#CompositeLearner.update), [`update`](learner.md#Learner.update), [`Updates`](update_transformation.md#Updates), [`ForwardFn`](update_transformation.md#ForwardFn), [`LearnerModule`](learner_base.md#LearnerModule), [`should_update_with_optimizers`](learner.md#Learner.should_update_with_optimizers), [`should_update_with_optimizers`](learner.md#CompositeLearner.should_update_with_optimizers), [`Learner`](learner.md#Learner), [`ForwardBackwardOutputs`](update_transformation.md#ForwardBackwardOutputs), [`CompositeLearner`](learner.md#CompositeLearner)
- used by: [`LearnerModule`](learner_base.md#LearnerModule), [`Config`](learner.md#Learner.Config), [`Learner`](learner.md#Learner), [`Config`](learner.md#CompositeLearner.Config), [`CompositeLearner`](learner.md#CompositeLearner)

### `CompositeLearner`  ·  implements/extends BaseLearner
- def: [`axlearn/common/learner.py:384`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L384)
- doc: The composite learner supports different sub learners on different subset of parameters.
- signature: `class CompositeLearner(BaseLearner):`
- members:
  - `_learner_tree(self, params: Nested[Any])` — [`L434`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L434) — Returns a tree of the same structure as params where each leaf is the name of the
  - `create_state_partition_specs(self, model_param_specs: Nested[ParameterSpec])` — [`L454`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L454)
  - `forward_and_backward(self, *, fn: ForwardFn, inputs: Nested[Tensor], opt_params: Nested[OptParam])` — [`L554`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L554)
  - `init(self, model_params: Nested[OptParam])` — [`L479`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L479)
  - `should_apply(tree: Nested[Any])` — [`L518`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L518)
  - `should_update_with_optimizers(self, model_params: Nested[OptParam])` — [`L574`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L574) — Returns whether each parameter should be updated with the optimizers.
  - `update(self, updates: Updates)` — [`L502`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L502) — Computes `model_params` updates from `update`.
  - `ema` — [`L432`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L432)
- protocol/private: `__init__`[`L412`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L412)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`ParameterSpec`](base_layer.md#ParameterSpec), [`PartitionedGradientTransformation`](optimizer_base.md#PartitionedGradientTransformation), [`child_context`](module.md#child_context), [`__init__`](module.md#Module.__init__), [`new_output_collection`](module.md#new_output_collection), [`path`](module.md#Module.path), [`OptParam`](optimizer_base.md#OptParam), [`_value_and_grad`](learner.md#_value_and_grad), [`init`](optimizer_base.md#PartitionedGradientTransformation.init), [`update`](optimizer_base.md#PartitionedGradientTransformation.update), [`tree_paths`](utils.md#tree_paths), [`instantiate`](config.md#InstantiableConfig.instantiate), [`partition`](optimizer_base.md#PartitionedGradientTransformation.partition), [`Updates`](update_transformation.md#Updates), [`opt_params`](update_transformation.md#Updates.opt_params), [`ForwardFn`](update_transformation.md#ForwardFn), [`add_state_update`](module.md#Module.add_state_update), [`register_per_param_settings`](utils.md#register_per_param_settings), [`match_regex_rules`](utils.md#match_regex_rules), [`param_values`](update_transformation.md#Updates.param_values), [`state`](module.md#Module.state), [`BaseLearner`](learner.md#BaseLearner), [`mask`](update_transformation.md#Updates.mask), [`ema`](learner.md#Learner.Config.ema), [`ema`](learner.md#CompositeLearner.Config.ema), [`BackwardOutputs`](update_transformation.md#BackwardOutputs), [`Config`](learner.md#CompositeLearner.Config), [`mask_tree`](update_transformation.md#mask_tree), [`learners`](learner.md#CompositeLearner.Config.learners), [`ForwardBackwardOutputs`](update_transformation.md#ForwardBackwardOutputs), [`rules`](learner.md#CompositeLearner.Config.rules), [`forward_pass`](update_transformation.md#Updates.forward_pass), [`outputs`](update_transformation.md#ForwardPass.outputs), [`backward_outputs`](update_transformation.md#ForwardBackwardOutputs.backward_outputs), [`forward_outputs`](update_transformation.md#ForwardBackwardOutputs.forward_outputs)  (+1 more)
- used by: [`forward_and_backward`](learner.md#BaseLearner.forward_and_backward), [`create_state_partition_specs`](learner_base.md#LearnerModule.create_state_partition_specs), [`init`](learner_base.md#LearnerModule.init), [`update`](learner.md#BaseLearner.update), [`BaseLearner`](learner.md#BaseLearner), [`should_update_with_optimizers`](learner.md#BaseLearner.should_update_with_optimizers)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/learner.py:394`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L394)
- doc: Configures CompositeLearner.
- signature: `class Config(BaseLearner.Config):`
- members:
  - `ema` — [`L170`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L170)
  - `ema` — [`L410`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L410)
  - `enable_per_variable_summaries` — [`L174`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L174)
  - `forward_fn_transformation` — [`L180`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L180)
  - `learners` — [`L400`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L400)
  - `optimizer` — [`L152`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L152)
  - `rules` — [`L406`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L406)
  - `update_rules` — [`L162`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L162)
- uses (calls/refs, reference-scoped): [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`InstantiableConfig`](config.md#InstantiableConfig), [`config_for_function`](config.md#config_for_function), [`ConfigOr`](config.md#ConfigOr), [`Config`](module.md#Module.Config), [`param_ema`](optimizers.md#param_ema), [`BaseLearner`](learner.md#BaseLearner), [`Config`](learner.md#Learner.Config), [`Learner`](learner.md#Learner), [`UpdateType`](learner.md#UpdateType), [`ForwardFnTransformation`](learner.md#ForwardFnTransformation)
- used by: [`__init__`](learner.md#CompositeLearner.__init__), [`__init__`](learner.md#Learner.__init__), [`ema`](learner.md#Learner.ema), [`ema`](learner.md#CompositeLearner.ema), [`optimizer`](learner.md#Learner.optimizer), [`_forward_fn_transformation`](learner.md#Learner._forward_fn_transformation)

### `Learner`  ·  implements/extends BaseLearner
- def: [`axlearn/common/learner.py:145`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L145)
- doc: The learner module.
- signature: `class Learner(BaseLearner):`
- members:
  - `create_state_partition_specs(self, model_param_specs: Nested[ParameterSpec])` — [`L199`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L199)
  - `forward_and_backward(self, *, fn: ForwardFn, inputs: Nested[Tensor], opt_params: Nested[OptParam])` — [`L339`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L339)
  - `init(self, model_params: Nested[OptParam])` — [`L213`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L213)
  - `should_update_with_optimizers(self, model_params: Nested[OptParam])` — [`L234`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L234) — Returns whether each parameter should be updated with the optimizers.
  - `update(self, updates: Updates)` — [`L245`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L245) — Computes `model_params` updates from `update`.
  - `ema` — [`L193`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L193)
  - `optimizer` — [`L191`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L191)
- protocol/private: `__init__`[`L182`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L182), `_compute_updated_params`[`L266`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L266), `_forward_fn_transformation`[`L195`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L195), `_get_optimizer_model_params`[`L208`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L208), `_update_types`[`L225`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L225)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`default_config`](config.md#Configurable.default_config), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`ParameterSpec`](base_layer.md#ParameterSpec), [`flatten_items`](utils.md#flatten_items), [`PartitionedGradientTransformation`](optimizer_base.md#PartitionedGradientTransformation), [`maybe_instantiate`](config.md#maybe_instantiate), [`vlog`](module.md#Module.vlog), [`Config`](module.md#Module.Config), [`add_summary`](module.md#Module.add_summary), [`__init__`](module.md#Module.__init__), [`path`](module.md#Module.path), [`OptParam`](optimizer_base.md#OptParam), [`_value_and_grad`](learner.md#_value_and_grad), [`init`](optimizer_base.md#PartitionedGradientTransformation.init), [`update`](optimizer_base.md#PartitionedGradientTransformation.update), [`tree_paths`](utils.md#tree_paths), [`instantiate`](config.md#InstantiableConfig.instantiate), [`partition`](optimizer_base.md#PartitionedGradientTransformation.partition), [`Updates`](update_transformation.md#Updates), [`delta_updates`](update_transformation.md#Updates.delta_updates), [`opt_params`](update_transformation.md#Updates.opt_params), [`ForwardFn`](update_transformation.md#ForwardFn), [`UpdateTransformation`](update_transformation.md#UpdateTransformation), [`create_state_partition_specs`](learner_base.md#LearnerModule.create_state_partition_specs), [`init`](learner_base.md#LearnerModule.init), [`add_state_update`](module.md#Module.add_state_update), [`get_recursively`](utils.md#get_recursively), [`prune_empty`](utils.md#prune_empty), [`inplace_updates`](update_transformation.md#Updates.inplace_updates), [`LearnerModule`](learner_base.md#LearnerModule), [`register_per_param_settings`](utils.md#register_per_param_settings), [`match_regex_rules`](utils.md#match_regex_rules), [`WrappedPartitionedGradientTransformation`](update_transformation.md#WrappedPartitionedGradientTransformation), [`state`](module.md#Module.state), [`BaseLearner`](learner.md#BaseLearner), [`Config`](learner.md#Learner.Config)  (+16 more)
- used by: [`test_conditional_update_transformation`](update_transformation_test.md#ConditionalUpdateTransformationTest.test_conditional_update_transformation), [`get_learner_from_su`](update_transformation_test.md#ConditionalUpdateTransformationTest.get_learner_from_su), [`forward_and_backward`](learner.md#BaseLearner.forward_and_backward), [`create_state_partition_specs`](learner_base.md#LearnerModule.create_state_partition_specs), [`init`](learner_base.md#LearnerModule.init), [`update`](learner.md#BaseLearner.update), [`learner`](trainer.md#SpmdTrainer.Config.learner), [`BaseLearner`](learner.md#BaseLearner), [`should_update_with_optimizers`](learner.md#BaseLearner.should_update_with_optimizers), [`learners`](learner.md#CompositeLearner.Config.learners)  (1 test-only)

### `UpdateType`  ·  implements/extends Enum
- def: [`axlearn/common/learner.py:62`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L62)
- doc: UpdateType specifies which update types are allowed for the parameter.
- signature: `class UpdateType(enum.Enum):`
- members:
  - `ALL_UPDATES` — [`L86`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L86)
  - `NO_UPDATE` — [`L76`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L76)
  - `OPTIMIZERS` — [`L80`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L80)
  - `STATE_UPDATES` — [`L83`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L83)
- used by: [`_update_types`](learner.md#Learner._update_types), [`should_apply_state_updates`](learner.md#should_apply_state_updates), [`should_update_with_optimizers`](learner.md#should_update_with_optimizers), [`update_rules`](learner.md#Learner.Config.update_rules)

## Functions
- `_apply_updates(base: Nested[Tensor], updates: Nested[Tensor])` — [`L358`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L358) — Applies updates from `updates` to `base` in-place, keeping `updates` unchanged.
- `_as_loss_fn(fun: ForwardFn)` — [`L658`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L658) — Convert a `ForwardFn` to a function with the same signature execpt that it outputs
- `_split_gradients(fun: ForwardFn, *, should_compute_gradients: Nested[bool])` — [`L604`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L604) — Return a function that is the same as `fun` but where the call signature is now
- `_value_and_grad(fun: ForwardFn, *, opt_params: Nested[OptParam], inputs: Nested[Tensor], should_compute_gradients: Optional[Nested[bool]] = None)` — [`L684`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L684) — Computes the value and grad of `fun`.
- `filtered_forward(model_params: Nested[Tensor], *, inputs: Any)` — [`L630`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L630)
- `forward(model_params: Nested[Tensor], *, inputs: Any)` — [`L671`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L671)
- `should_apply_state_updates(update_type: UpdateType)` — [`L93`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L93)
- `should_update_with_optimizers(update_type: UpdateType)` — [`L89`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L89)
- `split_params_fn(model_params: Nested)` — [`L641`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L641)
- `split_params_fn(params)` — [`L706`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L706)

## Module values
- `ForwardFnTransformation` — [`L59`](../../../../../../raw/code/axlearn/axlearn/common/learner.py#L59)

