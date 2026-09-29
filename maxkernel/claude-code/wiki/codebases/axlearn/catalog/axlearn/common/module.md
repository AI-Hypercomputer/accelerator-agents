---
title: 'Module: axlearn/common/module.py'
type: catalog
provenance: extracted
module: axlearn/common/module.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.module`/
symbols:
  Module: Module#
  functional: functional().
  Module._add_child: Module#_add_child().
  child_context: child_context().
  Module.vlog: Module#vlog().
  Module.get_shared_module: Module#get_shared_module().
  Module.Config: Module#Config#
  Module.add_summary: Module#add_summary().
  new_output_collection: new_output_collection().
  InvocationContext.add_child: InvocationContext#add_child().
  Module.__init__: Module#__init__().
  Module.path: Module#path().
  OutputCollection.summaries: OutputCollection#summaries.
  InvocationContext.add_summary: InvocationContext#add_summary().
  nowrap: nowrap().
  _call_method_in_context: _call_method_in_context().
  current_context: current_context().
  Module.children: Module#children().
  set_current_context: set_current_context().
  InvocationContext.output_collection: InvocationContext#output_collection.
  Module.get_invocation_context: Module#get_invocation_context().
  InvocationContext: InvocationContext#
  OutputCollection.add_child: OutputCollection#add_child().
  _Functional.__call__: _Functional#__call__().
  InvocationContext.module: InvocationContext#module.
  OutputCollection.state_updates: OutputCollection#state_updates.
  _global_context_stack: _global_context_stack.
  scan_in_context.scan_fn: scan_in_context().scan_fn().
  propagate_repeated_output_collections: propagate_repeated_output_collections().
  Module.prng_key: Module#prng_key().
  Module.add_state_update: Module#add_state_update().
  OutputCollection.module_outputs: OutputCollection#module_outputs.
  Module.path_to_descendant_module: Module#path_to_descendant_module().
  scan_in_context: scan_in_context().
  install_context_stack: install_context_stack().
  InvocationContext.path: InvocationContext#path().
  Module.parent: Module#parent().
  Module.context_shares_module: Module#context_shares_module().
  OutputCollection: OutputCollection#
  OutputCollection.update: OutputCollection#update().
  Module.Config.name: Module#Config#name.
  Module._wrap_methods_with_auto_child_context: Module#_wrap_methods_with_auto_child_context().
  Module.state: Module#state().
  InvocationContext.state: InvocationContext#state.
  ContextStack.stack: ContextStack#stack.
  clone_context_stack: clone_context_stack().
  InvocationContext._get_child_state: InvocationContext#_get_child_state().
  InvocationContext.add_module_output: InvocationContext#add_module_output().
  InvocationContext.set_state_update: InvocationContext#set_state_update().
  Module.is_training: Module#is_training().
  Module._children: Module#_children.
  Module._share_with_descendants: Module#_share_with_descendants().
  Module.add_module_output: Module#add_module_output().
  InvocationContext.parent: InvocationContext#parent.
  InvocationContext.add_state_update: InvocationContext#add_state_update().
  _wrap_method_with_auto_child_context: _wrap_method_with_auto_child_context().
  Module._methods_to_wrap_for_auto_child_context: Module#_methods_to_wrap_for_auto_child_context().
  Module.name: Module#name().
  InvalidDescendantError: InvalidDescendantError#
  InvocationContext.validate: InvocationContext#validate().
  InvocationContext.get_summaries: InvocationContext#get_summaries().
  InvocationContext.get_state_updates: InvocationContext#get_state_updates().
  InvocationContext.get_module_outputs: InvocationContext#get_module_outputs().
  Module._wrapped_methods_for_auto_child_context: Module#_wrapped_methods_for_auto_child_context().
  InvocationContext.prng_key: InvocationContext#prng_key.
  Module._shared_module_names: Module#_shared_module_names.
  Module.vlog_is_on: Module#vlog_is_on().
  _Functional.context: _Functional#context.
  InvocationContext.name: InvocationContext#name.
  Module._paths_to_shared_modules: Module#_paths_to_shared_modules.
  Module.vprint: Module#vprint().
  Module.get_module_outputs: Module#get_module_outputs().
  Module._vlog_level: Module#_vlog_level.
  Module.SharedModuleInfo.module: Module#SharedModuleInfo#module.
  ContextStack.thread_id: ContextStack#thread_id.
  _Functional.method_fn: _Functional#method_fn.
  propagate_repeated_output_collections._accumulate: propagate_repeated_output_collections()._accumulate().
  _wrap_method_with_auto_child_context.wrap_method_fn: _wrap_method_with_auto_child_context().wrap_method_fn().
  Module._name: Module#_name.
  Module._should_wrap_method: Module#_should_wrap_method().
  Module.__getattr__: Module#__getattr__().
  Module.SharedModuleInfo.state: Module#SharedModuleInfo#state.
  Module.Config.vlog: Module#Config#vlog.
  _Functional: _Functional#
  OutputConflictError.__init__: OutputConflictError#__init__().
  OutputCollection.__contains__: OutputCollection#__contains__().
  Summable.__add__: Summable#__add__().
  InvocationContext.functional: InvocationContext#functional().
  Module.__post_init__: Module#__post_init__().
  Module.__repr__: Module#__repr__().
  Module.__call__: Module#__call__().
  _CallableT: _CallableT.
  OutputConflictError: OutputConflictError#
  T: T.
  InvocationContext.is_training: InvocationContext#is_training.
  Module.SharedModuleInfo: Module#SharedModuleInfo#
  _generate_seed_from_name: _generate_seed_from_name().
  _HOW_TO_CALL_MODULE_MULTIPLE_TIMES: _HOW_TO_CALL_MODULE_MULTIPLE_TIMES.
  ChildNameConflictError: ChildNameConflictError#
  Summable: Summable#
  ContextStack: ContextStack#
  _PostInitMeta: _PostInitMeta#
  Module._parent: Module#_parent.
  Module.SharedModuleInfo.name: Module#SharedModuleInfo#name.
  _Functional.require_parent: _Functional#require_parent.
  _Functional.copy_args_tree: _Functional#copy_args_tree.
  HF_MODULE_KEY: HF_MODULE_KEY.
  _PostInitMeta.__call__: _PostInitMeta#__call__().
  Module.__str__: Module#__str__().
---
# Module: [`axlearn/common/module.py`](../../../../../../raw/code/axlearn/axlearn/common/module.py)

## Classes
### `ChildNameConflictError`  ·  implements/extends ValueError
- def: [`axlearn/common/module.py:243`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L243)
- signature: `class ChildNameConflictError(ValueError):`
- used by: [`_add_child`](module.md#Module._add_child)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/module.py:776`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L776)
- doc: Module config.
- signature: `class Config(Configurable.Config):`
- members:
  - `name` — [`L784`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L784)
  - `vlog` — [`L785`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L785)
- uses (calls/refs, reference-scoped): [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`Configurable`](config.md#Configurable), [`Config`](config.md#Configurable.Config)
- used by: [`_add_child`](module.md#Module._add_child), [`Config`](base_layer.md#BaseLayer.Config), [`__call__`](state_builder.md#OrbaxStateBuilder.__call__), `test_build_remat_spec_neuron`, [`__init__`](module.md#Module.__init__), `test_build_remat_spec`, [`__init__`](learner.md#Learner.__init__), [`__init__`](update_transformation.md#ConditionalUpdateTransformation.__init__), [`Config`](state_builder.md#Builder.Config), [`Config`](trainer.md#SpmdTrainer.Config), [`Config`](input_base.md#Input.Config), [`Config`](evaler.md#BaseMetricCalculator.Config), [`Config`](summary_writer.md#BaseWriter.Config), [`Config`](state_builder.md#Converter.Config), [`_wrap_methods_with_auto_child_context`](base_layer.md#BaseLayer._wrap_methods_with_auto_child_context), [`tensor_stats`](base_layer.md#DefaultTensorStats.Config.tensor_stats), [`_maybe_wrap_with_remat`](base_layer.md#BaseLayer._maybe_wrap_with_remat), [`sink`](inference_output.md#OutputRecordWriter.Config.sink), [`Config`](checkpointer.md#BaseCheckpointer.Config), [`Config`](evaler.md#SpmdEvaler.Config), [`Config`](input_dispatch.md#BaseInputDispatcher.Config), [`Config`](learner.md#Learner.Config), [`Config`](base_layer.md#CompositeTensorStats.Config), [`Config`](inference_output.md#BaseOutputWriter.Config), [`Config`](update_transformation.md#WrappedPartitionedGradientTransformation.Config), [`__init__`](checkpointer.md#BaseCheckpointer.__init__), [`Config`](inference_output.md#JsonlExampleRecordSink.Config), [`Config`](inference_output.md#TfExampleRecordSink.Config), [`Config`](learner.md#CompositeLearner.Config), [`Config`](update_transformation.md#ConditionalUpdateTransformation.Config), [`tensor_stats`](base_layer.md#BaseLayer.Config.tensor_stats), [`tensor_stats`](base_layer.md#CompositeTensorStats.Config.tensor_stats), [`_vlog_level`](module.md#Module._vlog_level), [`_name`](module.md#Module._name)  (1 test-only)

### `ContextStack`  ·  implements/extends local
- def: [`axlearn/common/module.py:555`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L555)
- doc: See `install_context_stack` on how to ensure thread-safety of the global stack.
- signature: `class ContextStack(threading.local):`
- members:
  - `stack` — [`L558`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L558)
  - `thread_id` — [`L559`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L559)
- uses (calls/refs, reference-scoped): [`InvocationContext`](module.md#InvocationContext)
- used by: [`child_context`](module.md#child_context), [`current_context`](module.md#current_context), [`set_current_context`](module.md#set_current_context), [`_global_context_stack`](module.md#_global_context_stack), [`install_context_stack`](module.md#install_context_stack), [`clone_context_stack`](module.md#clone_context_stack)

### `InvalidDescendantError`  ·  implements/extends ValueError
- def: [`axlearn/common/module.py:247`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L247)
- signature: `class InvalidDescendantError(ValueError):`
- used by: [`get_shared_module`](module.md#Module.get_shared_module), [`_call_method_in_context`](module.md#_call_method_in_context), [`path_to_descendant_module`](module.md#Module.path_to_descendant_module)

### `InvocationContext`
- def: [`axlearn/common/module.py:367`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L367)
- doc: The invocation context for `Module.__call__()`.
- signature: `class InvocationContext:`
- members:
  - `add_child(self, name: str, **override_kwargs)` — [`L395`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L395) — Creates a child context with the given `name`.
  - `add_module_output(self, name: str, value: Tensor)` — [`L498`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L498) — Add module output to the output collection.
  - `add_state_update(self, name: str, value: Tensor)` — [`L494`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L494) — Adds a state update to the output collection.
  - `add_summary(self, name: str, value: Nested[Union[Summary, Tensor]])` — [`L474`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L474) — Adds the named value to the `OutputCollection.summaries`.
  - `functional(self, method_fn: Callable)` — [`L535`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L535) — Transforms `method_fn` (with this context) into a pure functional Callable.
  - `get_module_outputs(self)` — [`L532`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L532)
  - `get_state_updates(self)` — [`L529`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L529)
  - `get_summaries(self)` — [`L526`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L526)
  - `path(self)` — [`L389`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L389)
  - `set_state_update(self, value: Any)` — [`L513`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L513) — Sets the state update field of the output collection.
  - `validate(leaf)` — [`L486`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L486)
  - `is_training` — [`L385`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L385)
  - `module` — [`L383`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L383)
  - `name` — [`L381`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L381)
  - `output_collection` — [`L387`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L387)
  - `parent` — [`L382`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L382)
  - `prng_key` — [`L386`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L386)
  - `state` — [`L384`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L384)
- protocol/private: `_get_child_state`[`L468`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L468)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`Summary`](summary.md#Summary), [`path`](module.md#Module.path), [`summaries`](module.md#OutputCollection.summaries), [`children`](module.md#Module.children), [`add_child`](module.md#OutputCollection.add_child), [`state_updates`](module.md#OutputCollection.state_updates), [`module_outputs`](module.md#OutputCollection.module_outputs), [`parent`](module.md#Module.parent), [`OutputCollection`](module.md#OutputCollection), [`validate`](summary.md#Summary.validate), [`name`](module.md#Module.name), [`_Functional`](module.md#_Functional), [`OutputConflictError`](module.md#OutputConflictError), [`_generate_seed_from_name`](module.md#_generate_seed_from_name)
- used by: [`functional`](module.md#functional), [`test_wrapped_partitioned_gradient_transformation`](update_transformation_test.md#UpdateTransformationTest.test_wrapped_partitioned_gradient_transformation), [`_run`](pipeline.md#Pipeline._run), [`child_context`](module.md#child_context), [`get_shared_module`](module.md#Module.get_shared_module), [`_run`](repeat.md#Repeat._run), [`transform_update`](update_transformation.md#ConditionalUpdateTransformation.transform_update), [`add_summary`](module.md#Module.add_summary), [`_watchdog_loop`](trainer.md#SpmdTrainer._watchdog_loop), [`maybe_call_with_remat`](base_layer.md#BaseLayer.maybe_call_with_remat), [`update_fn`](optimizers.md#skip_and_clip_by_global_norm.update_fn), [`_add_tensor_stats`](base_layer.md#BaseLayer._add_tensor_stats), [`_call_method_in_context`](module.md#_call_method_in_context), [`current_context`](module.md#current_context), [`set_current_context`](module.md#set_current_context), `test_learned_positional_embedding_1d`, [`get_invocation_context`](module.md#Module.get_invocation_context), [`transform_update`](update_transformation.md#WrappedPartitionedGradientTransformation.transform_update), [`_weight_decay_scales`](optimizers.md#_weight_decay_scales), [`add_stats`](base_layer.md#CompositeTensorStats.add_stats), [`_gc_loop`](checkpointer.md#Checkpointer._gc_loop), [`__call__`](module.md#_Functional.__call__), [`scan_fn`](module.md#scan_in_context.scan_fn), [`_log_per_layer_stats`](optimizers.md#_log_per_layer_stats), [`update_fn`](optimizers.md#scale_update_per_param.update_fn), [`vmap_fn`](pipeline.md#Pipeline.vmap_fn), [`add_state_update`](module.md#Module.add_state_update), [`prng_key`](module.md#Module.prng_key), [`_update2`](optimizers.md#adastar_optimizer.update2_fn._update2), [`scan_in_context`](module.md#scan_in_context), [`install_context_stack`](module.md#install_context_stack), [`context_shares_module`](module.md#Module.context_shares_module), [`state`](module.md#Module.state), [`clone_context_stack`](module.md#clone_context_stack), [`stack`](module.md#ContextStack.stack), [`is_training`](module.md#Module.is_training), [`add_module_output`](module.md#Module.add_module_output), [`_temporary_output_collection`](decoder.md#_temporary_output_collection), [`scale_fn`](optimizers.md#scale_from_learning_rate.scale_fn), [`update_fn`](optimizers.md#clip_by_global_norm.update_fn)  (+3 more; 2 test-only)

### `Module`
- def: [`axlearn/common/module.py:772`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L772) — documented in [axlearn-common-attention](../../../concepts/axlearn-common-attention.md)
- members:
  - `__call__(self, *args, **kwargs)` — [`L1134`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1134) — A shortcut for self.forward(*args, **kwargs).
  - `_add_child(self, name: str, child_config: Module.Config, **kwargs)` — [`L935`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L935) — Adds a child module. — documented in [axlearn-common-attention](../../../concepts/axlearn-common-attention.md)
  - `_methods_to_wrap_for_auto_child_context(self)` — [`L849`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L849) — Returns methods to be wrapped in `_wrapped_methods_for_auto_child_context`.
  - `_share_with_descendants(self, module: Module, *, shared_module_name: str)` — [`L995`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L995) — Share `module` with self's descendant modules.
  - `_wrap_methods_with_auto_child_context(self, methods: dict[str, Callable])` — [`L821`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L821) — Wrap each method in `methods` with an auto child context.
  - `_wrapped_methods_for_auto_child_context(self)` — [`L804`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L804) — Returns methods that have been wrapped and bound to `self`.
  - `add_module_output(self, name: str, value: Tensor)` — [`L1127`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1127)
  - `add_state_update(self, name: str, value: Tensor)` — [`L1123`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1123) — Adds a state update to the output collection of the current context.
  - `add_summary(self, name: str, value: Union[Summable, Tensor, Summary])` — [`L1114`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1114) — Adds the named value to `OutputCollection.summaries`.
  - `children(self)` — [`L1099`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1099)
  - `context_shares_module(ctx: InvocationContext)` — [`L1043`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1043)
  - `get_invocation_context(self)` — [`L1084`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1084)
  - `get_module_outputs(self)` — [`L1130`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1130)
  - `get_shared_module(self, shared_module_or_name: Union[Module, str])` — [`L1026`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1026) — Gets the shared module and state with the given name from a nearest ancestor.
  - `is_training(self)` — [`L1103`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1103)
  - `name(self)` — [`L895`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L895)
  - `parent(self)` — [`L891`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L891)
  - `path(self)` — [`L898`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L898)
  - `path_to_descendant_module(self, module: Module)` — [`L972`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L972) — Returns the relative path from `self` to `module`.
  - `prng_key(self)` — [`L1107`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1107)
  - `state(self)` — [`L1111`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1111)
  - `vlog(self, level: int, msg: str, *args, **kwargs)` — [`L912`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L912)
  - `vlog_is_on(self, level: int)` — [`L909`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L909)
  - `vprint(self, level: int, msg: str, *args, **kwargs)` — [`L916`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L916) — Prints debug info with if level <= config.vlog.
- protocol/private: `__getattr__`[`L878`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L878), `__init__`[`L787`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L787), `__post_init__`[`L799`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L799), `__repr__`[`L906`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L906), `__str__`[`L903`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L903), `_children`[`L792`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L792), `_name`[`L790`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L790), `_parent`[`L791`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L791), `_paths_to_shared_modules`[`L794`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L794), `_shared_module_names`[`L796`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L796), `_should_wrap_method`[`L858`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L858), `_vlog_level`[`L797`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L797)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`BaseLayer`](base_layer.md#BaseLayer), [`instantiate`](config.md#Configurable.Config.instantiate), [`Builder`](state_builder.md#Builder), [`Configurable`](config.md#Configurable), [`Summary`](summary.md#Summary), [`Config`](module.md#Module.Config), [`add_summary`](module.md#InvocationContext.add_summary), [`BaseWriter`](summary_writer.md#BaseWriter), [`current_context`](module.md#current_context), [`Input`](input_base.md#Input), [`InvocationContext`](module.md#InvocationContext), [`Converter`](state_builder.md#Converter), [`module`](module.md#InvocationContext.module), [`BaseInputDispatcher`](input_dispatch.md#BaseInputDispatcher), [`BaseMetricCalculator`](evaler.md#BaseMetricCalculator), [`TensorStats`](base_layer.md#TensorStats), [`BaseCheckpointer`](checkpointer.md#BaseCheckpointer), [`LearnerModule`](learner_base.md#LearnerModule), [`RequiredFieldValue`](config.md#RequiredFieldValue), [`_wrap_methods_with_auto_child_context`](base_layer.md#BaseLayer._wrap_methods_with_auto_child_context), [`name`](module.md#Module.Config.name), [`SpmdTrainer`](trainer.md#SpmdTrainer), [`state`](module.md#InvocationContext.state), [`__init__`](config.md#Configurable.__init__), [`add_module_output`](module.md#InvocationContext.add_module_output), [`BaseRecordSink`](inference_output.md#BaseRecordSink), [`no_stack_summary`](traceback_util.md#no_stack_summary), [`BaseOutputWriter`](inference_output.md#BaseOutputWriter), [`_wrap_method_with_auto_child_context`](module.md#_wrap_method_with_auto_child_context), [`add_state_update`](module.md#InvocationContext.add_state_update), [`parent`](module.md#InvocationContext.parent), [`_methods_to_wrap_for_auto_child_context`](base_layer.md#BaseLayer._methods_to_wrap_for_auto_child_context), [`InvalidDescendantError`](module.md#InvalidDescendantError), [`get_module_outputs`](module.md#InvocationContext.get_module_outputs), [`prng_key`](module.md#InvocationContext.prng_key), [`module`](module.md#Module.SharedModuleInfo.module), [`partial_with_fn_metadata`](utils.md#partial_with_fn_metadata), [`state`](module.md#Module.SharedModuleInfo.state)  (+7 more)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), [`functional`](module.md#functional), [`Builder`](state_builder.md#Builder), [`initialize_parameters_recursively`](base_layer.md#BaseLayer.initialize_parameters_recursively), [`_forward_for_mode`](attention.md#MultiheadAttention._forward_for_mode), [`_compute_attention`](flash_attention/layer.md#FlashAttention._compute_attention), [`_mask_tpl`](rattention/rattention.md#RAttention._mask_tpl), [`_mask_tpl`](attention.md#MultiheadAttention._mask_tpl), [`_run`](pipeline.md#Pipeline._run), [`run`](trainer.md#SpmdTrainer.run), [`Config`](base_layer.md#BaseLayer.Config), [`_init_with_prebuilt_state`](trainer.md#SpmdTrainer._init_with_prebuilt_state), [`__init__`](base_layer.md#BaseLayer.__init__), [`__init__`](attention.md#QKVLinear.__init__), [`_train_step`](trainer.md#SpmdTrainer._train_step), [`_compute_updated_params`](learner.md#Learner._compute_updated_params), [`create_parameter_specs_recursively`](base_layer.md#BaseLayer.create_parameter_specs_recursively), [`__init__`](attention.md#TransformerFeedForwardLayer.__init__), [`_forward_for_mode`](attention.md#TransformerAttentionLayer._forward_for_mode), [`_forward_for_mode`](attention.md#TransformerLayer._forward_for_mode), [`eval_step`](evaler.md#SpmdEvaler.eval_step), [`_compute_attention`](attention.md#MultiheadAttention._compute_attention), [`parameters`](base_layer.md#BaseLayer.parameters), [`__init__`](attention.md#RoFormerQKVLinear.__init__), [`__init__`](decoder.md#Decoder.__init__), [`_run`](repeat.md#Repeat._run), [`__init__`](attention.md#FusedGroupedQKVLinear.__init__), [`__init__`](attention.md#MultiheadAttentionXL.__init__), [`transform_update`](update_transformation.md#ConditionalUpdateTransformation.transform_update), [`compile_train_step`](trainer.md#SpmdTrainer.compile_train_step), [`_watchdog_loop`](trainer.md#SpmdTrainer._watchdog_loop), [`add_child`](module.md#InvocationContext.add_child), [`restore_checkpoint`](trainer.md#SpmdTrainer.restore_checkpoint), [`__init__`](attention.md#FusedQKVLinear.__init__), [`forward`](attention.md#_TransformerPipeline.forward), [`_dot_general_maybe_quantized`](quantized_dot_general/layers.md#QuantizedDotGeneral._dot_general_maybe_quantized), [`_step_log`](trainer.md#SpmdTrainer._step_log), [`init`](trainer.md#SpmdTrainer.init), [`maybe_call_with_remat`](base_layer.md#BaseLayer.maybe_call_with_remat), [`__init__`](attention.md#PipelinedTransformerLayer.__init__)  (+165 more; 2 test-only)

### `OutputCollection`  ·  implements/extends NamedTuple
- def: [`axlearn/common/module.py:251`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L251)
- doc: Implicit outputs from module invocations.
- signature: `class OutputCollection(NamedTuple):`
- members:
  - `add_child(self, name: str)` — [`L265`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L265)
  - `update(self, collection: OutputCollection)` — [`L276`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L276)
  - `module_outputs` — [`L260`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L260)
  - `state_updates` — [`L259`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L259)
  - `summaries` — [`L258`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L258)
- protocol/private: `__contains__`[`L262`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L262)
- uses (calls/refs, reference-scoped): [`NestedTensor`](utils.md#NestedTensor), [`new_output_collection`](module.md#new_output_collection), [`OutputConflictError`](module.md#OutputConflictError)
- used by: [`functional`](module.md#functional), [`test_wrapped_partitioned_gradient_transformation`](update_transformation_test.md#UpdateTransformationTest.test_wrapped_partitioned_gradient_transformation), [`_run`](pipeline.md#Pipeline._run), [`_train_step`](trainer.md#SpmdTrainer._train_step), [`test_conditional_update_transformation`](update_transformation_test.md#ConditionalUpdateTransformationTest.test_conditional_update_transformation), [`_run`](repeat.md#Repeat._run), [`add_child`](module.md#InvocationContext.add_child), [`new_output_collection`](module.md#new_output_collection), [`add_summary`](module.md#InvocationContext.add_summary), [`maybe_call_with_remat`](base_layer.md#BaseLayer.maybe_call_with_remat), [`_add_tensor_stats`](base_layer.md#BaseLayer._add_tensor_stats), [`output_collection`](module.md#InvocationContext.output_collection), [`add_stats`](base_layer.md#CompositeTensorStats.add_stats), `test_add_value_rms_norm_summary`, [`scan_body`](gradient_accumulation.md#with_minibatch_steps.decorator.fwd_helper.scan_body), [`__call__`](module.md#_Functional.__call__), `test_add_tensor_stats`, [`_call_model`](evaler.md#BaseMetricCalculator._call_model), [`scan_fn`](module.md#scan_in_context.scan_fn), [`_forward_in_pjit`](evaler.md#ModelSummaryAccumulator._forward_in_pjit), [`_predict_in_pjit`](evaler.md#GlobalMetricCalculator._predict_in_pjit), [`propagate_repeated_output_collections`](module.md#propagate_repeated_output_collections), `test_add_dead_neuron_summary`, [`add_module_output`](module.md#InvocationContext.add_module_output), [`set_state_update`](module.md#InvocationContext.set_state_update), [`add_state_update`](module.md#InvocationContext.add_state_update), [`output_collection`](update_transformation.md#ForwardOutputs.output_collection), [`get_module_outputs`](module.md#InvocationContext.get_module_outputs), [`get_state_updates`](module.md#InvocationContext.get_state_updates), [`get_summaries`](module.md#InvocationContext.get_summaries)

### `OutputConflictError`  ·  implements/extends ValueError
- def: [`axlearn/common/module.py:238`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L238)
- signature: `class OutputConflictError(ValueError):`
- protocol/private: `__init__`[`L239`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L239)
- uses (calls/refs, reference-scoped): [`_HOW_TO_CALL_MODULE_MULTIPLE_TIMES`](module.md#_HOW_TO_CALL_MODULE_MULTIPLE_TIMES)
- used by: [`add_child`](module.md#OutputCollection.add_child), [`add_module_output`](module.md#InvocationContext.add_module_output)

### `SharedModuleInfo`  ·  implements/extends NamedTuple
- def: [`axlearn/common/module.py:1021`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1021)
- signature: `class SharedModuleInfo(NamedTuple):`
- members:
  - `module` — [`L1023`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1023)
  - `name` — [`L1022`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1022)
  - `state` — [`L1024`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1024)
- uses (calls/refs, reference-scoped): [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module)
- used by: [`get_shared_module`](module.md#Module.get_shared_module), [`_redirect`](layers.md#RedirectToSharedModule._redirect)

### `Summable`  ·  implements/extends Protocol
- def: [`axlearn/common/module.py:359`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L359)
- signature: `class Summable(Protocol):`
- protocol/private: `__add__`[`L361`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L361)
- uses (calls/refs, reference-scoped): [`T`](module.md#T)
- used by: [`add_summary`](module.md#Module.add_summary)

### `_Functional`
- def: [`axlearn/common/module.py:1152`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1152)
- members:
  - `__call__(self, *args, **kwargs)` — [`L1167`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1167) — Invokes method_fn in a pure functional fashion.
  - `context` — [`L1159`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1159)
  - `copy_args_tree` — [`L1165`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1165)
  - `method_fn` — [`L1156`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1156)
  - `require_parent` — [`L1163`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1163)
- uses (calls/refs, reference-scoped): [`set_current_context`](module.md#set_current_context), [`output_collection`](module.md#InvocationContext.output_collection), [`InvocationContext`](module.md#InvocationContext), [`OutputCollection`](module.md#OutputCollection), [`raise_for_cycles`](utils.md#raise_for_cycles)
- used by: [`functional`](module.md#functional), [`functional`](module.md#InvocationContext.functional)

### `_PostInitMeta`  ·  implements/extends type
- def: [`axlearn/common/module.py:718`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L718)
- doc: A metaclass that invokes `__post_init__`.
- signature: `class _PostInitMeta(type):`
- protocol/private: `__call__`[`L721`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L721)
- used by: [`Module`](module.md#Module)

## Functions
- `_accumulate(carry, iter_x)` — [`L334`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L334)
- `_call_method_in_context(module: Module, *args, method_fn: Callable, method_name: str, **kwargs)` — [`L651`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L651) — Call the given method within the invocation context corresponding to `module` and passing
- `_generate_seed_from_name(name: str)` — [`L213`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L213) — Generates a random seed from a name string.
- `_wrap_method_with_auto_child_context(*, method_fn: Callable, method_name: str)` — [`L729`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L729) — Wraps a method by proxying through `_call_method_in_context`.
- `child_context(name: str, **kwargs)` — [`L637`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L637)
- `clone_context_stack()` — [`L565`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L565) — Returns a copy of the current InvocationContext stack.
- `current_context()` — [`L601`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L601)
- `functional(module: Module, prng_key: Optional[Tensor], state: NestedTensor, inputs: Union[Sequence[Any], dict[str, Any]], *, method: str = "forward", is_training: bool, drop_output_collections: Sequence[str] = ("module_outputs",), copy_args_tree: bool = True)` — [`L1203`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1203) — Invokes <module>.<method> in a pure functional fashion. — documented in [axlearn-common-trainer](../../../concepts/axlearn-common-trainer.md)
- `install_context_stack(stack: list[InvocationContext])` — [`L574`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L574) — Installs the given context stack.
- `new_output_collection()` — [`L282`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L282)
- `nowrap(fun: _CallableT)` — [`L184`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L184) — Marks the specified module method as one that doesn't need to be wrapped.
- `propagate_repeated_output_collections(repeated_output_collection: OutputCollection, *, child_name_prefix: str, target_output_collection: OutputCollection, merge_summaries: bool = False)` — [`L286`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L286) — Propagates contents from `repeated_output_collection` to `target_target_output_collection`.
- `scan_fn(carry_i: NestedTensor, scan_i: NestedTensor)` — [`L1335`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1335)
- `scan_in_context(fn, *, carry: NestedTensor, xs: NestedTensor, drop_output: Optional[Callable[[str], bool]] = None, child_name_prefix: str = "iter", unroll: Union[int, bool] = 1, remat_kwargs: Optional[dict[str, Any]] = None, merge_summaries: bool = False)` — [`L1270`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L1270) — A thin wrapper around `jax.lax.scan` which is compatible with `OutputCollection`.
- `set_current_context(context: InvocationContext, *, require_parent: bool = True)` — [`L608`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L608)
- `wrap_method_fn(self, *args, **kwargs)` — [`L748`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L748)

## Module values
- `HF_MODULE_KEY` — [`L181`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L181)
- `T` — [`L355`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L355)
- `_CallableT` — [`L179`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L179)
- `_HOW_TO_CALL_MODULE_MULTIPLE_TIMES` — [`L230`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L230)
- `_global_context_stack` — [`L562`](../../../../../../raw/code/axlearn/axlearn/common/module.py#L562)

