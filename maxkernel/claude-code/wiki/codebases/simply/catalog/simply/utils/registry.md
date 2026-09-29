---
title: 'Module: simply/utils/registry.py'
type: catalog
provenance: extracted
module: simply/utils/registry.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.registry`/
symbols:
  RootRegistry.register: RootRegistry#register().
  RootRegistry: RootRegistry#
  RootRegistry.get: RootRegistry#get().
  RootRegistry.get_instance: RootRegistry#get_instance().
  FunctionRegistry: FunctionRegistry#
  RootRegistry.keys: RootRegistry#keys().
  RootRegistry.fullname: RootRegistry#fullname().
  RootRegistry.registry: RootRegistry#registry.
  RootRegistry.register_value: RootRegistry#register_value().
  RootRegistry.unregister: RootRegistry#unregister().
  RootRegistry._get_namespace_prefix: RootRegistry#_get_namespace_prefix().
  RootRegistry.reset: RootRegistry#reset().
  RootRegistry.namespace: RootRegistry#namespace.
  _is_running_in_colab: _is_running_in_colab().
  RootRegistry.OVERWRITE_DUPLICATE: RootRegistry#OVERWRITE_DUPLICATE.
  FunctionRegistry.namespace: FunctionRegistry#namespace.
---
# Module: [`simply/utils/registry.py`](../../../../../../raw/code/simply/simply/utils/registry.py)

## Classes
### `FunctionRegistry`  ·  implements/extends RootRegistry
- def: [`simply/utils/registry.py:99`](../../../../../../raw/code/simply/simply/utils/registry.py#L99)
- doc: A simple registry for name-to-function dictionary.
- signature: `class FunctionRegistry(RootRegistry):`
- members:
  - `namespace` — [`L102`](../../../../../../raw/code/simply/simply/utils/registry.py#L102)
- uses (calls/refs, reference-scoped): [`RootRegistry`](registry.md#RootRegistry)
- used by: [`RootRegistry`](registry.md#RootRegistry), [`moe_ffn`](../model_lib.md#MoEFeedForward.moe_ffn), [`apply`](control_flow.md#ControlFlow.apply), [`apply`](../model_lib.md#FeedForward.apply), [`_apply_dense_moe`](../model_lib.md#MoEFeedForward._apply_dense_moe), [`compute_block`](../model_lib.md#MoEFeedForward.compute_block), [`test_dummy_registry`](registry_test.md#RegistryTest.test_dummy_registry), [`CheckpointFormatRegistry`](checkpoint_lib.md#CheckpointFormatRegistry), [`simple_moe`](../model_lib_test.md#simple_moe), [`gelu`](../model_lib.md#gelu), [`squared_relu`](../model_lib.md#squared_relu), [`_dummy_fn`](registry_test.md#RegistryTest._dummy_fn)

### `RootRegistry`
- def: [`simply/utils/registry.py:25`](../../../../../../raw/code/simply/simply/utils/registry.py#L25) — documented in [simply-utils-module](../../../concepts/simply-utils-module.md)
- doc: A root registry for name-to-function/class dictionary.
- signature: `class RootRegistry:`
- members:
  - `_get_namespace_prefix(cls)` — [`L37`](../../../../../../raw/code/simply/simply/utils/registry.py#L37) — Returns the namespace prefix with a colon, or an empty string.
  - `fullname(cls, name: str)` — [`L56`](../../../../../../raw/code/simply/simply/utils/registry.py#L56)
  - `get(cls, name: str, raise_error: bool = True)` — [`L86`](../../../../../../raw/code/simply/simply/utils/registry.py#L86) — documented in [simply-utils-registry](../../../concepts/simply-utils-registry.md)
  - `get_instance(cls, name: str, raise_error: bool = True)` — [`L95`](../../../../../../raw/code/simply/simply/utils/registry.py#L95) — documented in [simply-utils-registry](../../../concepts/simply-utils-registry.md)
  - `keys(cls)` — [`L42`](../../../../../../raw/code/simply/simply/utils/registry.py#L42) — Returns all keys registered under the class namespace.
  - `register(cls, fn_or_cls: Any, name: str = '')` — [`L60`](../../../../../../raw/code/simply/simply/utils/registry.py#L60) — Registers a function or class. — documented in [simply-utils-module](../../../concepts/simply-utils-module.md)
  - `register_value(cls, value: Any, name: str)` — [`L75`](../../../../../../raw/code/simply/simply/utils/registry.py#L75) — Register a function that returns the given value.
  - `reset(cls)` — [`L52`](../../../../../../raw/code/simply/simply/utils/registry.py#L52)
  - `unregister(cls, name)` — [`L80`](../../../../../../raw/code/simply/simply/utils/registry.py#L80)
  - `OVERWRITE_DUPLICATE` — [`L33`](../../../../../../raw/code/simply/simply/utils/registry.py#L33)
  - `namespace` — [`L34`](../../../../../../raw/code/simply/simply/utils/registry.py#L34)
  - `registry` — [`L32`](../../../../../../raw/code/simply/simply/utils/registry.py#L32)
- uses (calls/refs, reference-scoped): [`ExperimentConfigRegistry`](../config_lib.md#ExperimentConfigRegistry), [`DataSourceRegistry`](../data_lib.md#DataSourceRegistry), [`LMFormatRegistry`](lm_format.md#LMFormatRegistry), [`FunctionRegistry`](registry.md#FunctionRegistry), [`TokenizerRegistry`](tokenization.md#TokenizerRegistry), [`EvaluationRegistry`](evaluation_lib.md#EvaluationRegistry), [`OptimizerRegistry`](optimizers.md#OptimizerRegistry), [`SamplingRegistry`](sampling_lib.md#SamplingRegistry), [`ShardingConfigRegistry`](../config_lib.md#ShardingConfigRegistry), [`LLMRegistry`](../agent/llm.md#LLMRegistry), [`DatasetConfigRegistry`](../data_lib.md#DatasetConfigRegistry), [`EnvRegistry`](../agent/env.md#EnvRegistry), [`TrainLoopRegistry`](../model_lib.md#TrainLoopRegistry), [`InputProcessorRegistry`](sampling_lib.md#InputProcessorRegistry), [`RewardNormalizerRegistry`](../rl_lib.md#RewardNormalizerRegistry), [`ScheduleRegistry`](optimizers.md#ScheduleRegistry), [`DistributionRegistry`](distributions.md#DistributionRegistry), [`DummyClassRegistry`](registry_test.md#DummyClassRegistry), [`PositionEncodingRegistry`](position_encoding.md#PositionEncodingRegistry), [`ToolRegistry`](../tool_lib.md#ToolRegistry), [`_is_running_in_colab`](registry.md#_is_running_in_colab)
- used by: [`run_experiment`](../rl_lib.md#run_experiment), [`main`](../eval/page_decode_eval.md#main), [`main`](../eval/decode_eval.md#main), [`main`](../serving/page_server.md#main), [`ExperimentConfigRegistry`](../config_lib.md#ExperimentConfigRegistry), [`moe_ffn`](../model_lib.md#MoEFeedForward.moe_ffn), [`main`](../serving/vanilla_server.md#main), [`create_train_batch`](../rl_lib.md#create_train_batch), [`main`](../eval/remote_decode_eval.md#main), [`create_iter_dataset`](../data_lib.md#create_iter_dataset), [`load_checkpoint_from_path`](checkpoint_lib.md#load_checkpoint_from_path), [`DatasetConfig`](../data_lib.md#DatasetConfig), [`DataSourceRegistry`](../data_lib.md#DataSourceRegistry), [`load`](pytree.md#load), [`RoPE`](position_encoding.md#RoPE), [`deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl`](../config_lib.md#deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl), [`gemma3_4b_it_simple_qa_number_only_tool_use_rl`](../config_lib.md#gemma3_4b_it_simple_qa_number_only_tool_use_rl), [`TransformerLM`](../model_lib.md#TransformerLM), [`lm_rl_test`](../config_lib.md#lm_rl_test), [`gemma2_2b_gsm8k_0shot_rl`](../config_lib.md#gemma2_2b_gsm8k_0shot_rl), [`apply`](control_flow.md#ControlFlow.apply), [`input_processor`](../serving/page_batcher.md#Batcher.input_processor), [`apply`](../model_lib.md#FeedForward.apply), [`EinsumLinear`](module.md#EinsumLinear), [`_apply_dense_moe`](../model_lib.md#MoEFeedForward._apply_dense_moe), [`compute_block`](../model_lib.md#MoEFeedForward.compute_block), [`input_processor`](../serving/vanilla_server.md#Batcher.input_processor), [`deepseek_qwen2_1p5b`](../config_lib.md#deepseek_qwen2_1p5b), [`test_dump_load_baseline_config`](../config_lib_test.md#ConfigLibTest.test_dump_load_baseline_config), [`update_params_from_checkpoint_path`](../serving/page_batcher.md#Batcher.update_params_from_checkpoint_path), [`LinearWarmupConstant`](optimizers.md#LinearWarmupConstant), [`ModuleRegistry`](module.md#ModuleRegistry), [`register_spm_vocabs`](../data_lib.md#register_spm_vocabs), [`BaseExperimentConfig`](../config_lib.md#BaseExperimentConfig), [`_B`](pytree_test.md#_B), [`Chunk`](sampling_lib.md#Chunk), [`SamplingOutput`](../model_lib.md#SamplingOutput), [`flops6e20_tfm2b_c4_l2048`](../config_lib.md#flops6e20_tfm2b_c4_l2048), [`test_dummy_registry`](registry_test.md#RegistryTest.test_dummy_registry), [`lm_test`](../config_lib.md#lm_test)  (+213 more)

## Functions
- `_is_running_in_colab()` — [`L21`](../../../../../../raw/code/simply/simply/utils/registry.py#L21)

