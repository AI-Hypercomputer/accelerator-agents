---
title: 'Module: axlearn/common/evaler.py'
type: catalog
provenance: extracted
module: axlearn/common/evaler.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.evaler`/
symbols:
  SpmdEvaler.eval_step: SpmdEvaler#eval_step().
  SpmdEvaler.input: SpmdEvaler#input.
  SpmdEvaler.__init__: SpmdEvaler#__init__().
  CompositeMetricCalculator._dependencies: CompositeMetricCalculator#_dependencies().
  GlobalMetricCalculator.get_summaries: GlobalMetricCalculator#get_summaries().
  BaseMetricCalculator._call_model: BaseMetricCalculator#_call_model().
  BaseMetricCalculator: BaseMetricCalculator#
  CompositeMetricCalculator._calculators: CompositeMetricCalculator#_calculators.
  ModelSummaryAccumulator._forward_in_pjit: ModelSummaryAccumulator#_forward_in_pjit().
  GlobalMetricCalculator._predict_in_pjit: GlobalMetricCalculator#_predict_in_pjit().
  GlobalMetricCalculator._compute_metrics_in_pjit: GlobalMetricCalculator#_compute_metrics_in_pjit().
  BaseMetricCalculator.__init__: BaseMetricCalculator#__init__().
  CompositeMetricCalculator.Config.metric_calculators: CompositeMetricCalculator#Config#metric_calculators.
  GlobalMetricCalculator.init_state: GlobalMetricCalculator#init_state().
  BaseMetricCalculator._input_partition_spec: BaseMetricCalculator#_input_partition_spec().
  BaseMetricCalculator.Config: BaseMetricCalculator#Config#
  BaseMetricCalculator.init_state: BaseMetricCalculator#init_state().
  BaseMetricCalculator.get_summaries: BaseMetricCalculator#get_summaries().
  BaseMetricCalculator._dispatch_global_batch: BaseMetricCalculator#_dispatch_global_batch().
  CompositeMetricCalculator._edge_names: CompositeMetricCalculator#_edge_names.
  ModelSummaryAccumulator.__init__: ModelSummaryAccumulator#__init__().
  CompositeMetricCalculator.__init__: CompositeMetricCalculator#__init__().
  CompositeMetricCalculator.forward: CompositeMetricCalculator#forward().
  SpmdEvaler.Config.summary_writer: SpmdEvaler#Config#summary_writer.
  GlobalMetricCalculator.__init__: GlobalMetricCalculator#__init__().
  GlobalMetricCalculator.forward: GlobalMetricCalculator#forward().
  SpmdEvaler.Config.input: SpmdEvaler#Config#input.
  BaseMetricCalculator.forward: BaseMetricCalculator#forward().
  SpmdEvaler.Config.metric_calculator: SpmdEvaler#Config#metric_calculator.
  SpmdEvaler.Config: SpmdEvaler#Config#
  ModelSummaryAccumulator.init_state: ModelSummaryAccumulator#init_state().
  CompositeMetricCalculator.init_state: CompositeMetricCalculator#init_state().
  CompositeMetricCalculator.get_summaries: CompositeMetricCalculator#get_summaries().
  BaseMetricCalculator._eval_cast: BaseMetricCalculator#_eval_cast().
  ModelSummaryAccumulator.Config: ModelSummaryAccumulator#Config#
  SpmdEvaler.Config.eval_policy: SpmdEvaler#Config#eval_policy.
  SpmdEvaler._eval_policy: SpmdEvaler#_eval_policy.
  GlobalMetricCalculator.Config: GlobalMetricCalculator#Config#
  GlobalMetricCalculator._jit_predict: GlobalMetricCalculator#_jit_predict.
  BaseMetricCalculator._pjit: BaseMetricCalculator#_pjit().
  ModelSummaryAccumulator.Config.metric_accumulator: ModelSummaryAccumulator#Config#metric_accumulator.
  ModelSummaryAccumulator.forward: ModelSummaryAccumulator#forward().
  ModelSummaryAccumulator.get_summaries: ModelSummaryAccumulator#get_summaries().
  SpmdEvaler: SpmdEvaler#
  CompositeMetricCalculator.Config: CompositeMetricCalculator#Config#
  SpmdEvaler.Config.output_writer: SpmdEvaler#Config#output_writer.
  GlobalMetricCalculator._calculate_metrics: GlobalMetricCalculator#_calculate_metrics().
  PredictionOutputs: PredictionOutputs#
  ModelSummaryAccumulator._jit_forward: ModelSummaryAccumulator#_jit_forward.
  CompositeMetricCalculator.Config.dependencies: CompositeMetricCalculator#Config#dependencies.
  CompositeMetricCalculator._calculator_dag: CompositeMetricCalculator#_calculator_dag.
  every_n_steps_policy: every_n_steps_policy().
  GlobalMetricCalculator._jit_compute_metrics: GlobalMetricCalculator#_jit_compute_metrics.
  CompositeMetricCalculator: CompositeMetricCalculator#
  GlobalMetricCalculator._metric_accumulator: GlobalMetricCalculator#_metric_accumulator.
  ModelSummaryAccumulator: ModelSummaryAccumulator#
  ModelSummaryAccumulator._per_example_outputs: ModelSummaryAccumulator#_per_example_outputs().
  ModelSummaryAccumulator._process_summaries: ModelSummaryAccumulator#_process_summaries().
  CompositeMetricCalculator.Dependency.dst_key: CompositeMetricCalculator#Dependency#dst_key.
  CompositeMetricCalculator.resolve_name: CompositeMetricCalculator#resolve_name().
  ModelSummaryAccumulator._metric_accumulator: ModelSummaryAccumulator#_metric_accumulator.
  SpmdEvaler.Config.eval_dtype: SpmdEvaler#Config#eval_dtype.
  BaseMetricCalculator.formatted_metric_name: BaseMetricCalculator#formatted_metric_name().
  PredictionOutputs.input_batch: PredictionOutputs#input_batch.
  GlobalMetricCalculator: GlobalMetricCalculator#
  CompositeMetricCalculator.Dependency.src: CompositeMetricCalculator#Dependency#src.
  EvalPolicy: EvalPolicy#
  SpmdEvaler._trace_steps: SpmdEvaler#_trace_steps.
  GlobalMetricCalculator._use_jit_for_metric_calculation: GlobalMetricCalculator#_use_jit_for_metric_calculation.
  BaseMetricCalculator._model: BaseMetricCalculator#_model.
  BaseMetricCalculator._model_param_partition_specs: BaseMetricCalculator#_model_param_partition_specs.
  CompositeMetricCalculator.Dependency: CompositeMetricCalculator#Dependency#
  CompositeMetricCalculator.Dependency.dst: CompositeMetricCalculator#Dependency#dst.
  every_n_steps_policy.fn: every_n_steps_policy().fn().
  BaseMetricCalculator.Config.eval_dtype: BaseMetricCalculator#Config#eval_dtype.
  BaseMetricCalculator.Config.prefix: BaseMetricCalculator#Config#prefix.
  ModelSummaryAccumulator.Config.model_method: ModelSummaryAccumulator#Config#model_method.
  ModelSummaryAccumulator.Config.model_method_kwargs: ModelSummaryAccumulator#Config#model_method_kwargs.
  EvalPolicy.__call__: EvalPolicy#__call__().
  SpmdEvaler.Config.trace_at_iters: SpmdEvaler#Config#trace_at_iters.
  PredictionOutputs.predict_outputs: PredictionOutputs#predict_outputs.
  GlobalMetricCalculator.Config.predict_method: GlobalMetricCalculator#Config#predict_method.
  GlobalMetricCalculator.Config.predict_input_field: GlobalMetricCalculator#Config#predict_input_field.
---
# Module: [`axlearn/common/evaler.py`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py)

## Classes
### `BaseMetricCalculator`  ·  implements/extends Module
- def: [`axlearn/common/evaler.py:45`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L45)
- doc: The base class of classes to calculate evaluation metrics.
- signature: `class BaseMetricCalculator(Module):`
- members:
  - `_call_model(self, *, method: str, prng_key: Tensor, model_params: NestedTensor, input_batch: NestedTensor, **kwargs)` — [`L205`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L205) — Computes self._model.method(input_batch).
  - `_pjit(self, fn: Callable)` — [`L167`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L167) — Compiles `fn` to run on the device mesh.
  - `formatted_metric_name(self, metric_name)` — [`L261`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L261) — Prepend the prefix to the metric_name.
  - `forward(self, input_batch: NestedTensor, *, model_params: NestedTensor, state: NestedTensor)` — [`L116`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L116) — Handles an input batch.
  - `get_summaries(self, *, model_params: NestedTensor, state: NestedTensor, all_forward_outputs: list[NestedTensor])` — [`L146`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L146) — Computes summaries.
  - `init_state(self, *, prng_key: Tensor, model_params: NestedTensor)` — [`L102`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L102) — Initializes the state.
- protocol/private: `__init__`[`L87`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L87), `_dispatch_global_batch`[`L253`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L253), `_eval_cast`[`L201`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L201), `_input_partition_spec`[`L245`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L245), `_model`[`L96`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L96), `_model_param_partition_specs`[`L97`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L97)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`functional`](module.md#functional), [`__init__`](module.md#Module.__init__), [`input`](evaler.md#SpmdEvaler.input), [`dispatch_global_batch`](input_base.md#Input.dispatch_global_batch), [`get_summaries`](evaler.md#GlobalMetricCalculator.get_summaries), [`MetricSummary`](metrics.md#MetricSummary), [`BaseModel`](base_model.md#BaseModel), [`init_state`](evaler.md#GlobalMetricCalculator.init_state), [`cast_floats`](utils.md#cast_floats), [`Config`](evaler.md#BaseMetricCalculator.Config), [`parent`](module.md#Module.parent), [`OutputCollection`](module.md#OutputCollection), [`partition_spec`](input_base.md#Input.partition_spec), [`forward`](evaler.md#CompositeMetricCalculator.forward), [`forward`](evaler.md#GlobalMetricCalculator.forward), [`NestedPartitionSpec`](utils.md#NestedPartitionSpec), [`input_partition_spec`](utils.md#input_partition_spec), [`get_summaries`](evaler.md#CompositeMetricCalculator.get_summaries), [`init_state`](evaler.md#CompositeMetricCalculator.init_state), [`init_state`](evaler.md#ModelSummaryAccumulator.init_state), [`SpmdEvaler`](evaler.md#SpmdEvaler), [`forward`](evaler.md#ModelSummaryAccumulator.forward), [`get_summaries`](evaler.md#ModelSummaryAccumulator.get_summaries), [`CompositeMetricCalculator`](evaler.md#CompositeMetricCalculator), [`ModelSummaryAccumulator`](evaler.md#ModelSummaryAccumulator), [`GlobalMetricCalculator`](evaler.md#GlobalMetricCalculator)
- used by: [`Module`](module.md#Module), [`_calculators`](evaler.md#CompositeMetricCalculator._calculators), [`_forward_in_pjit`](evaler.md#ModelSummaryAccumulator._forward_in_pjit), [`_predict_in_pjit`](evaler.md#GlobalMetricCalculator._predict_in_pjit), [`metric_calculators`](evaler.md#CompositeMetricCalculator.Config.metric_calculators), [`__init__`](evaler.md#CompositeMetricCalculator.__init__), [`__init__`](evaler.md#GlobalMetricCalculator.__init__), [`__init__`](evaler.md#ModelSummaryAccumulator.__init__), [`forward`](evaler.md#CompositeMetricCalculator.forward), [`metric_calculator`](evaler.md#SpmdEvaler.Config.metric_calculator), [`get_summaries`](evaler.md#CompositeMetricCalculator.get_summaries), [`init_state`](evaler.md#CompositeMetricCalculator.init_state), [`Config`](evaler.md#GlobalMetricCalculator.Config), [`Config`](evaler.md#ModelSummaryAccumulator.Config), [`_jit_predict`](evaler.md#GlobalMetricCalculator._jit_predict), [`Config`](evaler.md#CompositeMetricCalculator.Config), [`_jit_compute_metrics`](evaler.md#GlobalMetricCalculator._jit_compute_metrics), [`_jit_forward`](evaler.md#ModelSummaryAccumulator._jit_forward), [`CompositeMetricCalculator`](evaler.md#CompositeMetricCalculator), [`ModelSummaryAccumulator`](evaler.md#ModelSummaryAccumulator), [`GlobalMetricCalculator`](evaler.md#GlobalMetricCalculator)

### `CompositeMetricCalculator`  ·  implements/extends BaseMetricCalculator
- def: [`axlearn/common/evaler.py:361`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L361)
- doc: Runs multiple metric calculators over evaluation batches.
- signature: `class CompositeMetricCalculator(BaseMetricCalculator):`
- members:
  - `_dependencies(self)` — [`L444`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L444) — Expands regex patterns from `cfg.dependencies` and yields concrete tuples of
  - `forward(self, input_batch: NestedTensor, *, model_params: NestedTensor, state: NestedTensor)` — [`L467`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L467)
  - `get_summaries(self, *, model_params: NestedTensor, state: NestedTensor, all_forward_outputs: list[NestedTensor])` — [`L499`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L499)
  - `init_state(self, *, prng_key: Tensor, model_params: NestedTensor)` — [`L461`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L461)
  - `resolve_name(pattern: str)` — [`L451`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L451)
- protocol/private: `__init__`[`L395`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L395), `_calculator_dag`[`L407`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L407), `_calculators`[`L432`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L432), `_edge_names`[`L409`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L409)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`BaseMetricCalculator`](evaler.md#BaseMetricCalculator), [`MetricSummary`](metrics.md#MetricSummary), [`BaseModel`](base_model.md#BaseModel), [`__init__`](evaler.md#BaseMetricCalculator.__init__), [`metric_calculators`](evaler.md#CompositeMetricCalculator.Config.metric_calculators), [`get_summaries`](evaler.md#BaseMetricCalculator.get_summaries), [`init_state`](evaler.md#BaseMetricCalculator.init_state), [`NestedPartitionSpec`](utils.md#NestedPartitionSpec), [`forward`](evaler.md#BaseMetricCalculator.forward), [`Config`](evaler.md#CompositeMetricCalculator.Config), [`dependencies`](evaler.md#CompositeMetricCalculator.Config.dependencies), [`dst_key`](evaler.md#CompositeMetricCalculator.Dependency.dst_key), [`src`](evaler.md#CompositeMetricCalculator.Dependency.src), [`dst`](evaler.md#CompositeMetricCalculator.Dependency.dst)
- used by: [`BaseMetricCalculator`](evaler.md#BaseMetricCalculator), [`get_summaries`](evaler.md#BaseMetricCalculator.get_summaries), [`init_state`](evaler.md#BaseMetricCalculator.init_state), [`forward`](evaler.md#BaseMetricCalculator.forward), [`dependencies`](evaler.md#CompositeMetricCalculator.Config.dependencies)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/evaler.py:775`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L775)
- signature: `class Config(BaseMetricCalculator.Config):`
- members:
  - `dependencies` — [`L393`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L393)
  - `eval_dtype` — [`L78`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L78)
  - `eval_dtype` — [`L580`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L580)
  - `eval_policy` — [`L573`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L573)
  - `input` — [`L569`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L569)
  - `metric_accumulator` — [`L284`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L284)
  - `metric_calculator` — [`L582`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L582)
  - `metric_calculators` — [`L384`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L384)
  - `model_method` — [`L280`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L280)
  - `model_method_kwargs` — [`L282`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L282)
  - `output_writer` — [`L584`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L584)
  - `predict_input_field` — [`L780`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L780)
  - `predict_method` — [`L777`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L777)
  - `prefix` — [`L85`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L85)
  - `summary_writer` — [`L571`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L571)
  - `trace_at_iters` — [`L577`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L577)
- uses (calls/refs, reference-scoped): [`default_config`](config.md#Configurable.default_config), [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`InstantiableConfig`](config.md#InstantiableConfig), [`config_for_function`](config.md#config_for_function), [`Input`](input_base.md#Input), [`BaseMetricCalculator`](evaler.md#BaseMetricCalculator), [`Config`](input_base.md#Input.Config), [`Config`](evaler.md#BaseMetricCalculator.Config), [`MetricAccumulator`](metrics.md#MetricAccumulator), [`Config`](inference_output.md#BaseOutputWriter.Config), [`SummaryWriter`](summary_writer.md#SummaryWriter), [`BaseOutputWriter`](inference_output.md#BaseOutputWriter), [`every_n_steps_policy`](evaler.md#every_n_steps_policy), [`CompositeMetricCalculator`](evaler.md#CompositeMetricCalculator), [`ModelSummaryAccumulator`](evaler.md#ModelSummaryAccumulator), [`Dependency`](evaler.md#CompositeMetricCalculator.Dependency)
- used by: [`input`](evaler.md#SpmdEvaler.input), [`_evalers`](trainer.md#SpmdTrainer._evalers), [`__init__`](evaler.md#SpmdEvaler.__init__), [`_dependencies`](evaler.md#CompositeMetricCalculator._dependencies), [`_calculators`](evaler.md#CompositeMetricCalculator._calculators), [`_edge_names`](evaler.md#CompositeMetricCalculator._edge_names), [`__init__`](evaler.md#GlobalMetricCalculator.__init__), [`_eval_policy`](evaler.md#SpmdEvaler._eval_policy), [`resolve_name`](evaler.md#CompositeMetricCalculator.resolve_name)

### `Dependency`  ·  implements/extends PyTreeNode
- def: [`axlearn/common/evaler.py:370`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L370)
- signature: `class Dependency(flax_struct.PyTreeNode):`
- members:
  - `dst` — [`L374`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L374)
  - `dst_key` — [`L376`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L376)
  - `src` — [`L372`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L372)
- uses (calls/refs, reference-scoped): [`config_class`](config.md#config_class)
- used by: [`_dependencies`](evaler.md#CompositeMetricCalculator._dependencies), [`dependencies`](evaler.md#CompositeMetricCalculator.Config.dependencies)

### `EvalPolicy`  ·  implements/extends Protocol
- def: [`axlearn/common/evaler.py:524`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L524)
- doc: Decides whether evaler should run eval at the given step.
- signature: `class EvalPolicy(Protocol):`
- members:
  - `__call__(self, *, step: int, train_summaries: dict[str, Any])` — [`L527`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L527) — Implements the policy.
- used by: [`_eval_policy`](evaler.md#SpmdEvaler._eval_policy), [`every_n_steps_policy`](evaler.md#every_n_steps_policy)

### `GlobalMetricCalculator`  ·  implements/extends BaseMetricCalculator
- def: [`axlearn/common/evaler.py:767`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L767)
- doc: A metric calculator for tasks require evaluation on entire datasets.
- signature: `class GlobalMetricCalculator(BaseMetricCalculator):`
- members:
  - `_calculate_metrics(self, outputs: PredictionOutputs)` — [`L862`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L862) — Calculates metrics from ``concatenated_outputs`` of the whole evaluation set.
  - `_compute_metrics_in_pjit(self, model_params: NestedTensor, prng_key: Tensor, outputs: list[PredictionOutputs])` — [`L875`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L875) — Computes metrics and returns them in "replicated".
  - `_predict_in_pjit(self, model_params: NestedTensor, prng_key: Tensor, input_batch: NestedTensor)` — [`L840`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L840) — Core function that calls model's predict() method for each batch and will be pjit-ed.
  - `forward(self, input_batch: NestedTensor, *, model_params: NestedTensor, state: NestedTensor)` — [`L806`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L806) — Calls predict method of the model and returns input_batch and per-batch model outputs.
  - `get_summaries(self, *, model_params: NestedTensor, state: NestedTensor, all_forward_outputs: list[PredictionOutputs])` — [`L906`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L906)
  - `init_state(self, *, prng_key: Tensor, model_params: NestedTensor)` — [`L800`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L800)
- protocol/private: `__init__`[`L782`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L782), `_jit_compute_metrics`[`L797`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L797), `_jit_predict`[`L795`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L795), `_metric_accumulator`[`L798`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L798), `_use_jit_for_metric_calculation`[`L794`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L794)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`default_config`](config.md#Configurable.default_config), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`instantiate`](config.md#Configurable.Config.instantiate), [`WeightedSummary`](metrics.md#WeightedSummary), [`summaries`](module.md#OutputCollection.summaries), [`with_sharding_constraint`](utils.md#with_sharding_constraint), [`_call_model`](evaler.md#BaseMetricCalculator._call_model), [`BaseMetricCalculator`](evaler.md#BaseMetricCalculator), [`BaseModel`](base_model.md#BaseModel), [`__init__`](evaler.md#BaseMetricCalculator.__init__), [`get_recursively`](utils.md#get_recursively), [`MetricAccumulator`](metrics.md#MetricAccumulator), [`NestedPartitionSpec`](utils.md#NestedPartitionSpec), [`input_partition_spec`](utils.md#input_partition_spec), [`update`](metrics.md#MetricAccumulator.update), [`Config`](evaler.md#GlobalMetricCalculator.Config), [`_pjit`](evaler.md#BaseMetricCalculator._pjit), [`PredictionOutputs`](evaler.md#PredictionOutputs), [`replicate_to_local_data`](utils.md#replicate_to_local_data)
- used by: [`BaseMetricCalculator`](evaler.md#BaseMetricCalculator), [`get_summaries`](evaler.md#BaseMetricCalculator.get_summaries), [`init_state`](evaler.md#BaseMetricCalculator.init_state), [`forward`](evaler.md#BaseMetricCalculator.forward)

### `ModelSummaryAccumulator`  ·  implements/extends BaseMetricCalculator
- def: [`axlearn/common/evaler.py:269`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L269)
- doc: Accumulates model summaries over evaluation batches.
- signature: `class ModelSummaryAccumulator(BaseMetricCalculator):`
- members:
  - `_forward_in_pjit(self, model_params: NestedTensor, prng_key: Tensor, input_batch: NestedTensor)` — [`L318`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L318) — Calls `self._model` and returns summaries.
  - `forward(self, input_batch: NestedTensor, *, model_params: NestedTensor, state: NestedTensor)` — [`L305`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L305)
  - `get_summaries(self, *, model_params: NestedTensor, state: NestedTensor, all_forward_outputs: list[NestedTensor])` — [`L351`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L351)
  - `init_state(self, *, prng_key: Tensor, model_params: NestedTensor)` — [`L300`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L300)
- protocol/private: `__init__`[`L286`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L286), `_jit_forward`[`L298`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L298), `_metric_accumulator`[`L297`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L297), `_per_example_outputs`[`L343`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L343), `_process_summaries`[`L348`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L348)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`summaries`](module.md#OutputCollection.summaries), [`_call_model`](evaler.md#BaseMetricCalculator._call_model), [`BaseMetricCalculator`](evaler.md#BaseMetricCalculator), [`MetricSummary`](metrics.md#MetricSummary), [`BaseModel`](base_model.md#BaseModel), [`__init__`](evaler.md#BaseMetricCalculator.__init__), [`NestedPartitionSpec`](utils.md#NestedPartitionSpec), [`Config`](evaler.md#ModelSummaryAccumulator.Config), [`_pjit`](evaler.md#BaseMetricCalculator._pjit)
- used by: [`BaseMetricCalculator`](evaler.md#BaseMetricCalculator), [`get_summaries`](evaler.md#BaseMetricCalculator.get_summaries), [`init_state`](evaler.md#BaseMetricCalculator.init_state), [`forward`](evaler.md#BaseMetricCalculator.forward), [`metric_calculator`](evaler.md#SpmdEvaler.Config.metric_calculator)

### `PredictionOutputs`  ·  implements/extends NamedTuple
- def: [`axlearn/common/evaler.py:762`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L762)
- signature: `class PredictionOutputs(NamedTuple):`
- members:
  - `input_batch` — [`L763`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L763)
  - `predict_outputs` — [`L764`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L764)
- uses (calls/refs, reference-scoped): [`NestedTensor`](utils.md#NestedTensor)
- used by: [`get_summaries`](evaler.md#GlobalMetricCalculator.get_summaries), [`_compute_metrics_in_pjit`](evaler.md#GlobalMetricCalculator._compute_metrics_in_pjit), [`forward`](evaler.md#GlobalMetricCalculator.forward), [`_calculate_metrics`](evaler.md#GlobalMetricCalculator._calculate_metrics)

### `SpmdEvaler`
- def: [`axlearn/common/evaler.py:561`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L561)
- members:
  - `eval_step(self, step: int, *, prng_key: Tensor, model_params: NestedTensor, return_aux: bool = False, train_summaries: Optional[NestedTensor] = None, force_run: bool = False)` — [`L619`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L619) — Runs eval for the given step.
  - `input` — [`L603`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L603)
- protocol/private: `__init__`[`L586`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L586), `_eval_policy`[`L617`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L617), `_trace_steps`[`L616`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L616)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`flatten_items`](utils.md#flatten_items), [`vlog`](module.md#Module.vlog), [`shapes`](utils.md#shapes), [`__init__`](module.md#Module.__init__), [`path`](module.md#Module.path), [`instantiate`](config.md#InstantiableConfig.instantiate), [`maybe_set_config`](config.md#maybe_set_config), [`children`](module.md#Module.children), [`Input`](input_base.md#Input), [`batches`](input_base.md#Input.batches), [`BaseModel`](base_model.md#BaseModel), [`dataset`](input_base.md#Input.dataset), [`partition_spec`](input_base.md#Input.partition_spec), [`summary_writer`](evaler.md#SpmdEvaler.Config.summary_writer), [`NestedPartitionSpec`](utils.md#NestedPartitionSpec), [`input`](evaler.md#SpmdEvaler.Config.input), [`metric_calculator`](evaler.md#SpmdEvaler.Config.metric_calculator), [`Config`](evaler.md#SpmdEvaler.Config), [`eval_policy`](evaler.md#SpmdEvaler.Config.eval_policy), [`host_to_global_device_array`](utils.md#host_to_global_device_array), [`output_writer`](evaler.md#SpmdEvaler.Config.output_writer), [`is_in_elastic_mode`](elastic_input.md#ElasticInput.is_in_elastic_mode), [`ElasticInput`](elastic_input.md#ElasticInput), [`validate_float_dtype`](utils.md#validate_float_dtype), [`eval_dtype`](evaler.md#SpmdEvaler.Config.eval_dtype), [`EvalPolicy`](evaler.md#EvalPolicy)
- used by: [`_input_partition_spec`](evaler.md#BaseMetricCalculator._input_partition_spec), [`_dispatch_global_batch`](evaler.md#BaseMetricCalculator._dispatch_global_batch), [`_should_force_run_evals`](trainer.md#SpmdTrainer._should_force_run_evals), [`evalers`](trainer.md#SpmdTrainer.Config.evalers)

## Functions
- `every_n_steps_policy(n: int = 1, *, min_step: int = 1, max_step: Optional[int] = None)` — [`L541`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L541) — Evals every n steps, but not before `min_step`.
- `fn(*, step: int, train_summaries: dict[str, Any])` — [`L549`](../../../../../../raw/code/axlearn/axlearn/common/evaler.py#L549)

