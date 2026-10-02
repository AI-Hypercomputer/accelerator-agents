---
title: 'Module: axlearn/common/pipeline.py'
type: catalog
provenance: extracted
module: axlearn/common/pipeline.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.pipeline`/
symbols:
  Pipeline._run: Pipeline#_run().
  Pipeline.create_parameter_specs_recursively: Pipeline#create_parameter_specs_recursively().
  Pipeline.initialize_parameters_recursively: Pipeline#initialize_parameters_recursively().
  StreamSchedule.scan: StreamSchedule#scan().
  BaseSchedule: BaseSchedule#
  StreamSchedule.__init__: StreamSchedule#__init__().
  BaseSchedule.num_stages: BaseSchedule#num_stages().
  StreamSchedule.scan_fn: StreamSchedule#scan_fn().
  Pipeline.vmap_fn: Pipeline#vmap_fn().
  GPipeSchedule.scan: GPipeSchedule#scan().
  Pipeline.__init__: Pipeline#__init__().
  BaseSchedule.num_microbatches: BaseSchedule#num_microbatches().
  BaseSchedule.scan: BaseSchedule#scan().
  GPipeSchedule.scan_fn: GPipeSchedule#scan_fn().
  StreamSchedule._init_carry_in: StreamSchedule#_init_carry_in().
  Pipeline.Config: Pipeline#Config#
  Pipeline._schedule: Pipeline#_schedule.
  BaseSchedule.Config: BaseSchedule#Config#
  BaseSchedule.num_iterations: BaseSchedule#num_iterations().
  Pipeline: Pipeline#
  Pipeline.Config.schedule: Pipeline#Config#schedule.
  GPipeSchedule._init_carry_in: GPipeSchedule#_init_carry_in().
  GPipeSchedule._process_carry_in: GPipeSchedule#_process_carry_in().
  Pipeline._to_microbatches: Pipeline#_to_microbatches().
  BaseSchedule._is_valid_stage: BaseSchedule#_is_valid_stage().
  BaseSchedule.Config.num_microbatches: BaseSchedule#Config#num_microbatches.
  Pipeline.Config.num_layers: Pipeline#Config#num_layers.
  StreamSchedule.compute_carry_input: StreamSchedule#compute_carry_input().
  Pipeline.Config.layer: Pipeline#Config#layer.
  BaseSchedule.Config.num_stages: BaseSchedule#Config#num_stages.
  transpose_from_pipeline_stage_outputs: transpose_from_pipeline_stage_outputs().
  _mask_invalid_gradients: _mask_invalid_gradients().
  _shard_pipeline: _shard_pipeline().
  Pipeline.Config.num_microbatches: Pipeline#Config#num_microbatches.
  transpose_to_pipeline_stage_inputs: transpose_to_pipeline_stage_inputs().
  GPipeSchedule.compute_carry_input: GPipeSchedule#compute_carry_input().
  StreamSchedule.reshape_carry: StreamSchedule#reshape_carry().
  Pipeline.transform_factorization_spec: Pipeline#transform_factorization_spec().
  Pipeline.init: Pipeline#init().
  Pipeline.reshape_and_transpose: Pipeline#reshape_and_transpose().
  GPipeSchedule: GPipeSchedule#
  StreamSchedule: StreamSchedule#
  _select_input_or_previous_outputs: _select_input_or_previous_outputs().
  Pipeline.Output.carry: Pipeline#Output#carry.
  Pipeline.Output.ys: Pipeline#Output#ys.
  GPipeSchedule.pad_carry: GPipeSchedule#pad_carry().
  StreamSchedule.update_buffer: StreamSchedule#update_buffer().
  StreamSchedule.rotate_out: StreamSchedule#rotate_out().
  Pipeline._from_microbatches: Pipeline#_from_microbatches().
  Pipeline.transpose_and_reshape: Pipeline#transpose_and_reshape().
  _PerStageFn: _PerStageFn#
  _PerStageFn.__call__: _PerStageFn#__call__().
  Pipeline.Output: Pipeline#Output#
  Pipeline.stack_and_reshape: Pipeline#stack_and_reshape().
---
# Module: [`axlearn/common/pipeline.py`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py)

## Classes
### `BaseSchedule`  ·  implements/extends Configurable
- def: [`axlearn/common/pipeline.py:197`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L197)
- doc: A pipeline schedule.
- signature: `class BaseSchedule(Configurable):`
- members:
  - `_is_valid_stage(self, t: Tensor)` — [`L254`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L254) — Returns a mask indicating whether per-stage values correspond to valid microbatches.
  - `num_iterations(self)` — [`L249`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L249) — Number of microbatch iterations.
  - `num_microbatches(self)` — [`L243`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L243) — Number of microbatches.
  - `num_stages(self)` — [`L237`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L237) — Number of pipeline stages.
  - `scan(self, fn: _PerStageFn, *, carry: Nested[Tensor], state: Nested[Tensor], xs: Nested[Tensor])` — [`L211`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L211) — Implements the pipeline recurrence.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`Configurable`](config.md#Configurable), [`scan`](pipeline.md#StreamSchedule.scan), [`scan`](pipeline.md#GPipeSchedule.scan), [`Config`](pipeline.md#BaseSchedule.Config), [`num_microbatches`](pipeline.md#BaseSchedule.Config.num_microbatches), [`num_stages`](pipeline.md#BaseSchedule.Config.num_stages), [`GPipeSchedule`](pipeline.md#GPipeSchedule), [`StreamSchedule`](pipeline.md#StreamSchedule), [`_PerStageFn`](pipeline.md#_PerStageFn)
- used by: [`_run`](pipeline.md#Pipeline._run), [`Configurable`](config.md#Configurable), `test_stack_vs_pipeline`, [`scan`](pipeline.md#StreamSchedule.scan), [`__init__`](pipeline.md#StreamSchedule.__init__), [`scan_fn`](pipeline.md#StreamSchedule.scan_fn), [`scan`](pipeline.md#GPipeSchedule.scan), [`_init_carry_in`](pipeline.md#StreamSchedule._init_carry_in), [`_schedule`](pipeline.md#Pipeline._schedule), [`scan_fn`](pipeline.md#GPipeSchedule.scan_fn), [`schedule`](pipeline.md#Pipeline.Config.schedule), [`_init_carry_in`](pipeline.md#GPipeSchedule._init_carry_in), [`_process_carry_in`](pipeline.md#GPipeSchedule._process_carry_in), [`GPipeSchedule`](pipeline.md#GPipeSchedule), [`StreamSchedule`](pipeline.md#StreamSchedule)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/pipeline.py:671`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L671)
- doc: Configures Pipeline.
- signature: `class Config(BaseLayer.Config):`
- members:
  - `layer` — [`L674`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L674)
  - `num_layers` — [`L675`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L675)
  - `num_microbatches` — [`L209`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L209)
  - `num_microbatches` — [`L676`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L676)
  - `num_stages` — [`L208`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L208)
  - `schedule` — [`L677`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L677)
- uses (calls/refs, reference-scoped): [`default_config`](config.md#Configurable.default_config), [`BaseLayer`](base_layer.md#BaseLayer), [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`InstantiableConfig`](config.md#InstantiableConfig), [`Config`](base_layer.md#BaseLayer.Config), [`BaseSchedule`](pipeline.md#BaseSchedule), [`Config`](pipeline.md#BaseSchedule.Config), [`GPipeSchedule`](pipeline.md#GPipeSchedule)
- used by: [`_run`](pipeline.md#Pipeline._run), `test_stack_vs_pipeline`, [`create_parameter_specs_recursively`](pipeline.md#Pipeline.create_parameter_specs_recursively), [`initialize_parameters_recursively`](pipeline.md#Pipeline.initialize_parameters_recursively), [`__init__`](pipeline.md#StreamSchedule.__init__), [`num_stages`](pipeline.md#BaseSchedule.num_stages), [`__init__`](pipeline.md#Pipeline.__init__), [`num_microbatches`](pipeline.md#BaseSchedule.num_microbatches), [`_schedule`](pipeline.md#Pipeline._schedule), [`num_iterations`](pipeline.md#BaseSchedule.num_iterations), [`_to_microbatches`](pipeline.md#Pipeline._to_microbatches), [`pipeline`](attention.md#PipelinedTransformerLayer.Config.pipeline), [`reshape_and_transpose`](pipeline.md#Pipeline.reshape_and_transpose)

### `GPipeSchedule`  ·  implements/extends BaseSchedule
- def: [`axlearn/common/pipeline.py:267`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L267)
- doc: A basic schedule as seen in GPipe and GSPMD.
- signature: `class GPipeSchedule(BaseSchedule):`
- members:
  - `_init_carry_in(self, carry: Nested[Tensor])` — [`L357`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L357) — Computes initial loop state.
  - `_process_carry_in(self, carry_in: Nested[Tensor])` — [`L394`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L394) — Computes the vmap input for timestep `t`.
  - `compute_carry_input(v_input_t: Tensor, v_carry_output_t_1: Tensor)` — [`L411`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L411)
  - `pad_carry(v_carry: Tensor)` — [`L373`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L373) — Pads input from [M, microbatch_size, ...] to [M, N, microbatch_size, ...].
  - `scan(self, fn: _PerStageFn, *, carry: Nested[Tensor], state: Nested[Tensor], xs: Nested[Tensor])` — [`L280`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L280) — See `BaseSchedule.scan` for details.
  - `scan_fn(carry_in: Nested[Tensor], xs_t: tuple[Nested[Tensor], Nested[Tensor]])` — [`L295`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L295) — Processes timestep `t` in the pipeline (in parallel across pipeline stages).
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`BaseSchedule`](pipeline.md#BaseSchedule), [`num_stages`](pipeline.md#BaseSchedule.num_stages), [`num_microbatches`](pipeline.md#BaseSchedule.num_microbatches), [`_is_valid_stage`](pipeline.md#BaseSchedule._is_valid_stage), [`_mask_invalid_gradients`](pipeline.md#_mask_invalid_gradients), [`_select_input_or_previous_outputs`](pipeline.md#_select_input_or_previous_outputs), [`_PerStageFn`](pipeline.md#_PerStageFn)
- used by: `test_stack_vs_pipeline`, [`BaseSchedule`](pipeline.md#BaseSchedule), [`scan`](pipeline.md#BaseSchedule.scan), [`schedule`](pipeline.md#Pipeline.Config.schedule)

### `Output`  ·  implements/extends NamedTuple
- def: [`axlearn/common/pipeline.py:730`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L730)
- signature: `class Output(NamedTuple):`
- members:
  - `carry` — [`L731`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L731)
  - `ys` — [`L732`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L732)
- uses (calls/refs, reference-scoped): [`NestedTensor`](utils.md#NestedTensor)
- used by: [`_run`](pipeline.md#Pipeline._run), [`forward`](attention.md#_TransformerPipeline.forward)

### `Pipeline`  ·  implements/extends BaseLayer
- def: [`axlearn/common/pipeline.py:664`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L664)
- doc: A generic pipeline layer.
- signature: `class Pipeline(BaseLayer):`
- members:
  - `_run(self, fn: Callable[[NestedTensor, NestedTensor], NestedTensor], carry: Optional[NestedTensor] = None, *, xs: Optional[NestedTensor] = None, carry_partition_spec: Optional[NestedPartitionSpec] = None, xs_partition_spec: Optional[NestedPartitionSpec] = None, ys_partition_spec: Optional[NestedPartitionSpec] = None)` — [`L734`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L734) — Invokes 'fn' for each sub-layer with inputs already with the microbatch axis.
  - `_to_microbatches(self, inputs)` — [`L881`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L881) — Reshapes inputs from [batch_size, ...] to [M, microbatch_size, ...].
  - `create_parameter_specs_recursively(self)` — [`L688`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L688)
  - `init(prng_key_i)` — [`L720`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L720)
  - `initialize_parameters_recursively(self, prng_key: Union[Tensor, VDict], *, prebuilt: Optional[Nested[Optional[ParameterSpec]]] = None)` — [`L714`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L714)
  - `reshape_and_transpose(x: Tensor)` — [`L885`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L885)
  - `stack_and_reshape(*keys)` — [`L812`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L812)
  - `transform_factorization_spec(spec: Optional[FactorizationSpec])` — [`L692`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L692)
  - `transpose_and_reshape(x: Tensor)` — [`L894`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L894)
  - `vmap_fn(state_n: Tensor, carry_tn: Tensor, prng_key_tn: jax.random.PRNGKey, x_tn: Tensor)` — [`L822`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L822) — See `_PerStageFn` for details.
- protocol/private: `__init__`[`L679`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L679), `_from_microbatches`[`L893`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L893), `_schedule`[`L683`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L683)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`BaseLayer`](base_layer.md#BaseLayer), [`Module`](module.md#Module), [`instantiate`](config.md#Configurable.Config.instantiate), [`_add_child`](module.md#Module._add_child), [`ParameterSpec`](base_layer.md#ParameterSpec), [`__init__`](base_layer.md#BaseLayer.__init__), [`FanAxes`](param_init.md#FanAxes), [`child_context`](module.md#child_context), [`create_parameter_specs_recursively`](base_layer.md#BaseLayer.create_parameter_specs_recursively), [`vlog`](module.md#Module.vlog), [`shapes`](utils.md#shapes), [`new_output_collection`](module.md#new_output_collection), [`summaries`](module.md#OutputCollection.summaries), [`with_sharding_constraint`](utils.md#with_sharding_constraint), [`VDict`](utils.md#VDict), [`FactorizationSpec`](base_layer.md#FactorizationSpec), [`output_collection`](module.md#InvocationContext.output_collection), [`get_invocation_context`](module.md#Module.get_invocation_context), [`split_prng_key`](utils.md#split_prng_key), [`add_child`](module.md#OutputCollection.add_child), [`module`](module.md#InvocationContext.module), [`state_updates`](module.md#OutputCollection.state_updates), [`BaseSchedule`](pipeline.md#BaseSchedule), [`prng_key`](module.md#Module.prng_key), [`axes`](base_layer.md#FactorizationSpec.axes), [`module_outputs`](module.md#OutputCollection.module_outputs), [`Config`](pipeline.md#Pipeline.Config), [`NestedParameterSpec`](base_layer.md#NestedParameterSpec), [`scan`](pipeline.md#BaseSchedule.scan), [`update`](module.md#OutputCollection.update), [`NestedPartitionSpec`](utils.md#NestedPartitionSpec), [`schedule`](pipeline.md#Pipeline.Config.schedule), [`state`](module.md#InvocationContext.state), [`get_or_none`](utils.md#get_or_none), [`maybe_prepend_axis`](param_init.md#maybe_prepend_axis), [`num_layers`](pipeline.md#Pipeline.Config.num_layers)  (+11 more)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), [`initialize_parameters_recursively`](base_layer.md#BaseLayer.initialize_parameters_recursively), [`create_parameter_specs_recursively`](base_layer.md#BaseLayer.create_parameter_specs_recursively), [`forward`](attention.md#_TransformerPipeline.forward), [`_TransformerPipeline`](attention.md#_TransformerPipeline)

### `StreamSchedule`  ·  implements/extends BaseSchedule
- def: [`axlearn/common/pipeline.py:421`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L421)
- doc: A schedule utilizing a "streaming" buffer.
- signature: `class StreamSchedule(BaseSchedule):`
- members:
  - `_init_carry_in(self, carry: Nested[Tensor])` — [`L630`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L630) — Computes initial loop state.
  - `compute_carry_input(v_input_t: Tensor, v_carry_output_t_1: Tensor)` — [`L551`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L551)
  - `reshape_carry(v_carry: Tensor)` — [`L647`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L647) — Reshapes from [M, microbatch_size, ...] to [N, M // N, microbatch_size, ...].
  - `rotate_out(x: Tensor)` — [`L621`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L621)
  - `scan(self, fn: _PerStageFn, *, carry: Nested[Tensor], state: Nested[Tensor], xs: Nested[Tensor])` — [`L504`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L504) — See `BaseSchedule.scan` for details.
  - `scan_fn(carry_in: Nested[Tensor], xs_t: tuple[Nested[Tensor], Nested[Tensor]])` — [`L522`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L522) — Processes timestep `t` in the pipeline (in parallel across pipeline stages).
  - `update_buffer(buf: Tensor, buf_col_t: Tensor, v_out_t: Tensor)` — [`L566`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L566) — Updates the column of `buf` at `microbatch_idx `.
- protocol/private: `__init__`[`L495`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L495)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`BaseSchedule`](pipeline.md#BaseSchedule), [`num_stages`](pipeline.md#BaseSchedule.num_stages), [`num_microbatches`](pipeline.md#BaseSchedule.num_microbatches), [`Config`](pipeline.md#BaseSchedule.Config), [`__init__`](config.md#Configurable.__init__), [`_is_valid_stage`](pipeline.md#BaseSchedule._is_valid_stage), [`num_microbatches`](pipeline.md#BaseSchedule.Config.num_microbatches), [`num_stages`](pipeline.md#BaseSchedule.Config.num_stages), [`_mask_invalid_gradients`](pipeline.md#_mask_invalid_gradients), [`_shard_pipeline`](pipeline.md#_shard_pipeline), [`_select_input_or_previous_outputs`](pipeline.md#_select_input_or_previous_outputs), [`_PerStageFn`](pipeline.md#_PerStageFn)
- used by: `test_stack_vs_pipeline`, [`BaseSchedule`](pipeline.md#BaseSchedule), [`scan`](pipeline.md#BaseSchedule.scan)

### `_PerStageFn`  ·  implements/extends Protocol
- def: [`axlearn/common/pipeline.py:168`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L168)
- doc: Per-stage implementation.
- signature: `class _PerStageFn(Protocol):`
- members:
  - `__call__(self, state_n: Tensor, carry_tn: Tensor, prng_key_tn: Tensor, x_tn: Tensor)` — [`L171`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L171) — Computes single stage outputs.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor)
- used by: [`scan`](pipeline.md#StreamSchedule.scan), [`scan`](pipeline.md#GPipeSchedule.scan), [`scan`](pipeline.md#BaseSchedule.scan)

## Functions
- `_mask_invalid_gradients(state: Nested[Tensor], *, is_valid: Tensor)` — [`L144`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L144) — Uses stop_gradient to mask invalid (bubble) microbatch iterations.
- `_select_input_or_previous_outputs(input_t: Tensor, output_t_1: Tensor)` — [`L117`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L117) — Computes per-stage inputs by merging input and outputs from the previous time step.
- `_shard_pipeline(x: Tensor, *, axis: int = 0)` — [`L156`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L156) — Shards axis over 'pipeline'.
- `transpose_from_pipeline_stage_outputs(x: Tensor, partition_spec: Optional[PartitionSpec] = None)` — [`L84`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L84) — Transposes `x` from the 'pipeline-major' layout to the 'layer-major' layout.
- `transpose_to_pipeline_stage_inputs(x: Tensor, partition_spec: Optional[PartitionSpec] = None)` — [`L54`](../../../../../../raw/code/axlearn/axlearn/common/pipeline.py#L54) — Transposes `x` from the 'layer-major' layout to the 'pipeline-major' layout.

