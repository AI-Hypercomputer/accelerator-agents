---
title: 'Module: axlearn/common/repeat.py'
type: catalog
provenance: extracted
module: axlearn/common/repeat.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.repeat`/
symbols:
  Repeat._run: Repeat#_run().
  Repeat.create_parameter_specs_recursively: Repeat#create_parameter_specs_recursively().
  Repeat.initialize_parameters_recursively: Repeat#initialize_parameters_recursively().
  Repeat.__init__: Repeat#__init__().
  Repeat.Config: Repeat#Config#
  Repeat: Repeat#
  Repeat.Config.drop_output: Repeat#Config#drop_output.
  Repeat._drop_output: Repeat#_drop_output.
  Repeat.Config.layer: Repeat#Config#layer.
  Repeat.Config.num_layers: Repeat#Config#num_layers.
  Repeat.transform_factorization_spec: Repeat#transform_factorization_spec().
  Repeat.init: Repeat#init().
  Repeat.Output.carry: Repeat#Output#carry.
  Repeat.Output.ys: Repeat#Output#ys.
  _drop_by_regex: _drop_by_regex().
  Repeat._layer_config: Repeat#_layer_config().
  Repeat.Output: Repeat#Output#
  Repeat.Config.unroll: Repeat#Config#unroll.
---
# Module: [`axlearn/common/repeat.py`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py)

## Classes
### `Config`  ·  implements/extends Config
- def: [`axlearn/common/repeat.py:93`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L93)
- doc: Config class for the Repeat layer.
- signature: `class Config(BaseLayer.Config):`
- members:
  - `drop_output` — [`L103`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L103)
  - `layer` — [`L97`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L97)
  - `num_layers` — [`L99`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L99)
  - `unroll` — [`L111`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L111)
- uses (calls/refs, reference-scoped): [`BaseLayer`](base_layer.md#BaseLayer), [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`InstantiableConfig`](config.md#InstantiableConfig), [`config_for_function`](config.md#config_for_function), [`Config`](base_layer.md#BaseLayer.Config), [`_drop_by_regex`](repeat.md#_drop_by_regex)
- used by: [`_forward_for_mode`](attention.md#_TransformerRepeat._forward_for_mode), [`__init__`](repeat.md#Repeat.__init__), [`repeat`](attention.md#RepeatedTransformerLayer.Config.repeat), [`_drop_output`](repeat.md#Repeat._drop_output), [`Config`](attention.md#_TransformerRepeat.Config)

### `Output`  ·  implements/extends NamedTuple
- def: [`axlearn/common/repeat.py:166`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L166)
- signature: `class Output(NamedTuple):`
- members:
  - `carry` — [`L167`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L167)
  - `ys` — [`L168`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L168)
- uses (calls/refs, reference-scoped): [`NestedTensor`](utils.md#NestedTensor)
- used by: [`_forward_for_mode`](attention.md#_TransformerRepeat._forward_for_mode), [`_run`](repeat.md#Repeat._run)

### `Repeat`  ·  implements/extends BaseLayer
- def: [`axlearn/common/repeat.py:89`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L89)
- doc: A layer which repeats a sub layer sequentially using a jax.lax.scan loop.
- signature: `class Repeat(BaseLayer):`
- members:
  - `_run(self, fn, carry=None, *, xs=None)` — [`L170`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L170) — Invokes 'fn' for each sub-layer.
  - `create_parameter_specs_recursively(self)` — [`L125`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L125)
  - `init(prng_key_i)` — [`L154`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L154)
  - `initialize_parameters_recursively(self, prng_key: Tensor, *, prebuilt: Optional[Nested[Optional[ParameterSpec]]] = None)` — [`L151`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L151)
  - `transform_factorization_spec(spec: Optional[FactorizationSpec])` — [`L129`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L129)
- protocol/private: `__init__`[`L113`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L113), `_drop_output`[`L118`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L118), `_layer_config`[`L122`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L122)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`BaseLayer`](base_layer.md#BaseLayer), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`ParameterSpec`](base_layer.md#ParameterSpec), [`__init__`](base_layer.md#BaseLayer.__init__), [`FanAxes`](param_init.md#FanAxes), [`child_context`](module.md#child_context), [`create_parameter_specs_recursively`](base_layer.md#BaseLayer.create_parameter_specs_recursively), [`new_output_collection`](module.md#new_output_collection), [`VDict`](utils.md#VDict), [`FactorizationSpec`](base_layer.md#FactorizationSpec), [`instantiate`](config.md#InstantiableConfig.instantiate), [`output_collection`](module.md#InvocationContext.output_collection), [`get_invocation_context`](module.md#Module.get_invocation_context), [`split_prng_key`](utils.md#split_prng_key), [`prng_key`](module.md#Module.prng_key), [`axes`](base_layer.md#FactorizationSpec.axes), [`scan_in_context`](module.md#scan_in_context), [`NestedParameterSpec`](base_layer.md#NestedParameterSpec), [`update`](module.md#OutputCollection.update), [`state`](module.md#InvocationContext.state), [`get_or_none`](utils.md#get_or_none), [`Config`](repeat.md#Repeat.Config), [`maybe_prepend_axis`](param_init.md#maybe_prepend_axis), [`drop_output`](repeat.md#Repeat.Config.drop_output), [`AxisType`](param_init.md#FanAxes.AxisType), [`keys`](utils.md#StackedKeyArray.keys), [`_TransformerRepeat`](attention.md#_TransformerRepeat), [`carry`](repeat.md#Repeat.Output.carry), [`ys`](repeat.md#Repeat.Output.ys), [`BATCH_AXIS`](param_init.md#FanAxes.AxisType.BATCH_AXIS), [`Output`](repeat.md#Repeat.Output)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), [`initialize_parameters_recursively`](base_layer.md#BaseLayer.initialize_parameters_recursively), [`_forward_for_mode`](attention.md#_TransformerRepeat._forward_for_mode), [`create_parameter_specs_recursively`](base_layer.md#BaseLayer.create_parameter_specs_recursively), [`repeat`](attention.md#RepeatedTransformerLayer.Config.repeat), [`Config`](attention.md#_TransformerRepeat.Config), [`_TransformerRepeat`](attention.md#_TransformerRepeat)

## Functions
- `_drop_by_regex(rules: Sequence[str])` — [`L84`](../../../../../../raw/code/axlearn/axlearn/common/repeat.py#L84) — Returns a drop that regex-matches inputs against `rules`.

