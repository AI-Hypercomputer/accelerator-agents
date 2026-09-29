---
title: 'Module: axlearn/common/param_init.py'
type: catalog
provenance: extracted
module: axlearn/common/param_init.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.param_init`/
symbols:
  FanAxes: FanAxes#
  Initializer: Initializer#
  DefaultInitializer.__init__: DefaultInitializer#__init__().
  Initializer.initialize: Initializer#initialize().
  Initializer.debug_string: Initializer#debug_string().
  constant_initializer: constant_initializer().
  PerGroupInitializer.initializer: PerGroupInitializer#initializer.
  DefaultInitializer: DefaultInitializer#
  WeightInitializer.initialize: WeightInitializer#initialize().
  FanAxes._insert_axis: FanAxes#_insert_axis().
  Shape: Shape.
  ConstantInitializer: ConstantInitializer#
  WeightInitializer: WeightInitializer#
  PerGroupInitializer.initialize: PerGroupInitializer#initialize().
  FanAxes.in_axis: FanAxes#in_axis.
  FanAxes.out_axis: FanAxes#out_axis.
  DefaultInitializer.initialize: DefaultInitializer#initialize().
  maybe_prepend_axis: maybe_prepend_axis().
  ConstantInitializer.initialize: ConstantInitializer#initialize().
  GaussianInitializer.initialize: GaussianInitializer#initialize().
  gaussian_initializer: gaussian_initializer().
  PerGroupInitializer.__init__: PerGroupInitializer#__init__().
  PerGroupInitializer.debug_string: PerGroupInitializer#debug_string().
  DefaultInitializer.Config: DefaultInitializer#Config#
  PerGroupInitializer.Config.initializer: PerGroupInitializer#Config#initializer.
  PerGroupInitializer.init: PerGroupInitializer#init().
  FanAxes.canonicalize: FanAxes#canonicalize().
  FanAxes.prepend_axis: FanAxes#prepend_axis().
  DefaultInitializer.debug_string: DefaultInitializer#debug_string().
  PerGroupInitializer.Config: PerGroupInitializer#Config#
  FanAxes.AxisType: FanAxes#AxisType#
  FanAxes.append_axis: FanAxes#append_axis().
  ConstantInitializer.Config: ConstantInitializer#Config#
  ConstantInitializer.debug_string: ConstantInitializer#debug_string().
  GaussianInitializer.Config: GaussianInitializer#Config#
  GaussianInitializer.debug_string: GaussianInitializer#debug_string().
  WeightInitializer.Config: WeightInitializer#Config#
  WeightInitializer.debug_string: WeightInitializer#debug_string().
  PARAM_REGEXP_WEIGHT: PARAM_REGEXP_WEIGHT.
  GaussianInitializer: GaussianInitializer#
  DefaultInitializer.Config.init_by_param_name: DefaultInitializer#Config#init_by_param_name.
  FanAxes.__eq__: FanAxes#__eq__().
  ConstantInitializer.Config.value: ConstantInitializer#Config#value.
  GaussianInitializer.Config.std: GaussianInitializer#Config#std.
  truncated_normal: truncated_normal().
  uniform: uniform().
  FanAxes.AxisType.BATCH_AXIS: FanAxes#AxisType#BATCH_AXIS.
  FanAxes.batch_axis: FanAxes#batch_axis.
  PerGroupInitializer: PerGroupInitializer#
  PARAM_REGEXP_BIAS: PARAM_REGEXP_BIAS.
  PARAM_REGEXP_SCALE: PARAM_REGEXP_SCALE.
  FanAxes.AxisType.NONE: FanAxes#AxisType#NONE.
  FanAxes.move_axis: FanAxes#move_axis().
  truncated_normal.init: truncated_normal().init().
  uniform.init: uniform().init().
  FanAxes.AxisType.IN_AXIS: FanAxes#AxisType#IN_AXIS.
  FanAxes.AxisType.OUT_AXIS: FanAxes#AxisType#OUT_AXIS.
  GaussianInitializer.Config.mean: GaussianInitializer#Config#mean.
  WeightInitializer.Config.scale: WeightInitializer#Config#scale.
  WeightInitializer.Config.fan: WeightInitializer#Config#fan.
  WeightInitializer.Config.distribution: WeightInitializer#Config#distribution.
  PerGroupInitializer.Config.num_groups: PerGroupInitializer#Config#num_groups.
---
# Module: [`axlearn/common/param_init.py`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py)

## Classes
### `AxisType`  ·  implements/extends Enum
- def: [`axlearn/common/param_init.py:34`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L34)
- signature: `class AxisType(Enum):`
- members:
  - `BATCH_AXIS` — [`L37`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L37)
  - `IN_AXIS` — [`L35`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L35)
  - `NONE` — [`L38`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L38)
  - `OUT_AXIS` — [`L36`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L36)
- used by: [`create_parameter_specs_recursively`](attention.md#FusedQKVLinear.create_parameter_specs_recursively), [`create_parameter_specs_recursively`](pipeline.md#Pipeline.create_parameter_specs_recursively), [`create_parameter_specs_recursively`](repeat.md#Repeat.create_parameter_specs_recursively), [`_insert_axis`](param_init.md#FanAxes._insert_axis), [`maybe_prepend_axis`](param_init.md#maybe_prepend_axis), [`prepend_axis`](param_init.md#FanAxes.prepend_axis), [`append_axis`](param_init.md#FanAxes.append_axis)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/param_init.py:456`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L456)
- signature: `class Config(Initializer.Config):`
- members:
  - `distribution` — [`L298`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L298)
  - `fan` — [`L296`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L296)
  - `init_by_param_name` — [`L384`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L384)
  - `initializer` — [`L458`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L458)
  - `mean` — [`L215`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L215)
  - `num_groups` — [`L460`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L460)
  - `scale` — [`L293`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L293)
  - `std` — [`L214`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L214)
  - `value` — [`L183`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L183)
- uses (calls/refs, reference-scoped): [`default_config`](config.md#Configurable.default_config), [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`InstantiableConfig`](config.md#InstantiableConfig), [`Config`](config.md#Configurable.Config), [`Initializer`](param_init.md#Initializer), [`DefaultInitializer`](param_init.md#DefaultInitializer)
- used by: [`__init__`](param_init.md#DefaultInitializer.__init__), [`initializer`](param_init.md#PerGroupInitializer.initializer), [`__init__`](param_init.md#PerGroupInitializer.__init__)

### `ConstantInitializer`  ·  implements/extends Initializer
- def: [`axlearn/common/param_init.py:178`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L178)
- doc: Constant initializer.
- signature: `class ConstantInitializer(Initializer):`
- members:
  - `debug_string(self, name: Optional[str] = None, shape: Optional[Shape] = None, axes: Optional[FanAxes] = None)` — [`L196`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L196)
  - `initialize(self, name: str, *, prng_key: Tensor, shape: Shape, dtype: jnp.dtype, axes: Optional[FanAxes] = None)` — [`L185`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L185)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`FanAxes`](param_init.md#FanAxes), [`Initializer`](param_init.md#Initializer), [`Shape`](param_init.md#Shape)
- used by: [`_create_layer_parameter_specs`](rattention/rattention.md#ResidualLinearAttention._create_layer_parameter_specs), [`Initializer`](param_init.md#Initializer), [`default_config`](attention.md#PerDimScale.default_config), [`__init__`](param_init.md#DefaultInitializer.__init__), [`initialize`](param_init.md#Initializer.initialize), [`debug_string`](param_init.md#Initializer.debug_string), [`constant_initializer`](param_init.md#constant_initializer), [`gaussian_initializer`](param_init.md#gaussian_initializer)

### `DefaultInitializer`  ·  implements/extends Initializer
- def: [`axlearn/common/param_init.py:375`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L375)
- doc: The default initializer.
- signature: `class DefaultInitializer(Initializer):`
- members:
  - `__init__(self, cfg: Config)` — [`L386`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L386) — Default initializers for common param names if not specified by the user.
  - `debug_string(self, *, name: str, shape: Optional[Shape] = None, axes: Optional[FanAxes] = None)` — [`L427`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L427)
  - `initialize(self, name: str, *, prng_key: Tensor, shape: Shape, dtype: jnp.dtype, axes: Optional[FanAxes] = None)` — [`L400`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L400)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`default_config`](config.md#Configurable.default_config), [`config`](config.md#Configurable.config), [`FanAxes`](param_init.md#FanAxes), [`Initializer`](param_init.md#Initializer), [`Shape`](param_init.md#Shape), [`ConstantInitializer`](param_init.md#ConstantInitializer), [`WeightInitializer`](param_init.md#WeightInitializer), [`__init__`](config.md#Configurable.__init__), [`Config`](param_init.md#DefaultInitializer.Config), [`PARAM_REGEXP_WEIGHT`](param_init.md#PARAM_REGEXP_WEIGHT), [`init_by_param_name`](param_init.md#DefaultInitializer.Config.init_by_param_name), [`PARAM_REGEXP_BIAS`](param_init.md#PARAM_REGEXP_BIAS), [`PARAM_REGEXP_SCALE`](param_init.md#PARAM_REGEXP_SCALE)
- used by: [`__init__`](base_layer.md#BaseLayer.__init__), `test_compute_fan_axes`, [`Initializer`](param_init.md#Initializer), [`initialize`](param_init.md#Initializer.initialize), [`debug_string`](param_init.md#Initializer.debug_string), [`_is_initialized`](trainer.md#SpmdTrainer._is_initialized), [`initializer`](param_init.md#PerGroupInitializer.initializer), [`default_config`](layers.md#Embedding.default_config), [`default_config`](attention.md#LearnedPositionalEmbedding.default_config), [`param_init`](base_layer.md#BaseLayer.Config.param_init), [`debug_string`](param_init.md#PerGroupInitializer.debug_string), [`init`](param_init.md#PerGroupInitializer.init), [`initializer`](param_init.md#PerGroupInitializer.Config.initializer)

### `FanAxes`  ·  implements/extends NamedTuple
- def: [`axlearn/common/param_init.py:24`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L24)
- doc: FanAxes describes axis indices corresponding to input, output, and batch axes.
- signature: `class FanAxes(NamedTuple):`
- members:
  - `_insert_axis(self, axis: int, *, axis_type: FanAxes.AxisType)` — [`L71`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L71) — Returns a copy of this where a new axis of the given type has been inserted
  - `append_axis(self, *, axis_type: FanAxes.AxisType)` — [`L124`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L124) — Returns a copy of this where a new axis of the given type has been appended
  - `canonicalize(self)` — [`L47`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L47) — Returns a FanAxes equivalent to this one where all fields are tuples.
  - `move_axis(index: int)` — [`L91`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L91)
  - `prepend_axis(self, *, axis_type: FanAxes.AxisType)` — [`L111`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L111) — Returns a copy of this where a new axis of the given type has been prepended
  - `batch_axis` — [`L45`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L45)
  - `in_axis` — [`L41`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L41)
  - `out_axis` — [`L43`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L43)
- protocol/private: `__eq__`[`L64`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L64)
- uses (calls/refs, reference-scoped): [`AxisType`](param_init.md#FanAxes.AxisType), [`NONE`](param_init.md#FanAxes.AxisType.NONE)
- used by: [`_create_layer_parameter_specs`](rattention/rattention.md#ResidualLinearAttention._create_layer_parameter_specs), [`create_parameter_specs_recursively`](attention.md#FusedQKVLinear.create_parameter_specs_recursively), `test_compute_fan_axes`, [`create_parameter_specs_recursively`](pipeline.md#Pipeline.create_parameter_specs_recursively), [`_compute_fan_axes`](base_layer.md#BaseLayer._compute_fan_axes), [`create_parameter_specs_recursively`](repeat.md#Repeat.create_parameter_specs_recursively), [`initialize`](param_init.md#Initializer.initialize), [`debug_string`](param_init.md#Initializer.debug_string), [`_compute_fan_axes`](attention.md#LearnedPositionalEmbedding._compute_fan_axes), [`initialize`](param_init.md#WeightInitializer.initialize), [`_compute_fan_axes`](attention.md#BaseMultiheadLinear._compute_fan_axes), [`_compute_fan_axes`](layers.md#MultiLinear._compute_fan_axes), [`initialize`](param_init.md#PerGroupInitializer.initialize), [`fan_axes`](base_layer.md#ParameterSpec.fan_axes), [`initialize`](param_init.md#DefaultInitializer.initialize), [`maybe_prepend_axis`](param_init.md#maybe_prepend_axis), [`_compute_fan_axes`](attention.md#MultiheadInputLinear._compute_fan_axes), [`_compute_fan_axes`](attention.md#MultiheadOutputLinear._compute_fan_axes), [`_compute_fan_axes`](attention.md#MultiheadRelativePositionLinear._compute_fan_axes), [`debug_string`](param_init.md#PerGroupInitializer.debug_string), [`initialize`](param_init.md#ConstantInitializer.initialize), [`initialize`](param_init.md#GaussianInitializer.initialize), [`debug_string`](param_init.md#DefaultInitializer.debug_string), [`debug_string`](param_init.md#ConstantInitializer.debug_string), [`debug_string`](param_init.md#GaussianInitializer.debug_string), [`debug_string`](param_init.md#WeightInitializer.debug_string)  (2 test-only)

### `GaussianInitializer`  ·  implements/extends Initializer
- def: [`axlearn/common/param_init.py:209`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L209)
- doc: Gaussian initializer.
- signature: `class GaussianInitializer(Initializer):`
- members:
  - `debug_string(self, name: Optional[str] = None, shape: Optional[Shape] = None, axes: Optional[FanAxes] = None)` — [`L230`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L230)
  - `initialize(self, name: str, *, prng_key: Tensor, shape: Shape, dtype: jnp.dtype, axes: Optional[FanAxes] = None)` — [`L217`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L217)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`FanAxes`](param_init.md#FanAxes), [`Initializer`](param_init.md#Initializer), [`Shape`](param_init.md#Shape)
- used by: [`_create_layer_parameter_specs`](attention.md#MultiheadAttention._create_layer_parameter_specs), [`Initializer`](param_init.md#Initializer), [`initialize`](param_init.md#Initializer.initialize), [`debug_string`](param_init.md#Initializer.debug_string), [`gaussian_initializer`](param_init.md#gaussian_initializer)

### `Initializer`  ·  implements/extends Configurable
- def: [`axlearn/common/param_init.py:155`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L155)
- doc: Base class for initializers.
- signature: `class Initializer(Configurable):`
- members:
  - `debug_string(self, name: Optional[str] = None, shape: Optional[Shape] = None, axes: Optional[FanAxes] = None)` — [`L169`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L169)
  - `initialize(self, name: str, *, prng_key: Tensor, shape: Shape, dtype: jnp.dtype, axes: Optional[FanAxes] = None)` — [`L158`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L158)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Configurable`](config.md#Configurable), [`FanAxes`](param_init.md#FanAxes), [`DefaultInitializer`](param_init.md#DefaultInitializer), [`initialize`](param_init.md#WeightInitializer.initialize), [`Shape`](param_init.md#Shape), [`ConstantInitializer`](param_init.md#ConstantInitializer), [`WeightInitializer`](param_init.md#WeightInitializer), [`initialize`](param_init.md#PerGroupInitializer.initialize), [`initialize`](param_init.md#DefaultInitializer.initialize), [`debug_string`](param_init.md#PerGroupInitializer.debug_string), [`initialize`](param_init.md#ConstantInitializer.initialize), [`initialize`](param_init.md#GaussianInitializer.initialize), [`debug_string`](param_init.md#DefaultInitializer.debug_string), [`debug_string`](param_init.md#ConstantInitializer.debug_string), [`debug_string`](param_init.md#GaussianInitializer.debug_string), [`debug_string`](param_init.md#WeightInitializer.debug_string), [`GaussianInitializer`](param_init.md#GaussianInitializer), [`PerGroupInitializer`](param_init.md#PerGroupInitializer)  (2 test-only)
- used by: [`Configurable`](config.md#Configurable), [`_initialize_parameter`](base_layer.md#BaseLayer._initialize_parameter), [`initializer`](base_layer.md#ParameterSpec.initializer), [`DefaultInitializer`](param_init.md#DefaultInitializer), [`ConstantInitializer`](param_init.md#ConstantInitializer), [`WeightInitializer`](param_init.md#WeightInitializer), [`Config`](param_init.md#DefaultInitializer.Config), [`Config`](param_init.md#PerGroupInitializer.Config), [`Config`](param_init.md#ConstantInitializer.Config), [`Config`](param_init.md#GaussianInitializer.Config), [`param_init`](base_layer.md#BaseLayer.param_init), [`GaussianInitializer`](param_init.md#GaussianInitializer), [`PerGroupInitializer`](param_init.md#PerGroupInitializer)  (3 test-only)

### `PerGroupInitializer`  ·  implements/extends Initializer
- def: [`axlearn/common/param_init.py:442`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L442)
- doc: The per-group initializer.
- signature: `class PerGroupInitializer(Initializer):`
- members:
  - `debug_string(self, name: Optional[str] = None, shape: Optional[Shape] = None, axes: Optional[FanAxes] = None)` — [`L503`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L503)
  - `init(prng_key_i: Tensor)` — [`L489`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L489)
  - `initialize(self, name: str, *, prng_key: Tensor, shape: Shape, dtype: jnp.dtype, axes: Optional[FanAxes] = None)` — [`L470`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L470) — Per-group initialization.
  - `initializer` — [`L466`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L466)
- protocol/private: `__init__`[`L462`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L462)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`default_config`](config.md#Configurable.default_config), [`config`](config.md#Configurable.config), [`instantiate`](config.md#Configurable.Config.instantiate), [`FanAxes`](param_init.md#FanAxes), [`Initializer`](param_init.md#Initializer), [`instantiate`](config.md#InstantiableConfig.instantiate), [`DefaultInitializer`](param_init.md#DefaultInitializer), [`Shape`](param_init.md#Shape), [`__init__`](config.md#Configurable.__init__), [`initialize`](param_init.md#DefaultInitializer.initialize), [`initializer`](param_init.md#PerGroupInitializer.Config.initializer), [`Config`](param_init.md#PerGroupInitializer.Config), [`debug_string`](param_init.md#DefaultInitializer.debug_string)
- used by: [`Initializer`](param_init.md#Initializer), [`initialize`](param_init.md#Initializer.initialize), [`debug_string`](param_init.md#Initializer.debug_string)

### `WeightInitializer`  ·  implements/extends Initializer
- def: [`axlearn/common/param_init.py:283`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L283)
- doc: Default weight initializer.
- signature: `class WeightInitializer(Initializer):`
- members:
  - `debug_string(self, name: Optional[str] = None, shape: Optional[Shape] = None, axes: Optional[FanAxes] = None)` — [`L334`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L334)
  - `initialize(self, name, *, prng_key: Tensor, shape: Shape, dtype: jnp.dtype, axes: Optional[FanAxes] = None)` — [`L301`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L301)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`FanAxes`](param_init.md#FanAxes), [`Initializer`](param_init.md#Initializer), [`Shape`](param_init.md#Shape), [`truncated_normal`](param_init.md#truncated_normal), [`uniform`](param_init.md#uniform)
- used by: [`_create_layer_parameter_specs`](rattention/rattention.md#ResidualLinearAttention._create_layer_parameter_specs), `test_compute_fan_axes`, [`Initializer`](param_init.md#Initializer), [`__init__`](param_init.md#DefaultInitializer.__init__), [`initialize`](param_init.md#Initializer.initialize), [`debug_string`](param_init.md#Initializer.debug_string), [`default_config`](layers.md#Embedding.default_config), [`default_config`](attention.md#LearnedPositionalEmbedding.default_config)

## Functions
- `constant_initializer(value: Any)` — [`L205`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L205)
- `gaussian_initializer(std: float)` — [`L241`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L241)
- `init(key, shape, dtype=dtype)` — [`L256`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L256)
- `init(key, shape, dtype=dtype)` — [`L276`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L276)
- `maybe_prepend_axis(fan_axes: Optional[FanAxes], *, axis_type: FanAxes.AxisType)` — [`L138`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L138) — Returns `fan_axes.prepend_axis(axis_type=axis_type)` if `fan_axes` is not None.
- `truncated_normal(stddev: float = 0.01, dtype: jnp.dtype = jnp.float_)` — [`L245`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L245) — Truncated normal variant of jax.nn.initializers.
- `uniform(scale: float = 1, dtype: jnp.dtype = jnp.float_)` — [`L265`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L265) — Uniform initializer.

## Module values
- `PARAM_REGEXP_BIAS` — [`L20`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L20)
- `PARAM_REGEXP_SCALE` — [`L21`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L21)
- `PARAM_REGEXP_WEIGHT` — [`L19`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L19)
- `Shape` — [`L17`](../../../../../../raw/code/axlearn/axlearn/common/param_init.py#L17)

