---
title: 'Module: axlearn/common/quantized_dot_general/layers.py'
type: catalog
provenance: extracted
module: axlearn/common/quantized_dot_general/layers.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.quantized_dot_general.layers`/
symbols:
  QuantizedDotGeneral._create_layer_parameter_specs: QuantizedDotGeneral#_create_layer_parameter_specs().
  QuantizedDotGeneral._dot_general_maybe_quantized: QuantizedDotGeneral#_dot_general_maybe_quantized().
  QuantizedDotGeneral.einsum_maybe_quantized: QuantizedDotGeneral#einsum_maybe_quantized().
  set_quantized_dot_general_recursively: set_quantized_dot_general_recursively().
  QuantizedDotGeneral.rhs_act_dot_general: QuantizedDotGeneral#rhs_act_dot_general.
  QuantizedDotGeneral.__init__: QuantizedDotGeneral#__init__().
  DenseGeneralBaseLayer: DenseGeneralBaseLayer#
  DenseGeneralBaseLayer.__init__: DenseGeneralBaseLayer#__init__().
  QuantizedDotGeneral._fp8_dot: QuantizedDotGeneral#_fp8_dot().
  DenseGeneralBaseLayer.Config: DenseGeneralBaseLayer#Config#
  QuantizedDotGeneral.Config: QuantizedDotGeneral#Config#
  QuantizedDotGeneral.Config.quantization_type: QuantizedDotGeneral#Config#quantization_type.
  DotGeneralQuantizationType: DotGeneralQuantizationType#
  DenseGeneralBaseLayer.Config.quantized_dot_general: DenseGeneralBaseLayer#Config#quantized_dot_general.
  set_quantized_dot_general_recursively.visit_fn: set_quantized_dot_general_recursively().visit_fn().
  QuantizedDotGeneral: QuantizedDotGeneral#
  BaseQuantizedEinsum: BaseQuantizedEinsum#
  QuantizedDotGeneral.Config.activation_clipping: QuantizedDotGeneral#Config#activation_clipping.
  DenseGeneralBaseLayer.einsum_maybe_quantized: DenseGeneralBaseLayer#einsum_maybe_quantized().
  set_quantized_dot_general_recursively.enter_fn: set_quantized_dot_general_recursively().enter_fn().
  get_all_fp8_param_names: get_all_fp8_param_names().
  BaseQuantizedEinsum.einsum_maybe_quantized: BaseQuantizedEinsum#einsum_maybe_quantized().
  QuantizedDotGeneral.Config.clipping_choice: QuantizedDotGeneral#Config#clipping_choice.
  DotGeneralQuantizationType.FP_8: DotGeneralQuantizationType#FP_8.
  ClippingChoice: ClippingChoice#
  QuantizedDotGeneral.Config.fp8_amax_history_length: QuantizedDotGeneral#Config#fp8_amax_history_length.
  QuantizedDotGeneral.lhs_act_dot_general: QuantizedDotGeneral#lhs_act_dot_general.
  FP8ScaleParams: FP8ScaleParams#
  FP8AmaxHistoryParams: FP8AmaxHistoryParams#
  DotGeneralQuantizationType.INT_8: DotGeneralQuantizationType#INT_8.
  ClippingChoice.INPUT_ACTIVATION: ClippingChoice#INPUT_ACTIVATION.
  PrecisionLike: PrecisionLike.
  ClippingChoice.OUTPUT_ACTIVATION: ClippingChoice#OUTPUT_ACTIVATION.
  FP8ScaleParams.INPUT_SCALE: FP8ScaleParams#INPUT_SCALE.
  FP8ScaleParams.KERNEL_SCALE: FP8ScaleParams#KERNEL_SCALE.
  FP8ScaleParams.OUTPUT_GRADE_SCALE: FP8ScaleParams#OUTPUT_GRADE_SCALE.
  FP8AmaxHistoryParams.INPUT_AMAX_HISTORY: FP8AmaxHistoryParams#INPUT_AMAX_HISTORY.
  FP8AmaxHistoryParams.KERNEL_AMAX_HISTORY: FP8AmaxHistoryParams#KERNEL_AMAX_HISTORY.
  FP8AmaxHistoryParams.OUTPUT_GRAD_AMAX_HISTORY: FP8AmaxHistoryParams#OUTPUT_GRAD_AMAX_HISTORY.
---
# Module: [`axlearn/common/quantized_dot_general/layers.py`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py)

## Classes
### `BaseQuantizedEinsum`  ·  implements/extends BaseLayer
- def: [`axlearn/common/quantized_dot_general/layers.py:85`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L85)
- doc: An abstract class to define the common interface for layers implementing
- signature: `class BaseQuantizedEinsum(BaseLayer):`
- members:
  - `einsum_maybe_quantized(self, subscripts, *, activation: Tensor, kernel: Tensor)` — [`L90`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L90) — Implements activation-kernel einsum with quantization (e.g. fakequant, fp8-fp8, etc.)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`BaseLayer`](../base_layer.md#BaseLayer), [`einsum_maybe_quantized`](layers.md#QuantizedDotGeneral.einsum_maybe_quantized), [`QuantizedDotGeneral`](layers.md#QuantizedDotGeneral)
- used by: [`BaseLayer`](../base_layer.md#BaseLayer), [`quantized_dot_general`](layers.md#DenseGeneralBaseLayer.Config.quantized_dot_general), [`QuantizedDotGeneral`](layers.md#QuantizedDotGeneral)

### `ClippingChoice`  ·  implements/extends Enum
- def: [`axlearn/common/quantized_dot_general/layers.py:59`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L59)
- doc: Which tensor should we apply clipping to.
- signature: `class ClippingChoice(Enum):`
- members:
  - `INPUT_ACTIVATION` — [`L63`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L63)
  - `OUTPUT_ACTIVATION` — [`L65`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L65)
- used by: [`einsum_maybe_quantized`](layers.md#QuantizedDotGeneral.einsum_maybe_quantized), [`clipping_choice`](layers.md#QuantizedDotGeneral.Config.clipping_choice)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/quantized_dot_general/layers.py:383`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L383)
- signature: `class Config(BaseLayer.Config):`
- members:
  - `activation_clipping` — [`L128`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L128)
  - `clipping_choice` — [`L130`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L130)
  - `fp8_amax_history_length` — [`L136`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L136)
  - `quantization_type` — [`L126`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L126)
  - `quantized_dot_general` — [`L390`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L390)
- uses (calls/refs, reference-scoped): [`BaseLayer`](../base_layer.md#BaseLayer), [`config_class`](../config.md#config_class), [`Config`](../base_layer.md#BaseLayer.Config), [`BaseActivationClippingLayer`](activation_clipping.md#BaseActivationClippingLayer), [`Config`](activation_clipping.md#BaseActivationClippingLayer.Config), [`DotGeneralQuantizationType`](layers.md#DotGeneralQuantizationType), [`BaseQuantizedEinsum`](layers.md#BaseQuantizedEinsum), [`ClippingChoice`](layers.md#ClippingChoice), [`INPUT_ACTIVATION`](layers.md#ClippingChoice.INPUT_ACTIVATION)
- used by: [`_create_layer_parameter_specs`](layers.md#QuantizedDotGeneral._create_layer_parameter_specs), [`_dot_general_maybe_quantized`](layers.md#QuantizedDotGeneral._dot_general_maybe_quantized), [`rhs_act_dot_general`](layers.md#QuantizedDotGeneral.rhs_act_dot_general), [`__init__`](layers.md#QuantizedDotGeneral.__init__), [`__init__`](layers.md#DenseGeneralBaseLayer.__init__), [`Config`](../attention.md#BaseMultiheadLinear.Config), [`visit_fn`](layers.md#set_quantized_dot_general_recursively.visit_fn), [`Config`](../layers.md#Linear.Config), [`enter_fn`](layers.md#set_quantized_dot_general_recursively.enter_fn)

### `DenseGeneralBaseLayer`  ·  implements/extends BaseLayer
- def: [`axlearn/common/quantized_dot_general/layers.py:367`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L367)
- doc: Base Layer for all Linear transformations.
- signature: `class DenseGeneralBaseLayer(BaseLayer):`
- members:
  - `einsum_maybe_quantized(self, subscript: str, *, activation: Tensor, kernel: Tensor)` — [`L398`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L398) — Computes einsum with `layer.quantized_dot_general` if available.
- protocol/private: `__init__`[`L392`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L392)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`config`](../config.md#Configurable.config), [`BaseLayer`](../base_layer.md#BaseLayer), [`Module`](../module.md#Module), [`_add_child`](../module.md#Module._add_child), [`__init__`](../base_layer.md#BaseLayer.__init__), [`children`](../module.md#Module.children), [`BaseMultiheadLinear`](../attention.md#BaseMultiheadLinear), [`Config`](layers.md#DenseGeneralBaseLayer.Config), [`Linear`](../layers.md#Linear), [`quantized_dot_general`](layers.md#DenseGeneralBaseLayer.Config.quantized_dot_general)
- used by: [`BaseLayer`](../base_layer.md#BaseLayer), [`forward`](../layers.md#Linear.forward), [`BaseMultiheadLinear`](../attention.md#BaseMultiheadLinear), [`Config`](../attention.md#BaseMultiheadLinear.Config), [`forward`](../attention.md#BaseMultiheadLinear.forward), [`Linear`](../layers.md#Linear), [`visit_fn`](layers.md#set_quantized_dot_general_recursively.visit_fn), [`Config`](../layers.md#Linear.Config), [`enter_fn`](layers.md#set_quantized_dot_general_recursively.enter_fn)

### `DotGeneralQuantizationType`  ·  implements/extends Enum
- def: [`axlearn/common/quantized_dot_general/layers.py:50`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L50)
- doc: Types of hardware accelerated quantization available.
- signature: `class DotGeneralQuantizationType(Enum):`
- members:
  - `FP_8` — [`L56`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L56)
  - `INT_8` — [`L54`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L54)
- used by: [`_create_layer_parameter_specs`](layers.md#QuantizedDotGeneral._create_layer_parameter_specs), `test_qkv_equality`, [`_dot_general_maybe_quantized`](layers.md#QuantizedDotGeneral._dot_general_maybe_quantized), [`rhs_act_dot_general`](layers.md#QuantizedDotGeneral.rhs_act_dot_general), [`__init__`](layers.md#QuantizedDotGeneral.__init__), [`quantization_type`](layers.md#QuantizedDotGeneral.Config.quantization_type)

### `FP8AmaxHistoryParams`  ·  implements/extends Enum
- def: [`axlearn/common/quantized_dot_general/layers.py:75`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L75)
- signature: `class FP8AmaxHistoryParams(Enum):`
- members:
  - `INPUT_AMAX_HISTORY` — [`L76`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L76)
  - `KERNEL_AMAX_HISTORY` — [`L77`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L77)
  - `OUTPUT_GRAD_AMAX_HISTORY` — [`L78`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L78)
- used by: [`_create_layer_parameter_specs`](layers.md#QuantizedDotGeneral._create_layer_parameter_specs), [`_fp8_dot`](layers.md#QuantizedDotGeneral._fp8_dot), [`get_all_fp8_param_names`](layers.md#get_all_fp8_param_names)

### `FP8ScaleParams`  ·  implements/extends Enum
- def: [`axlearn/common/quantized_dot_general/layers.py:69`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L69)
- signature: `class FP8ScaleParams(Enum):`
- members:
  - `INPUT_SCALE` — [`L70`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L70)
  - `KERNEL_SCALE` — [`L71`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L71)
  - `OUTPUT_GRADE_SCALE` — [`L72`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L72)
- used by: [`_create_layer_parameter_specs`](layers.md#QuantizedDotGeneral._create_layer_parameter_specs), [`_fp8_dot`](layers.md#QuantizedDotGeneral._fp8_dot), [`get_all_fp8_param_names`](layers.md#get_all_fp8_param_names)

### `QuantizedDotGeneral`  ·  implements/extends BaseQuantizedEinsum
- def: [`axlearn/common/quantized_dot_general/layers.py:106`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L106)
- doc: Hardware accelerated quantized dot general layer.
- signature: `class QuantizedDotGeneral(BaseQuantizedEinsum):`
- members:
  - `_dot_general_maybe_quantized(self, lhs: Tensor, rhs: Tensor, dimension_numbers: DotDimensionNumbers, precision: PrecisionLike = None, preferred_element_type: Optional[DTypeLike] = None, prng_key: Optional[Tensor] = None, lhs_is_activation: bool = True)` — [`L205`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L205) — Utilize hardware accelerated quantized dot_general depending on config and hardware.
  - `einsum_maybe_quantized(self, subscripts, *, activation: Tensor, kernel: Tensor)` — [`L325`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L325) — jnp.einsum which uses hardware accelerated quantization if applicable.
  - `lhs_act_dot_general` — [`L161`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L161)
  - `rhs_act_dot_general` — [`L163`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L163)
- protocol/private: `__init__`[`L138`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L138), `_create_layer_parameter_specs`[`L182`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L182), `_fp8_dot`[`L282`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L282)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`config`](../config.md#Configurable.config), [`Module`](../module.md#Module), [`_add_child`](../module.md#Module._add_child), [`ParameterSpec`](../base_layer.md#ParameterSpec), [`shape`](../utils.md#TensorSpec.shape), [`mesh_axes`](../utils.md#TensorSpec.mesh_axes), [`dtype`](../utils.md#TensorSpec.dtype), [`__init__`](../base_layer.md#BaseLayer.__init__), [`_create_layer_parameter_specs`](../base_layer.md#BaseLayer._create_layer_parameter_specs), [`parameters`](../base_layer.md#BaseLayer.parameters), [`children`](../module.md#Module.children), [`constant_initializer`](../param_init.md#constant_initializer), [`initializer`](../base_layer.md#ParameterSpec.initializer), [`weight_decay_scale`](../base_layer.md#ParameterSpec.weight_decay_scale), [`prng_key`](../module.md#Module.prng_key), [`Config`](layers.md#QuantizedDotGeneral.Config), [`quantization_type`](layers.md#QuantizedDotGeneral.Config.quantization_type), [`DotGeneralQuantizationType`](layers.md#DotGeneralQuantizationType), [`BaseQuantizedEinsum`](layers.md#BaseQuantizedEinsum), [`activation_clipping`](layers.md#QuantizedDotGeneral.Config.activation_clipping), [`ClippingChoice`](layers.md#ClippingChoice), [`FP_8`](layers.md#DotGeneralQuantizationType.FP_8), [`fp8_amax_history_length`](layers.md#QuantizedDotGeneral.Config.fp8_amax_history_length), [`FP8AmaxHistoryParams`](layers.md#FP8AmaxHistoryParams), [`FP8ScaleParams`](layers.md#FP8ScaleParams), [`INPUT_ACTIVATION`](layers.md#ClippingChoice.INPUT_ACTIVATION), [`INT_8`](layers.md#DotGeneralQuantizationType.INT_8), [`is_einsum_swapped_operands`](utils.md#is_einsum_swapped_operands), [`lhs_activation_aqt_config`](utils.md#lhs_activation_aqt_config), [`rhs_activation_aqt_config`](utils.md#rhs_activation_aqt_config), [`OUTPUT_ACTIVATION`](layers.md#ClippingChoice.OUTPUT_ACTIVATION), [`PrecisionLike`](layers.md#PrecisionLike), [`q_dot_q`](fp8_ops.md#q_dot_q)
- used by: [`_create_layer_parameter_specs`](../base_layer.md#BaseLayer._create_layer_parameter_specs), `test_qkv_equality`, [`set_quantized_dot_general_recursively`](layers.md#set_quantized_dot_general_recursively), [`BaseQuantizedEinsum`](layers.md#BaseQuantizedEinsum), [`einsum_maybe_quantized`](layers.md#BaseQuantizedEinsum.einsum_maybe_quantized)

## Functions
- `enter_fn(_, value, default_kv)` — [`L441`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L441)
- `get_all_fp8_param_names()` — [`L81`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L81)
- `set_quantized_dot_general_recursively(cfg: BaseLayer.Config, quantized_dot_general: Optional[QuantizedDotGeneral.Config], set_only_if_none: bool = False)` — [`L419`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L419) — Sets QuantizedDotGeneral.Config recursively.
- `visit_fn(_, value)` — [`L436`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L436)

## Module values
- `PrecisionLike` — [`L47`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers.py#L47)

