---
title: 'Module: axlearn/common/quantized_dot_general/activation_clipping.py'
type: catalog
provenance: extracted
module: axlearn/common/quantized_dot_general/activation_clipping.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.quantized_dot_general.activation_clipping`/
symbols:
  BaseActivationClippingLayer: BaseActivationClippingLayer#
  HardActivationClippingLayer.__init__: HardActivationClippingLayer#__init__().
  DummyActivationClippingLayer.__init__: DummyActivationClippingLayer#__init__().
  TanhActivationClippingLayer.__init__: TanhActivationClippingLayer#__init__().
  BaseActivationClippingLayer._maybe_add_clipping_summary: BaseActivationClippingLayer#_maybe_add_clipping_summary().
  BaseActivationClippingLayer.Config: BaseActivationClippingLayer#Config#
  BaseActivationClippingLayer.forward: BaseActivationClippingLayer#forward().
  HardActivationClippingLayer.Config: HardActivationClippingLayer#Config#
  DummyActivationClippingLayer.Config: DummyActivationClippingLayer#Config#
  TanhActivationClippingLayer.Config: TanhActivationClippingLayer#Config#
  HardActivationClippingLayer.Config.clipping_max_abs: HardActivationClippingLayer#Config#clipping_max_abs.
  DummyActivationClippingLayer.Config.clipping_max_abs: DummyActivationClippingLayer#Config#clipping_max_abs.
  TanhActivationClippingLayer.Config.clipping_max_abs: TanhActivationClippingLayer#Config#clipping_max_abs.
  HardActivationClippingLayer.forward: HardActivationClippingLayer#forward().
  DummyActivationClippingLayer.forward: DummyActivationClippingLayer#forward().
  TanhActivationClippingLayer.forward: TanhActivationClippingLayer#forward().
  HardActivationClippingLayer: HardActivationClippingLayer#
  DummyActivationClippingLayer: DummyActivationClippingLayer#
  TanhActivationClippingLayer: TanhActivationClippingLayer#
  BaseActivationClippingLayer.Config.clipping_summary: BaseActivationClippingLayer#Config#clipping_summary.
---
# Module: [`axlearn/common/quantized_dot_general/activation_clipping.py`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py)

## Classes
### `BaseActivationClippingLayer`  ·  implements/extends BaseLayer
- def: [`axlearn/common/quantized_dot_general/activation_clipping.py:19`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L19)
- doc: Base class for activation clipping.
- signature: `class BaseActivationClippingLayer(BaseLayer):`
- members:
  - `_maybe_add_clipping_summary(self, x: Tensor, clipping_max_abs: Union[Tensor, float])` — [`L32`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L32) — Adds summary for activation clipping depending on config.
  - `forward(self, x: Tensor)` — [`L57`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L57) — Forward function for clipping method.
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`config`](../config.md#Configurable.config), [`BaseLayer`](../base_layer.md#BaseLayer), [`WeightedSummary`](../metrics.md#WeightedSummary), [`add_summary`](../module.md#Module.add_summary), [`forward`](activation_clipping.md#DummyActivationClippingLayer.forward), [`forward`](activation_clipping.md#HardActivationClippingLayer.forward), [`forward`](activation_clipping.md#TanhActivationClippingLayer.forward), [`DummyActivationClippingLayer`](activation_clipping.md#DummyActivationClippingLayer), [`HardActivationClippingLayer`](activation_clipping.md#HardActivationClippingLayer), [`TanhActivationClippingLayer`](activation_clipping.md#TanhActivationClippingLayer)
- used by: [`BaseLayer`](../base_layer.md#BaseLayer), [`Config`](activation_clipping.md#DummyActivationClippingLayer.Config), [`Config`](activation_clipping.md#HardActivationClippingLayer.Config), [`Config`](activation_clipping.md#TanhActivationClippingLayer.Config), [`forward`](activation_clipping.md#DummyActivationClippingLayer.forward), [`forward`](activation_clipping.md#HardActivationClippingLayer.forward), [`forward`](activation_clipping.md#TanhActivationClippingLayer.forward), [`activation_clipping`](layers.md#QuantizedDotGeneral.Config.activation_clipping), [`DummyActivationClippingLayer`](activation_clipping.md#DummyActivationClippingLayer), [`HardActivationClippingLayer`](activation_clipping.md#HardActivationClippingLayer), [`TanhActivationClippingLayer`](activation_clipping.md#TanhActivationClippingLayer)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/quantized_dot_general/activation_clipping.py:158`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L158)
- signature: `class Config(BaseActivationClippingLayer.Config):`
- members:
  - `clipping_max_abs` — [`L87`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L87)
  - `clipping_max_abs` — [`L123`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L123)
  - `clipping_max_abs` — [`L162`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L162)
  - `clipping_summary` — [`L30`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L30)
- uses (calls/refs, reference-scoped): [`config_class`](../config.md#config_class), [`REQUIRED`](../config.md#REQUIRED), [`Required`](../config.md#Required), [`BaseActivationClippingLayer`](activation_clipping.md#BaseActivationClippingLayer), [`Config`](activation_clipping.md#BaseActivationClippingLayer.Config)
- used by: [`__init__`](activation_clipping.md#DummyActivationClippingLayer.__init__), [`__init__`](activation_clipping.md#HardActivationClippingLayer.__init__), [`__init__`](activation_clipping.md#TanhActivationClippingLayer.__init__)

### `DummyActivationClippingLayer`  ·  implements/extends BaseActivationClippingLayer
- def: [`axlearn/common/quantized_dot_general/activation_clipping.py:115`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L115)
- doc: Does not clip. Only adds summary.
- signature: `class DummyActivationClippingLayer(BaseActivationClippingLayer):`
- members:
  - `forward(self, x: Tensor)` — [`L134`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L134) — Does not clip. Only adds summary.
- protocol/private: `__init__`[`L125`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L125)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`config`](../config.md#Configurable.config), [`Module`](../module.md#Module), [`__init__`](../base_layer.md#BaseLayer.__init__), [`path`](../module.md#Module.path), [`BaseActivationClippingLayer`](activation_clipping.md#BaseActivationClippingLayer), [`_maybe_add_clipping_summary`](activation_clipping.md#BaseActivationClippingLayer._maybe_add_clipping_summary), [`Config`](activation_clipping.md#DummyActivationClippingLayer.Config), [`clipping_max_abs`](activation_clipping.md#DummyActivationClippingLayer.Config.clipping_max_abs)
- used by: [`BaseActivationClippingLayer`](activation_clipping.md#BaseActivationClippingLayer), [`forward`](activation_clipping.md#BaseActivationClippingLayer.forward)

### `HardActivationClippingLayer`  ·  implements/extends BaseActivationClippingLayer
- def: [`axlearn/common/quantized_dot_general/activation_clipping.py:76`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L76)
- doc: Implements "hard" clipping with jax.numpy.clip.
- signature: `class HardActivationClippingLayer(BaseActivationClippingLayer):`
- members:
  - `forward(self, x: Tensor)` — [`L98`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L98) — Implements "hard" clipping with jax.numpy.clip.
- protocol/private: `__init__`[`L89`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L89)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`config`](../config.md#Configurable.config), [`Module`](../module.md#Module), [`__init__`](../base_layer.md#BaseLayer.__init__), [`path`](../module.md#Module.path), [`BaseActivationClippingLayer`](activation_clipping.md#BaseActivationClippingLayer), [`_maybe_add_clipping_summary`](activation_clipping.md#BaseActivationClippingLayer._maybe_add_clipping_summary), [`Config`](activation_clipping.md#HardActivationClippingLayer.Config), [`clipping_max_abs`](activation_clipping.md#HardActivationClippingLayer.Config.clipping_max_abs)
- used by: [`BaseActivationClippingLayer`](activation_clipping.md#BaseActivationClippingLayer), [`forward`](activation_clipping.md#BaseActivationClippingLayer.forward)

### `TanhActivationClippingLayer`  ·  implements/extends BaseActivationClippingLayer
- def: [`axlearn/common/quantized_dot_general/activation_clipping.py:151`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L151)
- doc: Implements "soft" clipping with jax.numpy.tanh.
- signature: `class TanhActivationClippingLayer(BaseActivationClippingLayer):`
- members:
  - `forward(self, x: Tensor)` — [`L173`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L173) — Implements "soft" clipping with jax.numpy.tanh.
- protocol/private: `__init__`[`L164`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/activation_clipping.py#L164)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`config`](../config.md#Configurable.config), [`Module`](../module.md#Module), [`__init__`](../base_layer.md#BaseLayer.__init__), [`path`](../module.md#Module.path), [`BaseActivationClippingLayer`](activation_clipping.md#BaseActivationClippingLayer), [`_maybe_add_clipping_summary`](activation_clipping.md#BaseActivationClippingLayer._maybe_add_clipping_summary), [`Config`](activation_clipping.md#TanhActivationClippingLayer.Config), [`clipping_max_abs`](activation_clipping.md#TanhActivationClippingLayer.Config.clipping_max_abs)
- used by: [`BaseActivationClippingLayer`](activation_clipping.md#BaseActivationClippingLayer), [`forward`](activation_clipping.md#BaseActivationClippingLayer.forward)

