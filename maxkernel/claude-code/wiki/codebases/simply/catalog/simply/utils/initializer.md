---
title: 'Module: simply/utils/initializer.py'
type: catalog
provenance: extracted
module: simply/utils/initializer.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.initializer`/
symbols:
  Initializer: Initializer#
  XavierUniformInit: XavierUniformInit#
  Initializer.init: Initializer#init().
  LecunNormalInit: LecunNormalInit#
  ZeroInit: ZeroInit#
  InitializerRegistry: InitializerRegistry#
  HeNormalInit: HeNormalInit#
  Initializer.__post_init__: Initializer#__post_init__().
  InitializerRegistry.namespace: InitializerRegistry#namespace.
  HeNormalInit.scale: HeNormalInit#scale.
  IdentityInit: IdentityInit#
  Initializer.__call__: Initializer#__call__().
  XavierUniformInit.init: XavierUniformInit#init().
  LecunNormalInit.init: LecunNormalInit#init().
  XavierUniformInit.scale: XavierUniformInit#scale.
  LecunNormalInit.scale: LecunNormalInit#scale.
  IdentityInit.init: IdentityInit#init().
  ZeroInit.init: ZeroInit#init().
---
# Module: [`simply/utils/initializer.py`](../../../../../../raw/code/simply/simply/utils/initializer.py)

## Classes
### `HeNormalInit`
- def: [`simply/utils/initializer.py:156`](../../../../../../raw/code/simply/simply/utils/initializer.py#L156)
- doc: Initializes with He normal distribution.
- signature: `class HeNormalInit(LecunNormalInit):`
- members:
  - `scale` — [`L159`](../../../../../../raw/code/simply/simply/utils/initializer.py#L159)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`LecunNormalInit`](initializer.md#LecunNormalInit), [`InitializerRegistry`](initializer.md#InitializerRegistry)
- used by: [`test_he_normal`](initializer_test.md#InitializerTest.test_he_normal)

### `IdentityInit`
- def: [`simply/utils/initializer.py:164`](../../../../../../raw/code/simply/simply/utils/initializer.py#L164)
- doc: Initializes 2D arrays as identity matrices.
- signature: `class IdentityInit(Initializer):`
- members:
  - `init(self, prng_key: jax.Array, shape: Sequence[int], dim_annotation: str, dtype: jax.typing.DTypeLike)` — [`L172`](../../../../../../raw/code/simply/simply/utils/initializer.py#L172) — Initializes 2D arrays as identity matrices.
- uses (calls/refs, reference-scoped): [`Initializer`](initializer.md#Initializer)
- used by: [`init`](initializer.md#Initializer.init), [`test_gmoe_routing`](module_test.md#EinsumLinearTest.test_gmoe_routing)

### `Initializer`  ·  implements/extends ABC
- def: [`simply/utils/initializer.py:29`](../../../../../../raw/code/simply/simply/utils/initializer.py#L29)
- doc: Initializer.
- signature: `class Initializer(abc.ABC):`
- members:
  - `init(self, prng_key: jax.Array, shape: Sequence[int], dim_annotation: str, dtype: jax.typing.DTypeLike)` — [`L45`](../../../../../../raw/code/simply/simply/utils/initializer.py#L45) — Initializes an array given.
- protocol/private: `__call__`[`L67`](../../../../../../raw/code/simply/simply/utils/initializer.py#L67), `__post_init__`[`L32`](../../../../../../raw/code/simply/simply/utils/initializer.py#L32)
- uses (calls/refs, reference-scoped): [`get`](registry.md#RootRegistry.get), [`fullname`](registry.md#RootRegistry.fullname), [`InitializerRegistry`](initializer.md#InitializerRegistry), [`init`](initializer.md#LecunNormalInit.init), [`init`](initializer.md#XavierUniformInit.init), [`init`](initializer.md#IdentityInit.init), [`init`](initializer.md#ZeroInit.init)
- used by: [`XavierUniformInit`](initializer.md#XavierUniformInit), [`LecunNormalInit`](initializer.md#LecunNormalInit), [`ZeroInit`](initializer.md#ZeroInit), [`ffn_weight_init`](../model_lib.md#FeedForward.ffn_weight_init), [`weight_init`](vision_encoder.md#PatchEncoder.weight_init), [`attn_weight_init`](../model_lib.md#TransformerBlock.attn_weight_init), [`bias_init`](module.md#EinsumLinear.bias_init), [`bias_init`](module.md#EmbeddingLinear.bias_init), [`embed_init`](module.md#EmbeddingLinear.embed_init), [`ffn_weight_init`](../model_lib.md#TransformerBlock.ffn_weight_init), [`weight_init`](../model_lib.md#Attention.weight_init), [`weight_init`](module.md#EinsumLinear.weight_init), [`weight_init`](module.md#EmbeddingLinear.weight_init), [`attn_weight_init`](../config_lib.md#BaseExperimentConfig.attn_weight_init), [`ffn_weight_init`](../config_lib.md#BaseExperimentConfig.ffn_weight_init), [`IdentityInit`](initializer.md#IdentityInit)

### `InitializerRegistry`
- def: [`simply/utils/initializer.py:78`](../../../../../../raw/code/simply/simply/utils/initializer.py#L78)
- members:
  - `namespace` — [`L87`](../../../../../../raw/code/simply/simply/utils/initializer.py#L87)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`RootRegistry`](registry.md#RootRegistry)
- used by: [`LecunNormalInit`](initializer.md#LecunNormalInit), [`ZeroInit`](initializer.md#ZeroInit), [`HeNormalInit`](initializer.md#HeNormalInit), [`__post_init__`](initializer.md#Initializer.__post_init__), [`scale`](initializer.md#HeNormalInit.scale)

### `LecunNormalInit`
- def: [`simply/utils/initializer.py:128`](../../../../../../raw/code/simply/simply/utils/initializer.py#L128)
- doc: Initializes with Lecun normal distribution.
- signature: `class LecunNormalInit(Initializer):`
- members:
  - `init(self, prng_key: jax.Array, shape: Sequence[int], dim_annotation: str, dtype: jax.typing.DTypeLike)` — [`L133`](../../../../../../raw/code/simply/simply/utils/initializer.py#L133) — Initializes with He normal distribution.
  - `scale` — [`L131`](../../../../../../raw/code/simply/simply/utils/initializer.py#L131)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`Initializer`](initializer.md#Initializer), [`InitializerRegistry`](initializer.md#InitializerRegistry)
- used by: [`init`](initializer.md#Initializer.init), [`HeNormalInit`](initializer.md#HeNormalInit), [`embed_init`](module.md#EmbeddingLinear.embed_init), [`weight_init`](module.md#EmbeddingLinear.weight_init)

### `XavierUniformInit`
- def: [`simply/utils/initializer.py:92`](../../../../../../raw/code/simply/simply/utils/initializer.py#L92)
- doc: Initializes with Xavier uniform distribution.
- signature: `class XavierUniformInit(Initializer):`
- members:
  - `init(self, prng_key: jax.Array, shape: Sequence[int], dim_annotation: str, dtype: jax.typing.DTypeLike)` — [`L97`](../../../../../../raw/code/simply/simply/utils/initializer.py#L97) — Initializes with Xavier uniform distribution.
  - `scale` — [`L95`](../../../../../../raw/code/simply/simply/utils/initializer.py#L95)
- uses (calls/refs, reference-scoped): [`Initializer`](initializer.md#Initializer)
- used by: [`init`](initializer.md#Initializer.init), [`ffn_weight_init`](../model_lib.md#FeedForward.ffn_weight_init), [`weight_init`](vision_encoder.md#PatchEncoder.weight_init), [`attn_weight_init`](../model_lib.md#TransformerBlock.attn_weight_init), [`ffn_weight_init`](../model_lib.md#TransformerBlock.ffn_weight_init), [`weight_init`](../model_lib.md#Attention.weight_init), [`weight_init`](module.md#EinsumLinear.weight_init), [`attn_weight_init`](../config_lib.md#BaseExperimentConfig.attn_weight_init), [`ffn_weight_init`](../config_lib.md#BaseExperimentConfig.ffn_weight_init), [`test_xavier_uniform`](initializer_test.md#InitializerTest.test_xavier_uniform)

### `ZeroInit`
- def: [`simply/utils/initializer.py:193`](../../../../../../raw/code/simply/simply/utils/initializer.py#L193)
- doc: Initializes arrays with all zeros.
- signature: `class ZeroInit(Initializer):`
- members:
  - `init(self, prng_key: jax.Array, shape: Sequence[int], dim_annotation: str, dtype: jax.typing.DTypeLike)` — [`L196`](../../../../../../raw/code/simply/simply/utils/initializer.py#L196) — Initializes arrays with all zeros.
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`Initializer`](initializer.md#Initializer), [`InitializerRegistry`](initializer.md#InitializerRegistry)
- used by: [`init`](initializer.md#Initializer.init), [`bias_init`](module.md#EinsumLinear.bias_init), [`bias_init`](module.md#EmbeddingLinear.bias_init)

