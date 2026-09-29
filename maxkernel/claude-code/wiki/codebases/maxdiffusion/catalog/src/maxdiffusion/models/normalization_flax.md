---
title: 'Module: src/maxdiffusion/models/normalization_flax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/normalization_flax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.normalization_flax`/
symbols:
  AdaLayerNormContinuous.__call__: AdaLayerNormContinuous#__call__().
  AdaLayerNormZero.__call__: AdaLayerNormZero#__call__().
  AdaLayerNormZeroSingle.__call__: AdaLayerNormZeroSingle#__call__().
  FP32LayerNorm: FP32LayerNorm#
  AdaLayerNormZero.dtype: AdaLayerNormZero#dtype.
  AdaLayerNormZero.weights_dtype: AdaLayerNormZero#weights_dtype.
  AdaLayerNormContinuous.elementwise_affine: AdaLayerNormContinuous#elementwise_affine.
  AdaLayerNormZero: AdaLayerNormZero#
  AdaLayerNormZero.precision: AdaLayerNormZero#precision.
  AdaLayerNormZeroSingle.dtype: AdaLayerNormZeroSingle#dtype.
  AdaLayerNormZeroSingle.weights_dtype: AdaLayerNormZeroSingle#weights_dtype.
  FP32LayerNorm.__call__: FP32LayerNorm#__call__().
  AdaLayerNormContinuous: AdaLayerNormContinuous#
  AdaLayerNormContinuous.eps: AdaLayerNormContinuous#eps.
  AdaLayerNormContinuous.dtype: AdaLayerNormContinuous#dtype.
  AdaLayerNormContinuous.weights_dtype: AdaLayerNormContinuous#weights_dtype.
  AdaLayerNormContinuous.precision: AdaLayerNormContinuous#precision.
  AdaLayerNormZero.norm_type: AdaLayerNormZero#norm_type.
  AdaLayerNormZeroSingle: AdaLayerNormZeroSingle#
  AdaLayerNormZeroSingle.norm_type: AdaLayerNormZeroSingle#norm_type.
  AdaLayerNormZeroSingle.precision: AdaLayerNormZeroSingle#precision.
  AdaLayerNormContinuous.embedding_dim: AdaLayerNormContinuous#embedding_dim.
  AdaLayerNormContinuous.bias: AdaLayerNormContinuous#bias.
  AdaLayerNormContinuous.norm_type: AdaLayerNormContinuous#norm_type.
  AdaLayerNormZero.embedding_dim: AdaLayerNormZero#embedding_dim.
  AdaLayerNormZero.bias: AdaLayerNormZero#bias.
  AdaLayerNormZeroSingle.embedding_dim: AdaLayerNormZeroSingle#embedding_dim.
  AdaLayerNormZeroSingle.bias: AdaLayerNormZeroSingle#bias.
  FP32LayerNorm.layer_norm: FP32LayerNorm#layer_norm.
  FP32LayerNorm.__init__: FP32LayerNorm#__init__().
---
# Module: [`src/maxdiffusion/models/normalization_flax.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py)

## Classes
### `AdaLayerNormContinuous`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/normalization_flax.py:23`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L23)
- signature: `class AdaLayerNormContinuous(nn.Module):`
- members:
  - `bias` — [`L27`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L27)
  - `dtype` — [`L29`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L29)
  - `elementwise_affine` — [`L25`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L25)
  - `embedding_dim` — [`L24`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L24)
  - `eps` — [`L26`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L26)
  - `norm_type` — [`L28`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L28)
  - `precision` — [`L31`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L31)
  - `weights_dtype` — [`L30`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L30)
- protocol/private: `__call__`[`L34`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L34)
- used by: [`norm_out`](flux/transformers/transformer_flux_flax.md#FluxTransformer2DModel.norm_out)

### `AdaLayerNormZero`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/normalization_flax.py:53`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L53)
- doc: Norm layer adaptive layer norm zero (adaLN-Zero).
- signature: `class AdaLayerNormZero(nn.Module):`
- members:
  - `bias` — [`L64`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L64)
  - `dtype` — [`L65`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L65)
  - `embedding_dim` — [`L62`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L62)
  - `norm_type` — [`L63`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L63)
  - `precision` — [`L67`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L67)
  - `weights_dtype` — [`L66`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L66)
- protocol/private: `__call__`[`L70`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L70)
- used by: [`img_norm1`](flux/transformers/transformer_flux_flax.md#FluxTransformerBlock.img_norm1), [`txt_norm1`](flux/transformers/transformer_flux_flax.md#FluxTransformerBlock.txt_norm1)

### `AdaLayerNormZeroSingle`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/normalization_flax.py:103`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L103)
- doc: Norm layer adaptive layer norm zero (adaLN-Zero).
- signature: `class AdaLayerNormZeroSingle(nn.Module):`
- members:
  - `bias` — [`L114`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L114)
  - `dtype` — [`L115`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L115)
  - `embedding_dim` — [`L112`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L112)
  - `norm_type` — [`L113`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L113)
  - `precision` — [`L117`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L117)
  - `weights_dtype` — [`L116`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L116)
- protocol/private: `__call__`[`L120`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L120)
- used by: [`norm`](flux/transformers/transformer_flux_flax.md#FluxSingleTransformerBlock.norm)

### `FP32LayerNorm`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/normalization_flax.py:153`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L153) — documented in [maxdiffusion-models-embeddings_flax](../../../../concepts/maxdiffusion-models-embeddings_flax.md)
- signature: `class FP32LayerNorm(nnx.Module):`
- members:
  - `layer_norm` — [`L156`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L156)
- protocol/private: `__call__`[`L166`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L166), `__init__`[`L155`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/normalization_flax.py#L155)
- used by: [`norm1`](embeddings_flax.md#NNXWanImageEmbedding.norm1), [`norm1`](wan/transformers/transformer_wan_vace.md#WanVACETransformerBlock.norm1), [`norm1`](wan/transformers/transformer_wan.md#WanTransformerBlock.norm1), [`norm2`](embeddings_flax.md#NNXWanImageEmbedding.norm2), [`norm2`](wan/transformers/transformer_wan_vace.md#WanVACETransformerBlock.norm2), [`norm2`](wan/transformers/transformer_wan.md#WanTransformerBlock.norm2), [`norm3`](wan/transformers/transformer_wan_vace.md#WanVACETransformerBlock.norm3), [`norm3`](wan/transformers/transformer_wan.md#WanTransformerBlock.norm3), [`norm_out`](wan/transformers/transformer_wan_animate.md#WanAnimateTransformer3DModel.norm_out), [`norm_out`](wan/transformers/transformer_wan_vace.md#WanVACEModel.norm_out), [`norm_out`](wan/transformers/transformer_wan.md#WanModel.norm_out)

