---
title: 'Module: simply/utils/vision_encoder.py'
type: catalog
provenance: extracted
module: simply/utils/vision_encoder.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.vision_encoder`/
symbols:
  VisionTransformer.apply: VisionTransformer#apply().
  PatchEncoder.init: PatchEncoder#init().
  PatchEncoder.apply: PatchEncoder#apply().
  VisionTransformer.patch_encoder: VisionTransformer#patch_encoder.
  VisionTransformer.transformer_blocks: VisionTransformer#transformer_blocks.
  VisionTransformer.init: VisionTransformer#init().
  Array: Array.
  AnnotatedArray: AnnotatedArray.
  VisionTransformer.final_projection: VisionTransformer#final_projection.
  VisionTransformer: VisionTransformer#
  PatchEncoder.weight_init: PatchEncoder#weight_init.
  VisionTransformer.width: VisionTransformer#width.
  VisionTransformer.pre_downsample_norm: VisionTransformer#pre_downsample_norm.
  VisionTransformer.post_downsample_norm: VisionTransformer#post_downsample_norm.
  PyTree: PyTree.
  PatchEncoder.activation_dtype: PatchEncoder#activation_dtype.
  PatchEncoder.output_grid_shape: PatchEncoder#output_grid_shape().
  PatchEncoder.weight_dtype: PatchEncoder#weight_dtype.
  PatchEncoder.seq_len: PatchEncoder#seq_len().
  PatchEncoder.encoding_dim: PatchEncoder#encoding_dim.
  PatchEncoder: PatchEncoder#
  PatchEncoder._normalize_image: PatchEncoder#_normalize_image().
  VisionTransformer.sharding_config: VisionTransformer#sharding_config.
  VisionTransformer.downsample: VisionTransformer#downsample().
  PatchEncoder.patch_size: PatchEncoder#patch_size.
  VisionTransformer.num_heads: VisionTransformer#num_heads.
  PRNGKey: PRNGKey.
  DTypeLike: DTypeLike.
  PatchEncoder.image_shape: PatchEncoder#image_shape.
  VisionTransformer.image_shape: VisionTransformer#image_shape.
  VisionTransformer.patch_size: VisionTransformer#patch_size.
  VisionTransformer.depth: VisionTransformer#depth.
  VisionTransformer.mlp_dim: VisionTransformer#mlp_dim.
  VisionTransformer.output_patch_dims: VisionTransformer#output_patch_dims.
  VisionTransformer.output_embedding_dim: VisionTransformer#output_embedding_dim.
  VisionTransformer.setup: VisionTransformer#setup().
---
# Module: [`simply/utils/vision_encoder.py`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py)

## Classes
### `PatchEncoder`
- def: [`simply/utils/vision_encoder.py:38`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L38)
- doc: Patch encoding layer for vision models.
- signature: `class PatchEncoder(module.SimplyModule):`
- members:
  - `apply(self, params: PyTree, image: Array)` — [`L95`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L95)
  - `init(self, prng_key: PRNGKey)` — [`L61`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L61)
  - `output_grid_shape(self)` — [`L49`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L49)
  - `seq_len(self)` — [`L57`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L57)
  - `activation_dtype` — [`L46`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L46)
  - `encoding_dim` — [`L43`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L43)
  - `image_shape` — [`L41`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L41)
  - `patch_size` — [`L42`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L42)
  - `weight_dtype` — [`L45`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L45)
  - `weight_init` — [`L44`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L44)
- protocol/private: `_normalize_image`[`L91`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L91)
- uses (calls/refs, reference-scoped): [`get_raw_arrays`](common.md#get_raw_arrays), [`SimplyModule`](module.md#SimplyModule), [`convert_or_dequantize`](common.md#convert_or_dequantize), [`Initializer`](initializer.md#Initializer), [`create`](common.md#AnnotatedArray.create), [`XavierUniformInit`](initializer.md#XavierUniformInit), [`AnnotatedArray`](vision_encoder.md#AnnotatedArray), [`Array`](vision_encoder.md#Array), [`PyTree`](vision_encoder.md#PyTree), [`DTypeLike`](vision_encoder.md#DTypeLike), [`PRNGKey`](vision_encoder.md#PRNGKey)
- used by: [`apply`](module.md#SimplyModule.apply), [`init`](module.md#SimplyModule.init), [`patch_encoder`](vision_encoder.md#VisionTransformer.patch_encoder)

### `VisionTransformer`
- def: [`simply/utils/vision_encoder.py:124`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L124)
- doc: Simply implementation of vision encoder.
- signature: `class VisionTransformer(module.SimplyModule):`
- members:
  - `apply(self, params: PyTree, image: Array)` — [`L235`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L235)
  - `downsample(self, patch_embeddings: Array, input_patch_dims: tuple[int, int], output_patch_dims: tuple[int, int])` — [`L206`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L206)
  - `init(self, prng_key: PRNGKey)` — [`L191`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L191)
  - `setup(self)` — [`L143`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L143)
  - `depth` — [`L136`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L136)
  - `final_projection` — [`L185`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L185)
  - `image_shape` — [`L133`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L133)
  - `mlp_dim` — [`L137`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L137)
  - `num_heads` — [`L138`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L138)
  - `output_embedding_dim` — [`L140`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L140)
  - `output_patch_dims` — [`L139`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L139)
  - `patch_encoder` — [`L144`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L144)
  - `patch_size` — [`L134`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L134)
  - `post_downsample_norm` — [`L177`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L177)
  - `pre_downsample_norm` — [`L170`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L170)
  - `sharding_config` — [`L141`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L141)
  - `transformer_blocks` — [`L151`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L151)
  - `width` — [`L135`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L135)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`SimplyModule`](module.md#SimplyModule), [`ModuleRegistry`](module.md#ModuleRegistry), [`LayerNorm`](../model_lib.md#LayerNorm), [`gspmd_sharding`](../config_lib.md#gspmd_sharding), [`Array`](vision_encoder.md#Array), [`EinsumLinear`](../model_lib.md#EinsumLinear), [`TransformerBlock`](../model_lib.md#TransformerBlock), [`SimplyConfig`](../model_lib.md#SimplyConfig), [`PyTree`](vision_encoder.md#PyTree), [`PatchEncoder`](vision_encoder.md#PatchEncoder), [`PRNGKey`](vision_encoder.md#PRNGKey)
- used by: [`apply`](module.md#SimplyModule.apply), [`init`](module.md#SimplyModule.init), [`setup`](module.md#SimplyModule.setup)

## Module values
- `AnnotatedArray` — [`L33`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L33)
- `Array` — [`L30`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L30)
- `DTypeLike` — [`L32`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L32)
- `PRNGKey` — [`L31`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L31)
- `PyTree` — [`L29`](../../../../../../raw/code/simply/simply/utils/vision_encoder.py#L29)

