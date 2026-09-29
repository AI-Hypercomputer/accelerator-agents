---
title: 'Module: src/maxdiffusion/models/controlnet_flax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/controlnet_flax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.controlnet_flax`/FlaxControlNet
symbols:
  FlaxControlNetModel.controlnet_cond_embedding: Model#controlnet_cond_embedding.
  FlaxControlNetModel.__call__: Model#__call__().
  FlaxControlNetModel.mid_block: Model#mid_block.
  FlaxControlNetModel.init_weights: Model#init_weights().
  FlaxControlNetModel.add_time_proj: Model#add_time_proj.
  FlaxControlNetModel: Model#
  FlaxControlNetModel.time_proj: Model#time_proj.
  FlaxControlNetModel.time_embedding: Model#time_embedding.
  FlaxControlNetModel.dtype: Model#dtype.
  FlaxControlNetModel.add_embedding: Model#add_embedding.
  FlaxControlNetConditioningEmbedding.__call__: ConditioningEmbedding#__call__().
  FlaxControlNetModel.setup: Model#setup().
  FlaxControlNetConditioningEmbedding.conv_in: ConditioningEmbedding#conv_in.
  FlaxControlNetConditioningEmbedding.conv_out: ConditioningEmbedding#conv_out.
  FlaxControlNetConditioningEmbedding.block_out_channels: ConditioningEmbedding#block_out_channels.
  FlaxControlNetOutput: Output#
  FlaxControlNetConditioningEmbedding.dtype: ConditioningEmbedding#dtype.
  FlaxControlNetModel.sample_size: Model#sample_size.
  FlaxControlNetModel.conv_in: Model#conv_in.
  FlaxControlNetModel.controlnet_mid_block: Model#controlnet_mid_block.
  FlaxControlNetModel.down_block_types: Model#down_block_types.
  FlaxControlNetModel.layers_per_block: Model#layers_per_block.
  FlaxControlNetModel.dropout: Model#dropout.
  FlaxControlNetModel.addition_embed_type: Model#addition_embed_type.
  FlaxControlNetConditioningEmbedding.conditioning_embedding_channels: ConditioningEmbedding#conditioning_embedding_channels.
  FlaxControlNetModel.use_linear_projection: Model#use_linear_projection.
  FlaxControlNetModel.flip_sin_to_cos: Model#flip_sin_to_cos.
  FlaxControlNetModel.addition_time_embed_dim: Model#addition_time_embed_dim.
  FlaxControlNetConditioningEmbedding: ConditioningEmbedding#
  FlaxControlNetConditioningEmbedding.blocks: ConditioningEmbedding#blocks.
  FlaxControlNetModel.in_channels: Model#in_channels.
  FlaxControlNetModel.only_cross_attention: Model#only_cross_attention.
  FlaxControlNetModel.block_out_channels: Model#block_out_channels.
  FlaxControlNetModel.attention_head_dim: Model#attention_head_dim.
  FlaxControlNetModel.num_attention_heads: Model#num_attention_heads.
  FlaxControlNetModel.cross_attention_dim: Model#cross_attention_dim.
  FlaxControlNetModel.controlnet_conditioning_channel_order: Model#controlnet_conditioning_channel_order.
  FlaxControlNetModel.conditioning_embedding_out_channels: Model#conditioning_embedding_out_channels.
  FlaxControlNetModel.projection_class_embeddings_input_dim: Model#projection_class_embeddings_input_dim.
  FlaxControlNetModel.down_blocks: Model#down_blocks.
  FlaxControlNetModel.controlnet_down_blocks: Model#controlnet_down_blocks.
  FlaxControlNetOutput.down_block_res_samples: Output#down_block_res_samples.
  FlaxControlNetOutput.mid_block_res_sample: Output#mid_block_res_sample.
  FlaxControlNetConditioningEmbedding.setup: ConditioningEmbedding#setup().
  FlaxControlNetModel.freq_shift: Model#freq_shift.
---
# Module: [`src/maxdiffusion/models/controlnet_flax.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py)

## Classes
### `FlaxControlNetConditioningEmbedding`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/controlnet_flax.py:43`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L43)
- signature: `class FlaxControlNetConditioningEmbedding(nn.Module):`
- members:
  - `setup(self)` — [`L48`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L48)
  - `block_out_channels` — [`L45`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L45)
  - `blocks` — [`L75`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L75)
  - `conditioning_embedding_channels` — [`L44`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L44)
  - `conv_in` — [`L49`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L49)
  - `conv_out` — [`L77`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L77)
  - `dtype` — [`L46`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L46)
- protocol/private: `__call__`[`L86`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L86)
- used by: [`controlnet_cond_embedding`](controlnet_flax.md#FlaxControlNetModel.controlnet_cond_embedding)

### `FlaxControlNetModel`  ·  implements/extends ConfigMixin, FlaxModelMixin, Module
- def: [`src/maxdiffusion/models/controlnet_flax.py:100`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L100)
- doc: A ControlNet model.
- signature: `class FlaxControlNetModel(nn.Module, FlaxModelMixin, ConfigMixin):`
- members:
  - `__call__(self, sample, timesteps, encoder_hidden_states, controlnet_cond, conditioning_scale: float = 1, return_dict: bool = True, train: bool = False, added_cond_kwargs=None)` — [`L341`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L341) — Args:
  - `init_weights(self, rng: jax.Array, eval_only=False)` — [`L171`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L171)
  - `setup(self)` — [`L208`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L208)
  - `add_embedding` — [`L238`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L238)
  - `add_time_proj` — [`L235`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L235)
  - `addition_embed_type` — [`L167`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L167)
  - `addition_time_embed_dim` — [`L168`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L168)
  - `attention_head_dim` — [`L157`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L157)
  - `block_out_channels` — [`L155`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L155)
  - `conditioning_embedding_out_channels` — [`L166`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L166)
  - `controlnet_cond_embedding` — [`L240`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L240)
  - `controlnet_conditioning_channel_order` — [`L165`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L165)
  - `controlnet_down_blocks` — [`L320`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L320)
  - `controlnet_mid_block` — [`L332`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L332)
  - `conv_in` — [`L221`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L221)
  - `cross_attention_dim` — [`L159`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L159)
  - `down_block_types` — [`L148`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L148)
  - `down_blocks` — [`L319`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L319)
  - `dropout` — [`L160`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L160)
  - `dtype` — [`L162`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L162)
  - `flip_sin_to_cos` — [`L163`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L163)
  - `freq_shift` — [`L164`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L164)
  - `in_channels` — [`L147`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L147)
  - `layers_per_block` — [`L156`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L156)
  - `mid_block` — [`L324`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L324)
  - `num_attention_heads` — [`L158`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L158)
  - `only_cross_attention` — [`L154`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L154)
  - `projection_class_embeddings_input_dim` — [`L169`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L169)
  - `sample_size` — [`L146`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L146)
  - `time_embedding` — [`L233`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L233)
  - `time_proj` — [`L230`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L230)
  - `use_linear_projection` — [`L161`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L161)
- uses (calls/refs, reference-scoped): [`config`](../configuration_utils.md#ConfigMixin.config), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`FlaxModelMixin`](modeling_flax_utils.md#FlaxModelMixin), [`flax_register_to_config`](../configuration_utils.md#flax_register_to_config), [`FlaxTimestepEmbedding`](embeddings_flax.md#FlaxTimestepEmbedding), [`dtype`](embeddings_flax.md#FlaxTimestepEmbedding.dtype), [`in_channels`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn.in_channels), [`out_channels`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.out_channels), [`FlaxCrossAttnDownBlock2D`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D), [`FlaxTimesteps`](embeddings_flax.md#FlaxTimesteps), [`block_out_channels`](controlnet_flax.md#FlaxControlNetConditioningEmbedding.block_out_channels), [`dtype`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.dtype), [`dtype`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn.dtype), [`out_channels`](unet_2d_blocks_flax.md#FlaxDownBlock2D.out_channels), [`FlaxControlNetOutput`](controlnet_flax.md#FlaxControlNetOutput), [`FlaxDownBlock2D`](unet_2d_blocks_flax.md#FlaxDownBlock2D), [`FlaxUNetMidBlock2DCrossAttn`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn), [`add_downsample`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.add_downsample), [`add_downsample`](unet_2d_blocks_flax.md#FlaxDownBlock2D.add_downsample), [`dropout`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn.dropout), [`dtype`](unet_2d_blocks_flax.md#FlaxDownBlock2D.dtype), [`flip_sin_to_cos`](embeddings_flax.md#FlaxTimesteps.flip_sin_to_cos), [`freq_shift`](embeddings_flax.md#FlaxTimesteps.freq_shift), [`num_attention_heads`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.num_attention_heads), [`num_attention_heads`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn.num_attention_heads), [`dropout`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.dropout), [`dropout`](unet_2d_blocks_flax.md#FlaxDownBlock2D.dropout), [`in_channels`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.in_channels), [`in_channels`](unet_2d_blocks_flax.md#FlaxDownBlock2D.in_channels), [`num_layers`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.num_layers), [`num_layers`](unet_2d_blocks_flax.md#FlaxDownBlock2D.num_layers), [`only_cross_attention`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.only_cross_attention), [`use_linear_projection`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.use_linear_projection), [`use_linear_projection`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn.use_linear_projection), [`conditioning_embedding_channels`](controlnet_flax.md#FlaxControlNetConditioningEmbedding.conditioning_embedding_channels), [`FlaxControlNetConditioningEmbedding`](controlnet_flax.md#FlaxControlNetConditioningEmbedding)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`FlaxModelMixin`](modeling_flax_utils.md#FlaxModelMixin), [`_import_structure`](__init__.md#_import_structure), [`__init__`](../pipelines/controlnet/pipeline_flax_controlnet_sdxl.md#FlaxStableDiffusionXLControlNetPipeline.__init__), [`__init__`](../pipelines/controlnet/pipeline_flax_controlnet.md#FlaxStableDiffusionControlNetPipeline.__init__), [`init_weights`](modeling_flax_utils.md#FlaxModelMixin.init_weights)

### `FlaxControlNetOutput`  ·  implements/extends BaseOutput
- def: [`src/maxdiffusion/models/controlnet_flax.py:30`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L30)
- doc: The output of \[`FlaxControlNetModel`\].
- signature: `class FlaxControlNetOutput(BaseOutput):`
- members:
  - `down_block_res_samples` — [`L39`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L39)
  - `mid_block_res_sample` — [`L40`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/controlnet_flax.py#L40)
- uses (calls/refs, reference-scoped): [`BaseOutput`](../utils/outputs.md#BaseOutput)
- used by: [`BaseOutput`](../utils/outputs.md#BaseOutput), [`__call__`](controlnet_flax.md#FlaxControlNetModel.__call__)

