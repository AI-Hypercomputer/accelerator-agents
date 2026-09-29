---
title: 'Module: src/maxdiffusion/models/unet_2d_condition_flax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/unet_2d_condition_flax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.unet_2d_condition_flax`/
symbols:
  FlaxUNet2DConditionModel.mid_block: FlaxUNet2DConditionModel#mid_block.
  FlaxUNet2DConditionModel.add_time_proj: FlaxUNet2DConditionModel#add_time_proj.
  FlaxUNet2DConditionModel.__call__: FlaxUNet2DConditionModel#__call__().
  FlaxUNet2DConditionModel.time_embedding: FlaxUNet2DConditionModel#time_embedding.
  FlaxUNet2DConditionModel.init_weights: FlaxUNet2DConditionModel#init_weights().
  FlaxUNet2DConditionModel: FlaxUNet2DConditionModel#
  FlaxUNet2DConditionModel.time_proj: FlaxUNet2DConditionModel#time_proj.
  FlaxUNet2DConditionModel.dtype: FlaxUNet2DConditionModel#dtype.
  FlaxUNet2DConditionModel.weights_dtype: FlaxUNet2DConditionModel#weights_dtype.
  FlaxUNet2DConditionModel.conv_in: FlaxUNet2DConditionModel#conv_in.
  FlaxUNet2DConditionModel.conv_norm_out: FlaxUNet2DConditionModel#conv_norm_out.
  FlaxUNet2DConditionModel.conv_out: FlaxUNet2DConditionModel#conv_out.
  FlaxUNet2DConditionModel.setup: FlaxUNet2DConditionModel#setup().
  FlaxUNet2DConditionModel.addition_embed_type: FlaxUNet2DConditionModel#addition_embed_type.
  FlaxUNet2DConditionModel.precision: FlaxUNet2DConditionModel#precision.
  FlaxUNet2DConditionModel.add_embedding: FlaxUNet2DConditionModel#add_embedding.
  FlaxUNet2DConditionModel.layers_per_block: FlaxUNet2DConditionModel#layers_per_block.
  FlaxUNet2DConditionModel.flash_block_sizes: FlaxUNet2DConditionModel#flash_block_sizes.
  FlaxUNet2DConditionModel.quant: FlaxUNet2DConditionModel#quant.
  FlaxUNet2DConditionModel.dropout: FlaxUNet2DConditionModel#dropout.
  FlaxUNet2DConditionOutput: FlaxUNet2DConditionOutput#
  FlaxUNet2DConditionModel.down_block_types: FlaxUNet2DConditionModel#down_block_types.
  Quant: Quant.
  FlaxUNet2DConditionModel.use_linear_projection: FlaxUNet2DConditionModel#use_linear_projection.
  FlaxUNet2DConditionModel.use_memory_efficient_attention: FlaxUNet2DConditionModel#use_memory_efficient_attention.
  FlaxUNet2DConditionModel.split_head_dim: FlaxUNet2DConditionModel#split_head_dim.
  FlaxUNet2DConditionModel.attention_kernel: FlaxUNet2DConditionModel#attention_kernel.
  FlaxUNet2DConditionModel.flash_min_seq_length: FlaxUNet2DConditionModel#flash_min_seq_length.
  FlaxUNet2DConditionModel.mesh: FlaxUNet2DConditionModel#mesh.
  FlaxUNet2DConditionModel.addition_time_embed_dim: FlaxUNet2DConditionModel#addition_time_embed_dim.
  FlaxUNet2DConditionOutput.sample: FlaxUNet2DConditionOutput#sample.
  FlaxUNet2DConditionModel.sample_size: FlaxUNet2DConditionModel#sample_size.
  FlaxUNet2DConditionModel.num_attention_heads: FlaxUNet2DConditionModel#num_attention_heads.
  FlaxUNet2DConditionModel.cross_attention_dim: FlaxUNet2DConditionModel#cross_attention_dim.
  FlaxUNet2DConditionModel.flip_sin_to_cos: FlaxUNet2DConditionModel#flip_sin_to_cos.
  FlaxUNet2DConditionModel.in_channels: FlaxUNet2DConditionModel#in_channels.
  FlaxUNet2DConditionModel.out_channels: FlaxUNet2DConditionModel#out_channels.
  FlaxUNet2DConditionModel.up_block_types: FlaxUNet2DConditionModel#up_block_types.
  FlaxUNet2DConditionModel.only_cross_attention: FlaxUNet2DConditionModel#only_cross_attention.
  FlaxUNet2DConditionModel.block_out_channels: FlaxUNet2DConditionModel#block_out_channels.
  FlaxUNet2DConditionModel.attention_head_dim: FlaxUNet2DConditionModel#attention_head_dim.
  FlaxUNet2DConditionModel.freq_shift: FlaxUNet2DConditionModel#freq_shift.
  FlaxUNet2DConditionModel.transformer_layers_per_block: FlaxUNet2DConditionModel#transformer_layers_per_block.
  FlaxUNet2DConditionModel.projection_class_embeddings_input_dim: FlaxUNet2DConditionModel#projection_class_embeddings_input_dim.
  FlaxUNet2DConditionModel.norm_num_groups: FlaxUNet2DConditionModel#norm_num_groups.
  FlaxUNet2DConditionModel.down_blocks: FlaxUNet2DConditionModel#down_blocks.
  FlaxUNet2DConditionModel.up_blocks: FlaxUNet2DConditionModel#up_blocks.
  FlaxUNet2DConditionModel.addition_embed_type_num_heads: FlaxUNet2DConditionModel#addition_embed_type_num_heads.
---
# Module: [`src/maxdiffusion/models/unet_2d_condition_flax.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py)

## Classes
### `FlaxUNet2DConditionModel`  ·  implements/extends ConfigMixin, FlaxModelMixin, Module
- def: [`src/maxdiffusion/models/unet_2d_condition_flax.py:54`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L54)
- doc: A conditional 2D UNet model that takes a noisy sample, conditional state, and a timestep and returns a sample
- signature: `class FlaxUNet2DConditionModel(nn.Module, FlaxModelMixin, ConfigMixin):`
- members:
  - `__call__(self, sample, timesteps, encoder_hidden_states, added_cond_kwargs: Optional[Union[Dict, FrozenDict]] = None, down_block_additional_residuals=None, mid_block_additional_residual=None, return_dict: bool = True, train: bool = False, cross_attention_kwargs: Optional[Union[Dict, FrozenDict]] = None)` — [`L396`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L396) — Args:
  - `init_weights(self, rng: jax.Array, eval_only: bool = False, quantization_enabled: bool = False)` — [`L152`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L152)
  - `setup(self)` — [`L203`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L203)
  - `add_embedding` — [`L254`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L254)
  - `add_time_proj` — [`L258`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L258)
  - `addition_embed_type` — [`L144`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L144)
  - `addition_embed_type_num_heads` — [`L146`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L146)
  - `addition_time_embed_dim` — [`L145`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L145)
  - `attention_head_dim` — [`L128`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L128)
  - `attention_kernel` — [`L139`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L139)
  - `block_out_channels` — [`L126`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L126)
  - `conv_in` — [`L221`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L221)
  - `conv_norm_out` — [`L382`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L382)
  - `conv_out` — [`L386`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L386)
  - `cross_attention_dim` — [`L130`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L130)
  - `down_block_types` — [`L118`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L118)
  - `down_blocks` — [`L306`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L306)
  - `dropout` — [`L131`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L131)
  - `dtype` — [`L133`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L133)
  - `flash_block_sizes` — [`L141`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L141)
  - `flash_min_seq_length` — [`L140`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L140)
  - `flip_sin_to_cos` — [`L135`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L135)
  - `freq_shift` — [`L136`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L136)
  - `in_channels` — [`L116`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L116)
  - `layers_per_block` — [`L127`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L127)
  - `mesh` — [`L142`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L142)
  - `mid_block` — [`L309`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L309)
  - `norm_num_groups` — [`L148`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L148)
  - `num_attention_heads` — [`L129`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L129)
  - `only_cross_attention` — [`L125`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L125)
  - `out_channels` — [`L117`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L117)
  - `precision` — [`L149`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L149)
  - `projection_class_embeddings_input_dim` — [`L147`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L147)
  - `quant` — [`L150`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L150)
  - `sample_size` — [`L115`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L115)
  - `split_head_dim` — [`L138`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L138)
  - `time_embedding` — [`L238`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L238)
  - `time_proj` — [`L235`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L235)
  - `transformer_layers_per_block` — [`L143`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L143)
  - `up_block_types` — [`L124`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L124)
  - `up_blocks` — [`L379`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L379)
  - `use_linear_projection` — [`L132`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L132)
  - `use_memory_efficient_attention` — [`L137`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L137)
  - `weights_dtype` — [`L134`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L134)
- uses (calls/refs, reference-scoped): [`config`](../configuration_utils.md#ConfigMixin.config), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`FlaxModelMixin`](modeling_flax_utils.md#FlaxModelMixin), [`BlockSizes`](../common_types.md#BlockSizes), [`FlaxTimestepEmbedding`](embeddings_flax.md#FlaxTimestepEmbedding), [`dtype`](embeddings_flax.md#FlaxTimestepEmbedding.dtype), [`in_channels`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn.in_channels), [`out_channels`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.out_channels), [`out_channels`](unet_2d_blocks_flax.md#FlaxCrossAttnUpBlock2D.out_channels), [`weights_dtype`](embeddings_flax.md#FlaxTimestepEmbedding.weights_dtype), [`FlaxCrossAttnDownBlock2D`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D), [`FlaxTimesteps`](embeddings_flax.md#FlaxTimesteps), [`dtype`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.dtype), [`dtype`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn.dtype), [`out_channels`](unet_2d_blocks_flax.md#FlaxDownBlock2D.out_channels), [`out_channels`](unet_2d_blocks_flax.md#FlaxUpBlock2D.out_channels), [`FlaxUNet2DConditionOutput`](unet_2d_condition_flax.md#FlaxUNet2DConditionOutput), [`flash_block_sizes`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.flash_block_sizes), [`flash_block_sizes`](unet_2d_blocks_flax.md#FlaxCrossAttnUpBlock2D.flash_block_sizes), [`flash_block_sizes`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn.flash_block_sizes), [`quant`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.quant), [`quant`](unet_2d_blocks_flax.md#FlaxCrossAttnUpBlock2D.quant), [`quant`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn.quant), [`FlaxDownBlock2D`](unet_2d_blocks_flax.md#FlaxDownBlock2D), [`FlaxUNetMidBlock2DCrossAttn`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn), [`add_downsample`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.add_downsample), [`add_downsample`](unet_2d_blocks_flax.md#FlaxDownBlock2D.add_downsample), [`dropout`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn.dropout), [`dtype`](unet_2d_blocks_flax.md#FlaxCrossAttnUpBlock2D.dtype), [`dtype`](unet_2d_blocks_flax.md#FlaxDownBlock2D.dtype), [`flip_sin_to_cos`](embeddings_flax.md#FlaxTimesteps.flip_sin_to_cos), [`freq_shift`](embeddings_flax.md#FlaxTimesteps.freq_shift), [`num_attention_heads`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.num_attention_heads), [`num_attention_heads`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn.num_attention_heads), [`precision`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn.precision), [`weights_dtype`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.weights_dtype), [`weights_dtype`](unet_2d_blocks_flax.md#FlaxCrossAttnUpBlock2D.weights_dtype), [`weights_dtype`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn.weights_dtype), [`Quant`](unet_2d_condition_flax.md#Quant), [`FlaxCrossAttnUpBlock2D`](unet_2d_blocks_flax.md#FlaxCrossAttnUpBlock2D)  (+48 more)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`FlaxModelMixin`](modeling_flax_utils.md#FlaxModelMixin), [`_import_structure`](__init__.md#_import_structure), [`__init__`](../pipelines/controlnet/pipeline_flax_controlnet_sdxl.md#FlaxStableDiffusionXLControlNetPipeline.__init__), [`__init__`](../pipelines/controlnet/pipeline_flax_controlnet.md#FlaxStableDiffusionControlNetPipeline.__init__), [`__init__`](../pipelines/stable_diffusion/pipeline_flax_stable_diffusion.md#FlaxStableDiffusionPipeline.__init__), [`__init__`](../pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.md#FlaxStableDiffusionXLPipeline.__init__), [`init_weights`](modeling_flax_utils.md#FlaxModelMixin.init_weights)

### `FlaxUNet2DConditionOutput`  ·  implements/extends BaseOutput
- def: [`src/maxdiffusion/models/unet_2d_condition_flax.py:41`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L41)
- doc: The output of \[`FlaxUNet2DConditionModel`\].
- signature: `class FlaxUNet2DConditionOutput(BaseOutput):`
- members:
  - `sample` — [`L50`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L50)
- uses (calls/refs, reference-scoped): [`BaseOutput`](../utils/outputs.md#BaseOutput), [`flax_register_to_config`](../configuration_utils.md#flax_register_to_config)
- used by: [`BaseOutput`](../utils/outputs.md#BaseOutput), [`__call__`](unet_2d_condition_flax.md#FlaxUNet2DConditionModel.__call__)

## Module values
- `Quant` — [`L37`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py#L37)

