---
title: 'Module: src/maxdiffusion/models/ltx2/transformer_ltx2.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx2/transformer_ltx2.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx2.transformer_ltx2`/LTX2
symbols:
  LTX2VideoTransformer3DModel.__call__: VideoTransformer3DModel#__call__().
  LTX2VideoTransformer3DModel.transformer_blocks: VideoTransformer3DModel#transformer_blocks.
  LTX2VideoTransformer3DModel.init_block: VideoTransformer3DModel#init_block().
  LTX2VideoTransformerBlock.__call__: VideoTransformerBlock#__call__().
  LTX2VideoTransformer3DModel.sharding_specs: VideoTransformer3DModel#sharding_specs.
  LTX2VideoTransformer3DModel.audio_rope: VideoTransformer3DModel#audio_rope.
  LTX2VideoTransformer3DModel.rope: VideoTransformer3DModel#rope.
  LTX2VideoTransformer3DModel.cross_attn_audio_rope: VideoTransformer3DModel#cross_attn_audio_rope.
  LTX2VideoTransformer3DModel.cross_attn_rope: VideoTransformer3DModel#cross_attn_rope.
  LTX2VideoTransformer3DModel.audio_caption_projection: VideoTransformer3DModel#audio_caption_projection.
  LTX2VideoTransformer3DModel.dtype: VideoTransformer3DModel#dtype.
  LTX2VideoTransformer3DModel.audio_proj_in: VideoTransformer3DModel#audio_proj_in.
  LTX2VideoTransformer3DModel.weights_dtype: VideoTransformer3DModel#weights_dtype.
  LTX2VideoTransformerBlock.sharding_specs: VideoTransformerBlock#sharding_specs.
  LTX2VideoTransformer3DModel.proj_in: VideoTransformer3DModel#proj_in.
  LTX2VideoTransformerBlock.norm1: VideoTransformerBlock#norm1.
  LTX2VideoTransformerBlock.norm2: VideoTransformerBlock#norm2.
  LTX2VideoTransformer3DModel.audio_prompt_adaln: VideoTransformer3DModel#audio_prompt_adaln.
  LTX2VideoTransformer3DModel.av_cross_attn_audio_v2a_gate: VideoTransformer3DModel#av_cross_attn_audio_v2a_gate.
  LTX2VideoTransformer3DModel.proj_out: VideoTransformer3DModel#proj_out.
  LTX2VideoTransformer3DModel.audio_proj_out: VideoTransformer3DModel#audio_proj_out.
  LTX2VideoTransformer3DModel.caption_projection: VideoTransformer3DModel#caption_projection.
  LTX2VideoTransformerBlock.audio_norm1: VideoTransformerBlock#audio_norm1.
  LTX2VideoTransformerBlock.audio_norm2: VideoTransformerBlock#audio_norm2.
  LTX2VideoTransformerBlock.audio_to_video_norm: VideoTransformerBlock#audio_to_video_norm.
  LTX2VideoTransformerBlock.video_to_audio_norm: VideoTransformerBlock#video_to_audio_norm.
  LTX2VideoTransformerBlock.norm3: VideoTransformerBlock#norm3.
  LTX2VideoTransformerBlock.audio_norm3: VideoTransformerBlock#audio_norm3.
  LTX2VideoTransformer3DModel.prompt_adaln: VideoTransformer3DModel#prompt_adaln.
  LTX2VideoTransformer3DModel.time_embed: VideoTransformer3DModel#time_embed.
  LTX2VideoTransformer3DModel.audio_time_embed: VideoTransformer3DModel#audio_time_embed.
  LTX2VideoTransformer3DModel.av_cross_attn_video_scale_shift: VideoTransformer3DModel#av_cross_attn_video_scale_shift.
  LTX2VideoTransformer3DModel.av_cross_attn_audio_scale_shift: VideoTransformer3DModel#av_cross_attn_audio_scale_shift.
  LTX2VideoTransformer3DModel.av_cross_attn_video_a2v_gate: VideoTransformer3DModel#av_cross_attn_video_a2v_gate.
  LTX2AdaLayerNormSingle.__call__: AdaLayerNormSingle#__call__().
  LTX2VideoTransformerBlock.attn1: VideoTransformerBlock#attn1.
  LTX2AdaLayerNormSingle: AdaLayerNormSingle#
  LTX2VideoTransformerBlock.dim: VideoTransformerBlock#dim.
  LTX2VideoTransformerBlock.norm_eps: VideoTransformerBlock#norm_eps.
  LTX2VideoTransformerBlock.norm_elementwise_affine: VideoTransformerBlock#norm_elementwise_affine.
  LTX2VideoTransformerBlock.cross_attn_mod: VideoTransformerBlock#cross_attn_mod.
  LTX2VideoTransformerBlock.scale_shift_table: VideoTransformerBlock#scale_shift_table.
  LTX2VideoTransformer3DModel.num_layers: VideoTransformer3DModel#num_layers.
  LTX2AdaLayerNormSingle.linear: AdaLayerNormSingle#linear.
  LTX2VideoTransformerBlock.audio_attn1: VideoTransformerBlock#audio_attn1.
  LTX2VideoTransformerBlock.attn2: VideoTransformerBlock#attn2.
  LTX2VideoTransformerBlock.audio_attn2: VideoTransformerBlock#audio_attn2.
  LTX2VideoTransformer3DModel.rope_type: VideoTransformer3DModel#rope_type.
  LTX2VideoTransformer3DModel.__init__: VideoTransformer3DModel#__init__().
  LTX2VideoTransformer3DModel.num_attention_heads: VideoTransformer3DModel#num_attention_heads.
  LTX2VideoTransformer3DModel.audio_num_attention_heads: VideoTransformer3DModel#audio_num_attention_heads.
  LTX2VideoTransformer3DModel.scan_layers: VideoTransformer3DModel#scan_layers.
  LTX2VideoTransformer3DModel.cross_attn_mod: VideoTransformer3DModel#cross_attn_mod.
  LTX2VideoTransformerBlock.video_a2v_cross_attn_scale_shift_table: VideoTransformerBlock#video_a2v_cross_attn_scale_shift_table.
  LTX2VideoTransformerBlock.audio_a2v_cross_attn_scale_shift_table: VideoTransformerBlock#audio_a2v_cross_attn_scale_shift_table.
  LTX2VideoTransformer3DModel.flash_min_seq_length: VideoTransformer3DModel#flash_min_seq_length.
  LTX2VideoTransformerBlock.attention_kernel: VideoTransformerBlock#attention_kernel.
  LTX2VideoTransformer3DModel.rope_theta: VideoTransformer3DModel#rope_theta.
  LTX2VideoTransformer3DModel.rope_double_precision: VideoTransformer3DModel#rope_double_precision.
  LTX2VideoTransformer3DModel.causal_offset: VideoTransformer3DModel#causal_offset.
  LTX2VideoTransformer3DModel.names_which_can_be_saved: VideoTransformer3DModel#names_which_can_be_saved.
  LTX2VideoTransformer3DModel.names_which_can_be_offloaded: VideoTransformer3DModel#names_which_can_be_offloaded.
  LTX2VideoTransformerBlock.audio_to_video_attn: VideoTransformerBlock#audio_to_video_attn.
  LTX2VideoTransformerBlock.video_to_audio_attn: VideoTransformerBlock#video_to_audio_attn.
  LTX2VideoTransformerBlock.ff: VideoTransformerBlock#ff.
  LTX2VideoTransformerBlock.audio_ff: VideoTransformerBlock#audio_ff.
  LTX2VideoTransformerBlock.prompt_scale_shift_table: VideoTransformerBlock#prompt_scale_shift_table.
  LTX2VideoTransformer3DModel.scale_shift_table: VideoTransformer3DModel#scale_shift_table.
  LTX2VideoTransformer3DModel.audio_scale_shift_table: VideoTransformer3DModel#audio_scale_shift_table.
  LTX2VideoTransformer3DModel.attention_head_dim: VideoTransformer3DModel#attention_head_dim.
  LTX2VideoTransformer3DModel.audio_attention_head_dim: VideoTransformer3DModel#audio_attention_head_dim.
  LTX2VideoTransformer3DModel.attention_kernel: VideoTransformer3DModel#attention_kernel.
  LTX2AdaLayerNormSingle.__init__: AdaLayerNormSingle#__init__().
  LTX2AdaLayerNormSingle.num_mod_params: AdaLayerNormSingle#num_mod_params.
  LTX2VideoTransformerBlock.__init__: VideoTransformerBlock#__init__().
  LTX2VideoTransformerBlock.perturbed_attn: VideoTransformerBlock#perturbed_attn.
  LTX2VideoTransformerBlock: VideoTransformerBlock#
  LTX2VideoTransformerBlock.audio_scale_shift_table: VideoTransformerBlock#audio_scale_shift_table.
  LTX2VideoTransformer3DModel.in_channels: VideoTransformer3DModel#in_channels.
  LTX2VideoTransformer3DModel.patch_size: VideoTransformer3DModel#patch_size.
  LTX2VideoTransformer3DModel.patch_size_t: VideoTransformer3DModel#patch_size_t.
  LTX2VideoTransformer3DModel.pos_embed_max_pos: VideoTransformer3DModel#pos_embed_max_pos.
  LTX2VideoTransformer3DModel.base_height: VideoTransformer3DModel#base_height.
  LTX2VideoTransformer3DModel.base_width: VideoTransformer3DModel#base_width.
  LTX2VideoTransformer3DModel.audio_in_channels: VideoTransformer3DModel#audio_in_channels.
  LTX2VideoTransformer3DModel.audio_patch_size: VideoTransformer3DModel#audio_patch_size.
  LTX2VideoTransformer3DModel.audio_patch_size_t: VideoTransformer3DModel#audio_patch_size_t.
  LTX2VideoTransformer3DModel.audio_pos_embed_max_pos: VideoTransformer3DModel#audio_pos_embed_max_pos.
  LTX2VideoTransformer3DModel.audio_sampling_rate: VideoTransformer3DModel#audio_sampling_rate.
  LTX2VideoTransformer3DModel.audio_hop_length: VideoTransformer3DModel#audio_hop_length.
  LTX2VideoTransformer3DModel.activation_fn: VideoTransformer3DModel#activation_fn.
  LTX2VideoTransformer3DModel.norm_elementwise_affine: VideoTransformer3DModel#norm_elementwise_affine.
  LTX2VideoTransformer3DModel.norm_eps: VideoTransformer3DModel#norm_eps.
  LTX2VideoTransformer3DModel.caption_channels: VideoTransformer3DModel#caption_channels.
  LTX2VideoTransformer3DModel.attention_bias: VideoTransformer3DModel#attention_bias.
  LTX2VideoTransformer3DModel.attention_out_bias: VideoTransformer3DModel#attention_out_bias.
  LTX2VideoTransformer3DModel.use_prompt_embeddings: VideoTransformer3DModel#use_prompt_embeddings.
  LTX2VideoTransformer3DModel.mesh: VideoTransformer3DModel#mesh.
  LTX2VideoTransformer3DModel.remat_policy: VideoTransformer3DModel#remat_policy.
  LTX2VideoTransformer3DModel.precision: VideoTransformer3DModel#precision.
  LTX2VideoTransformer3DModel.gated_attn: VideoTransformer3DModel#gated_attn.
  LTX2VideoTransformer3DModel.perturbed_attn: VideoTransformer3DModel#perturbed_attn.
  LTX2VideoTransformer3DModel.a2v_attention_kernel: VideoTransformer3DModel#a2v_attention_kernel.
  LTX2VideoTransformer3DModel.v2a_attention_kernel: VideoTransformer3DModel#v2a_attention_kernel.
  LTX2VideoTransformer3DModel.gradient_checkpoint: VideoTransformer3DModel#gradient_checkpoint.
  LTX2AdaLayerNormSingle.use_additional_conditions: AdaLayerNormSingle#use_additional_conditions.
  LTX2AdaLayerNormSingle.emb: AdaLayerNormSingle#emb.
  LTX2AdaLayerNormSingle.silu: AdaLayerNormSingle#silu.
  LTX2VideoTransformerBlock.audio_prompt_scale_shift_table: VideoTransformerBlock#audio_prompt_scale_shift_table.
  LTX2VideoTransformer3DModel.spatio_temporal_guidance_blocks: VideoTransformer3DModel#spatio_temporal_guidance_blocks.
  LTX2VideoTransformer3DModel.out_channels: VideoTransformer3DModel#out_channels.
  LTX2VideoTransformer3DModel.vae_scale_factors: VideoTransformer3DModel#vae_scale_factors.
  LTX2VideoTransformer3DModel.audio_out_channels: VideoTransformer3DModel#audio_out_channels.
  LTX2VideoTransformer3DModel.audio_scale_factor: VideoTransformer3DModel#audio_scale_factor.
  LTX2VideoTransformer3DModel.timestep_scale_multiplier: VideoTransformer3DModel#timestep_scale_multiplier.
  LTX2VideoTransformer3DModel.cross_attn_timestep_scale_multiplier: VideoTransformer3DModel#cross_attn_timestep_scale_multiplier.
  LTX2VideoTransformer3DModel.norm_out: VideoTransformer3DModel#norm_out.
  LTX2VideoTransformer3DModel.audio_norm_out: VideoTransformer3DModel#audio_norm_out.
  LTX2VideoTransformer3DModel.scan_fn_ltx2: VideoTransformer3DModel#scan_fn_ltx2().
  LTX2VideoTransformer3DModel.scan_fn_ltx23: VideoTransformer3DModel#scan_fn_ltx23().
  LTX2VideoTransformer3DModel: VideoTransformer3DModel#
  LTX2VideoTransformer3DModel.config_name: VideoTransformer3DModel#config_name.
  LTX2VideoTransformer3DModel.cross_attention_dim: VideoTransformer3DModel#cross_attention_dim.
  LTX2VideoTransformer3DModel.audio_cross_attention_dim: VideoTransformer3DModel#audio_cross_attention_dim.
  LTX2VideoTransformer3DModel.audio_caption_channels: VideoTransformer3DModel#audio_caption_channels.
---
# Module: [`src/maxdiffusion/models/ltx2/transformer_ltx2.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py)

## Classes
### `LTX2AdaLayerNormSingle`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx2/transformer_ltx2.py:32`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L32) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
- doc: Adaptive Layer Normalization (AdaLN) single modulation module for LTX-Video/LTX-2.
- signature: `class LTX2AdaLayerNormSingle(nnx.Module):`
- members:
  - `emb` — [`L56`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L56)
  - `linear` — [`L66`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L66)
  - `num_mod_params` — [`L51`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L51)
  - `silu` — [`L65`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L65) — ---
  - `use_additional_conditions` — [`L55`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L55)
- protocol/private: `__call__`[`L77`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L77), `__init__`[`L41`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L41)
- uses (calls/refs, reference-scoped): [`get_sharding_specs`](logical_sharding_ltx2.md#get_sharding_specs), [`LTX2DiTShardingSpecs`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs), [`adaln_bias`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.adaln_bias), [`adaln_kernel`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.adaln_kernel)
- used by: [`audio_prompt_adaln`](transformer_ltx2.md#LTX2VideoTransformer3DModel.audio_prompt_adaln), [`av_cross_attn_audio_v2a_gate`](transformer_ltx2.md#LTX2VideoTransformer3DModel.av_cross_attn_audio_v2a_gate), [`audio_time_embed`](transformer_ltx2.md#LTX2VideoTransformer3DModel.audio_time_embed), [`av_cross_attn_audio_scale_shift`](transformer_ltx2.md#LTX2VideoTransformer3DModel.av_cross_attn_audio_scale_shift), [`av_cross_attn_video_a2v_gate`](transformer_ltx2.md#LTX2VideoTransformer3DModel.av_cross_attn_video_a2v_gate), [`av_cross_attn_video_scale_shift`](transformer_ltx2.md#LTX2VideoTransformer3DModel.av_cross_attn_video_scale_shift), [`prompt_adaln`](transformer_ltx2.md#LTX2VideoTransformer3DModel.prompt_adaln), [`time_embed`](transformer_ltx2.md#LTX2VideoTransformer3DModel.time_embed)

### `LTX2VideoTransformer3DModel`  ·  implements/extends ConfigMixin, Module
- def: [`src/maxdiffusion/models/ltx2/transformer_ltx2.py:635`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L635)
- signature: `class LTX2VideoTransformer3DModel(nnx.Module, ConfigMixin):`
- members:
  - `init_block(rngs)` — [`L960`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L960) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `scan_fn_ltx2(carry, block)` — [`L1200`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L1200)
  - `scan_fn_ltx23(carry, block_and_mask)` — [`L1278`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L1278)
  - `a2v_attention_kernel` — [`L750`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L750)
  - `activation_fn` — [`L724`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L724)
  - `attention_bias` — [`L729`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L729)
  - `attention_head_dim` — [`L706`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L706)
  - `attention_kernel` — [`L746`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L746)
  - `attention_out_bias` — [`L730`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L730)
  - `audio_attention_head_dim` — [`L717`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L717)
  - `audio_caption_channels` — [`L728`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L728)
  - `audio_caption_projection` — [`L792`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L792)
  - `audio_cross_attention_dim` — [`L718`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L718)
  - `audio_hop_length` — [`L722`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L722)
  - `audio_in_channels` — [`L712`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L712)
  - `audio_norm_out` — [`L1050`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L1050)
  - `audio_num_attention_heads` — [`L716`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L716)
  - `audio_out_channels` — [`L713`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L713)
  - `audio_patch_size` — [`L714`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L714)
  - `audio_patch_size_t` — [`L715`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L715)
  - `audio_pos_embed_max_pos` — [`L720`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L720)
  - `audio_proj_in` — [`L773`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L773) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `audio_proj_out` — [`L1053`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L1053) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `audio_prompt_adaln` — [`L814`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L814) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `audio_rope` — [`L911`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L911) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `audio_sampling_rate` — [`L721`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L721)
  - `audio_scale_factor` — [`L719`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L719)
  - `audio_scale_shift_table` — [`L888`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L888)
  - `audio_time_embed` — [`L834`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L834) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `av_cross_attn_audio_scale_shift` — [`L852`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L852) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `av_cross_attn_audio_v2a_gate` — [`L870`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L870) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `av_cross_attn_video_a2v_gate` — [`L861`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L861) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `av_cross_attn_video_scale_shift` — [`L843`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L843) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `base_height` — [`L710`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L710)
  - `base_width` — [`L711`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L711)
  - `caption_channels` — [`L727`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L727)
  - `caption_projection` — [`L784`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L784)
  - `causal_offset` — [`L734`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L734)
  - `config_name` — [`L636`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L636)
  - `cross_attention_dim` — [`L707`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L707)
  - `cross_attn_audio_rope` — [`L942`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L942) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `cross_attn_mod` — [`L748`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L748)
  - `cross_attn_rope` — [`L928`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L928) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `cross_attn_timestep_scale_multiplier` — [`L736`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L736)
  - `dtype` — [`L738`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L738) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `flash_min_seq_length` — [`L752`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L752)
  - `gated_attn` — [`L747`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L747)
  - `gradient_checkpoint` — [`L1036`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L1036)
  - `in_channels` — [`L701`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L701)
  - `mesh` — [`L740`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L740)
  - `names_which_can_be_offloaded` — [`L744`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L744)
  - `names_which_can_be_saved` — [`L743`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L743)
  - `norm_elementwise_affine` — [`L725`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L725)
  - `norm_eps` — [`L726`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L726)
  - `norm_out` — [`L1037`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L1037)
  - `num_attention_heads` — [`L705`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L705)
  - `num_layers` — [`L723`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L723) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `out_channels` — [`L702`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L702)
  - `patch_size` — [`L703`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L703)
  - `patch_size_t` — [`L704`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L704)
  - `perturbed_attn` — [`L749`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L749)
  - `pos_embed_max_pos` — [`L709`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L709)
  - `precision` — [`L742`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L742)
  - `proj_in` — [`L764`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L764) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `proj_out` — [`L1040`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L1040) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `prompt_adaln` — [`L805`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L805) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `remat_policy` — [`L741`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L741)
  - `rope` — [`L896`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L896) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `rope_double_precision` — [`L732`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L732)
  - `rope_theta` — [`L731`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L731)
  - `rope_type` — [`L737`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L737) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `scale_shift_table` — [`L883`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L883)
  - `scan_layers` — [`L745`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L745)
  - `sharding_specs` — [`L756`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L756) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `spatio_temporal_guidance_blocks` — [`L700`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L700)
  - `time_embed` — [`L825`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L825) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `timestep_scale_multiplier` — [`L735`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L735)
  - `transformer_blocks` — [`L996`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L996) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `use_prompt_embeddings` — [`L733`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L733)
  - `v2a_attention_kernel` — [`L751`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L751)
  - `vae_scale_factors` — [`L708`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L708)
  - `weights_dtype` — [`L739`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L739) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
- protocol/private: `__call__`[`L1063`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L1063), `__init__`[`L639`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L639)
- uses (calls/refs, reference-scoped): [`get_sharding_specs`](logical_sharding_ltx2.md#get_sharding_specs), [`LTX2AdaLayerNormSingle`](transformer_ltx2.md#LTX2AdaLayerNormSingle), [`LTX2DiTShardingSpecs`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs), [`embed_bias`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.embed_bias), [`embed_kernel`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.embed_kernel), [`out_embed_bias`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.out_embed_bias), [`out_embed_kernel`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.out_embed_kernel), [`scale_shift_table`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.scale_shift_table), [`LTX2VideoTransformerBlock`](transformer_ltx2.md#LTX2VideoTransformerBlock)

### `LTX2VideoTransformerBlock`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx2/transformer_ltx2.py:96`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L96)
- signature: `class LTX2VideoTransformerBlock(nnx.Module):`
- members:
  - `attention_kernel` — [`L135`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L135)
  - `attn1` — [`L152`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L152) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `attn2` — [`L209`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L209)
  - `audio_a2v_cross_attn_scale_shift_table` — [`L387`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L387)
  - `audio_attn1` — [`L180`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L180)
  - `audio_attn2` — [`L237`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L237)
  - `audio_ff` — [`L345`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L345)
  - `audio_norm1` — [`L171`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L171) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `audio_norm2` — [`L228`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L228) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `audio_norm3` — [`L336`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L336) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `audio_prompt_scale_shift_table` — [`L394`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L394)
  - `audio_scale_shift_table` — [`L375`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L375)
  - `audio_to_video_attn` — [`L267`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L267)
  - `audio_to_video_norm` — [`L258`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L258) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `cross_attn_mod` — [`L358`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L358) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `dim` — [`L132`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L132)
  - `ff` — [`L326`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L326)
  - `norm1` — [`L143`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L143) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `norm2` — [`L200`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L200) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `norm3` — [`L317`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L317) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
  - `norm_elementwise_affine` — [`L134`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L134)
  - `norm_eps` — [`L133`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L133)
  - `perturbed_attn` — [`L136`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L136)
  - `prompt_scale_shift_table` — [`L369`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L369)
  - `scale_shift_table` — [`L362`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L362)
  - `sharding_specs` — [`L140`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L140)
  - `video_a2v_cross_attn_scale_shift_table` — [`L381`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L381)
  - `video_to_audio_attn` — [`L296`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L296)
  - `video_to_audio_norm` — [`L287`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L287) — documented in [maxdiffusion-models-ltx2-transformer_ltx2](../../../../../concepts/maxdiffusion-models-ltx2-transformer_ltx2.md)
- protocol/private: `__call__`[`L400`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L400), `__init__`[`L98`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/transformer_ltx2.py#L98)
- uses (calls/refs, reference-scoped): [`norm_scale`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.norm_scale), [`get_sharding_specs`](logical_sharding_ltx2.md#get_sharding_specs), [`LTX2DiTShardingSpecs`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs), [`scale_shift_table`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.scale_shift_table)
- used by: [`transformer_blocks`](transformer_ltx2.md#LTX2VideoTransformer3DModel.transformer_blocks), [`init_block`](transformer_ltx2.md#LTX2VideoTransformer3DModel.init_block)

