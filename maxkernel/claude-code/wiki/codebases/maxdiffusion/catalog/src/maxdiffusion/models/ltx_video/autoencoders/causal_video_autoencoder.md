---
title: 'Module: src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.autoencoders.causal_video_autoencoder`/
symbols:
  Decoder.forward: Decoder#forward().
  ResnetBlock3D.forward: ResnetBlock3D#forward().
  Encoder.forward: Encoder#forward().
  Decoder.up_blocks: Decoder#up_blocks.
  DepthToSpaceUpsample.forward: DepthToSpaceUpsample#forward().
  Encoder.down_blocks: Encoder#down_blocks.
  UNetMidBlock3D.forward: UNetMidBlock3D#forward().
  CausalVideoAutoencoder.from_config: CausalVideoAutoencoder#from_config().
  demo_video_autoencoder_forward_backward: demo_video_autoencoder_forward_backward().
  Encoder.conv_norm_out: Encoder#conv_norm_out.
  Decoder.conv_norm_out: Decoder#conv_norm_out.
  ResnetBlock3D.norm1: ResnetBlock3D#norm1.
  ResnetBlock3D.norm2: ResnetBlock3D#norm2.
  CausalVideoAutoencoder.from_pretrained: CausalVideoAutoencoder#from_pretrained().
  UNetMidBlock3D.attention_blocks: UNetMidBlock3D#attention_blocks.
  SpaceToDepthDownsample.forward: SpaceToDepthDownsample#forward().
  SpaceToDepthDownsample.stride: SpaceToDepthDownsample#stride.
  CausalVideoAutoencoder.load_state_dict: CausalVideoAutoencoder#load_state_dict().
  PER_CHANNEL_STATISTICS_PREFIX: PER_CHANNEL_STATISTICS_PREFIX.
  CausalVideoAutoencoder.set_use_tpu_flash_attention: CausalVideoAutoencoder#set_use_tpu_flash_attention().
  test_vae_patchify_unpatchify: test_vae_patchify_unpatchify().
  UNetMidBlock3D: UNetMidBlock3D#
  LayerNorm: LayerNorm#
  UNetMidBlock3D.res_blocks: UNetMidBlock3D#res_blocks.
  Decoder.causal: Decoder#causal.
  DepthToSpaceUpsample.conv: DepthToSpaceUpsample#conv.
  ResnetBlock3D.norm3: ResnetBlock3D#norm3.
  Decoder.timestep_conditioning: Decoder#timestep_conditioning.
  DepthToSpaceUpsample.stride: DepthToSpaceUpsample#stride.
  SpaceToDepthDownsample: SpaceToDepthDownsample#
  DepthToSpaceUpsample: DepthToSpaceUpsample#
  ResnetBlock3D: ResnetBlock3D#
  CausalVideoAutoencoder.to_json_string: CausalVideoAutoencoder#to_json_string().
  LayerNorm.forward: LayerNorm#forward().
  CausalVideoAutoencoder: CausalVideoAutoencoder#
  Encoder.latent_log_var: Encoder#latent_log_var.
  DepthToSpaceUpsample.pixel_shuffle: DepthToSpaceUpsample#pixel_shuffle.
  DepthToSpaceUpsample.residual: DepthToSpaceUpsample#residual.
  ResnetBlock3D.inject_noise: ResnetBlock3D#inject_noise.
  ResnetBlock3D.non_linearity: ResnetBlock3D#non_linearity.
  ResnetBlock3D.timestep_conditioning: ResnetBlock3D#timestep_conditioning.
  ResnetBlock3D._feed_spatial_noise: ResnetBlock3D#_feed_spatial_noise().
  patchify: patchify().
  unpatchify: unpatchify().
  logger: logger.
  CausalVideoAutoencoder.config: CausalVideoAutoencoder#config().
  Encoder.patch_size: Encoder#patch_size.
  Encoder.conv_in: Encoder#conv_in.
  Encoder.conv_act: Encoder#conv_act.
  Encoder.conv_out: Encoder#conv_out.
  Encoder.gradient_checkpointing: Encoder#gradient_checkpointing.
  Decoder.patch_size: Decoder#patch_size.
  Decoder.conv_in: Decoder#conv_in.
  Decoder.conv_act: Decoder#conv_act.
  Decoder.conv_out: Decoder#conv_out.
  Decoder.gradient_checkpointing: Decoder#gradient_checkpointing.
  Decoder.timestep_scale_multiplier: Decoder#timestep_scale_multiplier.
  Decoder.last_time_embedder: Decoder#last_time_embedder.
  Decoder.last_scale_shift_table: Decoder#last_scale_shift_table.
  UNetMidBlock3D.timestep_conditioning: UNetMidBlock3D#timestep_conditioning.
  UNetMidBlock3D.time_embedder: UNetMidBlock3D#time_embedder.
  SpaceToDepthDownsample.group_size: SpaceToDepthDownsample#group_size.
  SpaceToDepthDownsample.conv: SpaceToDepthDownsample#conv.
  DepthToSpaceUpsample.out_channels: DepthToSpaceUpsample#out_channels.
  DepthToSpaceUpsample.out_channels_reduction_factor: DepthToSpaceUpsample#out_channels_reduction_factor.
  LayerNorm.norm: LayerNorm#norm.
  ResnetBlock3D.conv1: ResnetBlock3D#conv1.
  ResnetBlock3D.per_channel_scale1: ResnetBlock3D#per_channel_scale1.
  ResnetBlock3D.dropout: ResnetBlock3D#dropout.
  ResnetBlock3D.conv2: ResnetBlock3D#conv2.
  ResnetBlock3D.per_channel_scale2: ResnetBlock3D#per_channel_scale2.
  ResnetBlock3D.conv_shortcut: ResnetBlock3D#conv_shortcut.
  ResnetBlock3D.scale_shift_table: ResnetBlock3D#scale_shift_table.
  create_video_autoencoder_demo_config: create_video_autoencoder_demo_config().
  Encoder: Encoder#
  Decoder: Decoder#
  CausalVideoAutoencoder.is_video_supported: CausalVideoAutoencoder#is_video_supported().
  CausalVideoAutoencoder.spatial_downscale_factor: CausalVideoAutoencoder#spatial_downscale_factor().
  CausalVideoAutoencoder.temporal_downscale_factor: CausalVideoAutoencoder#temporal_downscale_factor().
  CausalVideoAutoencoder.last_layer: CausalVideoAutoencoder#last_layer().
  Encoder.__init__: Encoder#__init__().
  Encoder.norm_layer: Encoder#norm_layer.
  Encoder.latent_channels: Encoder#latent_channels.
  Encoder.blocks_desc: Encoder#blocks_desc.
  Decoder.__init__: Decoder#__init__().
  Decoder.layers_per_block: Decoder#layers_per_block.
  Decoder.blocks_desc: Decoder#blocks_desc.
  UNetMidBlock3D.__init__: UNetMidBlock3D#__init__().
  SpaceToDepthDownsample.__init__: SpaceToDepthDownsample#__init__().
  DepthToSpaceUpsample.__init__: DepthToSpaceUpsample#__init__().
  LayerNorm.__init__: LayerNorm#__init__().
  ResnetBlock3D.__init__: ResnetBlock3D#__init__().
  ResnetBlock3D.in_channels: ResnetBlock3D#in_channels.
  ResnetBlock3D.out_channels: ResnetBlock3D#out_channels.
---
# Module: [`src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py)

## Classes
### `CausalVideoAutoencoder`  ·  implements/extends AutoencoderKLWrapper
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py:49`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L49)
- signature: `class CausalVideoAutoencoder(AutoencoderKLWrapper):`
- members:
  - `config(self)` — [`L169`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L169)
  - `from_config(config)` — [`L118`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L118)
  - `from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *args, **kwargs)` — [`L52`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L52)
  - `is_video_supported(self)` — [`L189`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L189) — Check if the model supports video inputs of shape (B, C, F, H, W). Otherwise, the model only supports 2D images.
  - `last_layer(self)` — [`L274`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L274)
  - `load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True)` — [`L236`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L236)
  - `set_use_tpu_flash_attention(self)` — [`L284`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L284)
  - `spatial_downscale_factor(self)` — [`L196`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L196)
  - `temporal_downscale_factor(self)` — [`L216`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L216)
  - `to_json_string(self)` — [`L231`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L231)
- uses (calls/refs, reference-scoped): [`attention_blocks`](causal_video_autoencoder.md#UNetMidBlock3D.attention_blocks), [`UNetMidBlock3D`](causal_video_autoencoder.md#UNetMidBlock3D), [`PER_CHANNEL_STATISTICS_PREFIX`](causal_video_autoencoder.md#PER_CHANNEL_STATISTICS_PREFIX), [`Decoder`](causal_video_autoencoder.md#Decoder), [`Encoder`](causal_video_autoencoder.md#Encoder), [`logger`](causal_video_autoencoder.md#logger)
- used by: [`demo_video_autoencoder_forward_backward`](causal_video_autoencoder.md#demo_video_autoencoder_forward_backward)

### `Decoder`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py:526`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L526)
- members:
  - `forward(self, sample: torch.FloatTensor, target_shape, timestep: Optional[torch.Tensor] = None)` — [`L693`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L693) — The forward method of the `Decoder` class. — documented in [maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder](../../../../../../concepts/maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder.md)
  - `blocks_desc` — [`L571`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L571)
  - `causal` — [`L570`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L570)
  - `conv_act` — [`L673`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L673)
  - `conv_in` — [`L582`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L582)
  - `conv_norm_out` — [`L667`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L667)
  - `conv_out` — [`L674`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L674)
  - `gradient_checkpointing` — [`L684`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L684)
  - `last_scale_shift_table` — [`L691`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L691)
  - `last_time_embedder` — [`L690`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L690)
  - `layers_per_block` — [`L568`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L568)
  - `patch_size` — [`L567`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L567)
  - `timestep_conditioning` — [`L686`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L686) — documented in [maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder](../../../../../../concepts/maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder.md)
  - `timestep_scale_multiplier` — [`L689`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L689)
  - `up_blocks` — [`L593`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L593) — documented in [maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder](../../../../../../concepts/maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder.md)
- protocol/private: `__init__`[`L551`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L551)
- uses (calls/refs, reference-scoped): [`LayerNorm`](causal_video_autoencoder.md#LayerNorm), [`UNetMidBlock3D`](causal_video_autoencoder.md#UNetMidBlock3D), [`DepthToSpaceUpsample`](causal_video_autoencoder.md#DepthToSpaceUpsample), [`ResnetBlock3D`](causal_video_autoencoder.md#ResnetBlock3D), [`unpatchify`](causal_video_autoencoder.md#unpatchify)
- used by: [`from_config`](causal_video_autoencoder.md#CausalVideoAutoencoder.from_config)

### `DepthToSpaceUpsample`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py:949`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L949)
- members:
  - `forward(self, x, causal: bool = True)` — [`L976`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L976)
  - `conv` — [`L963`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L963)
  - `out_channels` — [`L962`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L962)
  - `out_channels_reduction_factor` — [`L974`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L974)
  - `pixel_shuffle` — [`L972`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L972)
  - `residual` — [`L973`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L973)
  - `stride` — [`L961`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L961)
- protocol/private: `__init__`[`L951`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L951)
- used by: [`up_blocks`](causal_video_autoencoder.md#Decoder.up_blocks)

### `Encoder`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py:291`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L291)
- members:
  - `forward(self, sample: torch.FloatTensor)` — [`L482`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L482) — The forward method of the `Encoder` class. — documented in [maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder](../../../../../../concepts/maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder.md)
  - `blocks_desc` — [`L334`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L334)
  - `conv_act` — [`L459`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L459)
  - `conv_in` — [`L339`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L339)
  - `conv_norm_out` — [`L453`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L453)
  - `conv_out` — [`L470`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L470)
  - `down_blocks` — [`L350`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L350) — documented in [maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder](../../../../../../concepts/maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder.md)
  - `gradient_checkpointing` — [`L480`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L480)
  - `latent_channels` — [`L332`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L332)
  - `latent_log_var` — [`L333`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L333) — documented in [maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder](../../../../../../concepts/maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder.md)
  - `norm_layer` — [`L331`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L331)
  - `patch_size` — [`L330`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L330)
- protocol/private: `__init__`[`L316`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L316)
- uses (calls/refs, reference-scoped): [`LayerNorm`](causal_video_autoencoder.md#LayerNorm), [`UNetMidBlock3D`](causal_video_autoencoder.md#UNetMidBlock3D), [`ResnetBlock3D`](causal_video_autoencoder.md#ResnetBlock3D), [`SpaceToDepthDownsample`](causal_video_autoencoder.md#SpaceToDepthDownsample), [`patchify`](causal_video_autoencoder.md#patchify)
- used by: [`from_config`](causal_video_autoencoder.md#CausalVideoAutoencoder.from_config)

### `LayerNorm`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py:993`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L993)
- members:
  - `forward(self, x)` — [`L999`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L999)
  - `norm` — [`L997`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L997)
- protocol/private: `__init__`[`L995`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L995)
- used by: [`conv_norm_out`](causal_video_autoencoder.md#Decoder.conv_norm_out), [`conv_norm_out`](causal_video_autoencoder.md#Encoder.conv_norm_out), [`norm1`](causal_video_autoencoder.md#ResnetBlock3D.norm1), [`norm2`](causal_video_autoencoder.md#ResnetBlock3D.norm2), [`norm3`](causal_video_autoencoder.md#ResnetBlock3D.norm3)

### `ResnetBlock3D`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py:1006`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1006) — documented in [maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder](../../../../../../concepts/maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder.md)
- members:
  - `forward(self, input_tensor: torch.FloatTensor, causal: bool = True, timestep: Optional[torch.Tensor] = None)` — [`L1109`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1109) — documented in [maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder](../../../../../../concepts/maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder.md)
  - `conv1` — [`L1047`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1047)
  - `conv2` — [`L1070`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1070)
  - `conv_shortcut` — [`L1084`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1084)
  - `dropout` — [`L1068`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1068)
  - `in_channels` — [`L1033`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1033)
  - `inject_noise` — [`L1036`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1036) — documented in [maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder](../../../../../../concepts/maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder.md)
  - `non_linearity` — [`L1045`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1045) — documented in [maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder](../../../../../../concepts/maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder.md)
  - `norm1` — [`L1039`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1039) — documented in [maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder](../../../../../../concepts/maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder.md)
  - `norm2` — [`L1062`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1062) — documented in [maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder](../../../../../../concepts/maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder.md)
  - `norm3` — [`L1090`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1090) — documented in [maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder](../../../../../../concepts/maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder.md)
  - `out_channels` — [`L1035`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1035)
  - `per_channel_scale1` — [`L1059`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1059)
  - `per_channel_scale2` — [`L1082`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1082)
  - `scale_shift_table` — [`L1095`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1095)
  - `timestep_conditioning` — [`L1092`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1092)
- protocol/private: `__init__`[`L1019`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1019), `_feed_spatial_noise`[`L1097`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1097)
- uses (calls/refs, reference-scoped): [`LayerNorm`](causal_video_autoencoder.md#LayerNorm)
- used by: [`up_blocks`](causal_video_autoencoder.md#Decoder.up_blocks), [`down_blocks`](causal_video_autoencoder.md#Encoder.down_blocks), [`res_blocks`](causal_video_autoencoder.md#UNetMidBlock3D.res_blocks)

### `SpaceToDepthDownsample`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py:903`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L903)
- members:
  - `forward(self, x, causal: bool = True)` — [`L919`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L919)
  - `conv` — [`L909`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L909)
  - `group_size` — [`L908`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L908)
  - `stride` — [`L907`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L907)
- protocol/private: `__init__`[`L905`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L905)
- used by: [`down_blocks`](causal_video_autoencoder.md#Encoder.down_blocks)

### `UNetMidBlock3D`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py:754`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L754) — documented in [maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder](../../../../../../concepts/maxdiffusion-models-ltx_video-autoencoders-causal_video_autoencoder.md)
- members:
  - `forward(self, hidden_states: torch.FloatTensor, causal: bool = True, timestep: Optional[torch.Tensor] = None)` — [`L840`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L840)
  - `attention_blocks` — [`L819`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L819)
  - `res_blocks` — [`L801`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L801)
  - `time_embedder` — [`L799`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L799)
  - `timestep_conditioning` — [`L796`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L796)
- protocol/private: `__init__`[`L780`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L780)
- uses (calls/refs, reference-scoped): [`ResnetBlock3D`](causal_video_autoencoder.md#ResnetBlock3D)
- used by: [`forward`](causal_video_autoencoder.md#Decoder.forward), [`up_blocks`](causal_video_autoencoder.md#Decoder.up_blocks), [`down_blocks`](causal_video_autoencoder.md#Encoder.down_blocks), [`set_use_tpu_flash_attention`](causal_video_autoencoder.md#CausalVideoAutoencoder.set_use_tpu_flash_attention)

## Functions
- `create_video_autoencoder_demo_config(latent_channels: int = 64)` — [`L1201`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1201)
- `demo_video_autoencoder_forward_backward()` — [`L1249`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1249)
- `patchify(x, patch_size_hw, patch_size_t=1)` — [`L1164`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1164)
- `test_vae_patchify_unpatchify()` — [`L1240`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1240)
- `unpatchify(x, patch_size_hw, patch_size_t=1)` — [`L1183`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L1183)

## Module values
- `PER_CHANNEL_STATISTICS_PREFIX` — [`L45`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L45)
- `logger` — [`L46`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/causal_video_autoencoder.py#L46)

