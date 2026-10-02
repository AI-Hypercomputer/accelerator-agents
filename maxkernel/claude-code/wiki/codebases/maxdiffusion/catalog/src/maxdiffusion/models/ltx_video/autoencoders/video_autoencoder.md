---
title: 'Module: src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.autoencoders.video_autoencoder`/
symbols:
  Encoder.forward: Encoder#forward().
  Decoder.forward: Decoder#forward().
  ResnetBlock3D.forward: ResnetBlock3D#forward().
  Decoder.up_blocks: Decoder#up_blocks.
  Encoder.down_blocks: Encoder#down_blocks.
  VideoAutoencoder.from_config: VideoAutoencoder#from_config().
  DownEncoderBlock3D.downsample: DownEncoderBlock3D#downsample.
  UpDecoderBlock3D.upsample: UpDecoderBlock3D#upsample.
  demo_video_autoencoder_forward_backward: demo_video_autoencoder_forward_backward().
  Encoder.downscale_factor: Encoder#downscale_factor().
  Downsample3D.forward: Downsample3D#forward().
  Decoder.layers_per_block: Decoder#layers_per_block.
  Encoder.latent_log_var: Encoder#latent_log_var.
  Encoder.mid_block: Encoder#mid_block.
  VideoAutoencoder.from_pretrained: VideoAutoencoder#from_pretrained().
  DownEncoderBlock3D.forward: DownEncoderBlock3D#forward().
  UpDecoderBlock3D.forward: UpDecoderBlock3D#forward().
  Upsample3D.forward: Upsample3D#forward().
  test_vae_patchify_unpatchify: test_vae_patchify_unpatchify().
  Encoder.conv_norm_out: Encoder#conv_norm_out.
  Decoder.conv_norm_out: Decoder#conv_norm_out.
  ResnetBlock3D.norm1: ResnetBlock3D#norm1.
  ResnetBlock3D.norm2: ResnetBlock3D#norm2.
  Encoder.patch_size: Encoder#patch_size.
  VideoAutoencoder.load_state_dict: VideoAutoencoder#load_state_dict().
  UNetMidBlock3D.res_blocks: UNetMidBlock3D#res_blocks.
  Decoder.patch_size: Decoder#patch_size.
  ResnetBlock3D: ResnetBlock3D#
  VideoAutoencoder.to_json_string: VideoAutoencoder#to_json_string().
  DownEncoderBlock3D.__init__: DownEncoderBlock3D#__init__().
  UNetMidBlock3D.forward: UNetMidBlock3D#forward().
  UpDecoderBlock3D.__init__: UpDecoderBlock3D#__init__().
  VideoAutoencoder: VideoAutoencoder#
  Encoder.patch_size_t: Encoder#patch_size_t.
  Encoder.layers_per_block: Encoder#layers_per_block.
  Decoder.patch_size_t: Decoder#patch_size_t.
  Decoder.mid_block: Decoder#mid_block.
  ResnetBlock3D.non_linearity: ResnetBlock3D#non_linearity.
  Downsample3D.dims: Downsample3D#dims.
  Upsample3D.dims: Upsample3D#dims.
  Upsample3D.conv: Upsample3D#conv.
  patchify: patchify().
  unpatchify: unpatchify().
  UNetMidBlock3D: UNetMidBlock3D#
  Downsample3D: Downsample3D#
  logger: logger.
  VideoAutoencoder.config: VideoAutoencoder#config().
  Encoder.add_channel_padding: Encoder#add_channel_padding.
  Encoder.latent_channels: Encoder#latent_channels.
  Encoder.conv_in: Encoder#conv_in.
  Encoder.conv_act: Encoder#conv_act.
  Encoder.conv_out: Encoder#conv_out.
  Encoder.gradient_checkpointing: Encoder#gradient_checkpointing.
  Decoder.add_channel_padding: Decoder#add_channel_padding.
  Decoder.conv_in: Decoder#conv_in.
  Decoder.conv_act: Decoder#conv_act.
  Decoder.conv_out: Decoder#conv_out.
  Decoder.gradient_checkpointing: Decoder#gradient_checkpointing.
  DownEncoderBlock3D.res_blocks: DownEncoderBlock3D#res_blocks.
  UpDecoderBlock3D.res_blocks: UpDecoderBlock3D#res_blocks.
  ResnetBlock3D.conv1: ResnetBlock3D#conv1.
  ResnetBlock3D.dropout: ResnetBlock3D#dropout.
  ResnetBlock3D.conv2: ResnetBlock3D#conv2.
  ResnetBlock3D.conv_shortcut: ResnetBlock3D#conv_shortcut.
  Downsample3D.padding: Downsample3D#padding.
  Downsample3D.conv: Downsample3D#conv.
  create_video_autoencoder_pathify4x4x4_config: create_video_autoencoder_pathify4x4x4_config().
  Encoder: Encoder#
  Decoder: Decoder#
  DownEncoderBlock3D: DownEncoderBlock3D#
  UpDecoderBlock3D: UpDecoderBlock3D#
  Upsample3D: Upsample3D#
  VideoAutoencoder.is_video_supported: VideoAutoencoder#is_video_supported().
  VideoAutoencoder.downscale_factor: VideoAutoencoder#downscale_factor().
  VideoAutoencoder.last_layer: VideoAutoencoder#last_layer().
  Encoder.__init__: Encoder#__init__().
  Encoder.norm_layer: Encoder#norm_layer.
  Encoder.in_channels: Encoder#in_channels.
  Decoder.__init__: Decoder#__init__().
  Decoder.out_channels: Decoder#out_channels.
  UNetMidBlock3D.__init__: UNetMidBlock3D#__init__().
  UpDecoderBlock3D.resolution_idx: UpDecoderBlock3D#resolution_idx.
  ResnetBlock3D.__init__: ResnetBlock3D#__init__().
  ResnetBlock3D.in_channels: ResnetBlock3D#in_channels.
  ResnetBlock3D.out_channels: ResnetBlock3D#out_channels.
  ResnetBlock3D.use_conv_shortcut: ResnetBlock3D#use_conv_shortcut.
  Downsample3D.__init__: Downsample3D#__init__().
  Downsample3D.in_channels: Downsample3D#in_channels.
  Upsample3D.__init__: Upsample3D#__init__().
  Upsample3D.channels: Upsample3D#channels.
  Upsample3D.out_channels: Upsample3D#out_channels.
  create_video_autoencoder_config: create_video_autoencoder_config().
  create_video_autoencoder_pathify4x4_config: create_video_autoencoder_pathify4x4_config().
---
# Module: [`src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py)

## Classes
### `Decoder`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py:359`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L359)
- members:
  - `forward(self, sample: torch.FloatTensor, target_shape)` — [`L455`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L455) — The forward method of the `Decoder` class.
  - `add_channel_padding` — [`L396`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L396)
  - `conv_act` — [`L450`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L450)
  - `conv_in` — [`L404`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L404)
  - `conv_norm_out` — [`L446`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L446)
  - `conv_out` — [`L451`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L451)
  - `gradient_checkpointing` — [`L453`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L453)
  - `layers_per_block` — [`L397`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L397)
  - `mid_block` — [`L413`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L413)
  - `out_channels` — [`L402`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L402)
  - `patch_size` — [`L394`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L394)
  - `patch_size_t` — [`L395`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L395)
  - `up_blocks` — [`L414`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L414)
- protocol/private: `__init__`[`L380`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L380)
- uses (calls/refs, reference-scoped): [`UNetMidBlock3D`](video_autoencoder.md#UNetMidBlock3D), [`unpatchify`](video_autoencoder.md#unpatchify), [`UpDecoderBlock3D`](video_autoencoder.md#UpDecoderBlock3D)
- used by: [`from_config`](video_autoencoder.md#VideoAutoencoder.from_config)

### `DownEncoderBlock3D`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py:493`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L493)
- members:
  - `forward(self, hidden_states: torch.FloatTensor, downsample_in_time)` — [`L537`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L537)
  - `downsample` — [`L528`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L528)
  - `res_blocks` — [`L525`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L525)
- protocol/private: `__init__`[`L495`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L495)
- uses (calls/refs, reference-scoped): [`Identity`](../utils/torch_utils.md#Identity), [`ResnetBlock3D`](video_autoencoder.md#ResnetBlock3D), [`Downsample3D`](video_autoencoder.md#Downsample3D)
- used by: [`down_blocks`](video_autoencoder.md#Encoder.down_blocks)

### `Downsample3D`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py:731`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L731)
- members:
  - `forward(self, x, downsample_in_time=True)` — [`L755`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L755)
  - `conv` — [`L746`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L746)
  - `dims` — [`L745`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L745)
  - `in_channels` — [`L744`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L744)
  - `padding` — [`L743`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L743)
- protocol/private: `__init__`[`L733`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L733)
- used by: [`downsample`](video_autoencoder.md#DownEncoderBlock3D.downsample), [`downscale_factor`](video_autoencoder.md#Encoder.downscale_factor)

### `Encoder`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py:186`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L186)
- members:
  - `downscale_factor(self)` — [`L299`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L299)
  - `forward(self, sample: torch.FloatTensor, return_features=False)` — [`L302`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L302) — The forward method of the `Encoder` class.
  - `add_channel_padding` — [`L226`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L226)
  - `conv_act` — [`L285`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L285)
  - `conv_in` — [`L238`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L238)
  - `conv_norm_out` — [`L278`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L278)
  - `conv_out` — [`L294`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L294)
  - `down_blocks` — [`L247`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L247)
  - `gradient_checkpointing` — [`L296`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L296)
  - `in_channels` — [`L235`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L235)
  - `latent_channels` — [`L229`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L229)
  - `latent_log_var` — [`L230`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L230)
  - `layers_per_block` — [`L227`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L227)
  - `mid_block` — [`L267`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L267)
  - `norm_layer` — [`L228`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L228)
  - `patch_size` — [`L224`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L224)
  - `patch_size_t` — [`L225`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L225)
- protocol/private: `__init__`[`L209`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L209)
- uses (calls/refs, reference-scoped): [`Downsample3D`](video_autoencoder.md#Downsample3D), [`UNetMidBlock3D`](video_autoencoder.md#UNetMidBlock3D), [`patchify`](video_autoencoder.md#patchify), [`DownEncoderBlock3D`](video_autoencoder.md#DownEncoderBlock3D)
- used by: [`from_config`](video_autoencoder.md#VideoAutoencoder.from_config)

### `ResnetBlock3D`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py:650`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L650)
- members:
  - `forward(self, input_tensor: torch.FloatTensor)` — [`L704`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L704)
  - `conv1` — [`L687`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L687)
  - `conv2` — [`L696`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L696)
  - `conv_shortcut` — [`L698`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L698)
  - `dropout` — [`L694`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L694)
  - `in_channels` — [`L675`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L675)
  - `non_linearity` — [`L685`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L685)
  - `norm1` — [`L681`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L681)
  - `norm2` — [`L690`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L690)
  - `out_channels` — [`L677`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L677)
  - `use_conv_shortcut` — [`L678`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L678)
- protocol/private: `__init__`[`L663`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L663)
- used by: [`res_blocks`](video_autoencoder.md#UNetMidBlock3D.res_blocks), [`__init__`](video_autoencoder.md#DownEncoderBlock3D.__init__), [`__init__`](video_autoencoder.md#UpDecoderBlock3D.__init__)

### `UNetMidBlock3D`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py:546`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L546)
- members:
  - `forward(self, hidden_states: torch.FloatTensor)` — [`L592`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L592)
  - `res_blocks` — [`L577`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L577)
- protocol/private: `__init__`[`L564`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L564)
- uses (calls/refs, reference-scoped): [`ResnetBlock3D`](video_autoencoder.md#ResnetBlock3D)
- used by: [`up_blocks`](video_autoencoder.md#Decoder.up_blocks), [`mid_block`](video_autoencoder.md#Encoder.mid_block)

### `UpDecoderBlock3D`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py:599`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L599)
- members:
  - `forward(self, hidden_states: torch.FloatTensor, upsample_in_time=True)` — [`L641`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L641)
  - `res_blocks` — [`L632`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L632)
  - `resolution_idx` — [`L639`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L639)
  - `upsample` — [`L635`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L635)
- protocol/private: `__init__`[`L601`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L601)
- uses (calls/refs, reference-scoped): [`Identity`](../utils/torch_utils.md#Identity), [`ResnetBlock3D`](video_autoencoder.md#ResnetBlock3D), [`Upsample3D`](video_autoencoder.md#Upsample3D)
- used by: [`up_blocks`](video_autoencoder.md#Decoder.up_blocks)

### `Upsample3D`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py:771`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L771)
- members:
  - `forward(self, x, upsample_in_time)` — [`L785`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L785)
  - `channels` — [`L781`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L781)
  - `conv` — [`L783`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L783)
  - `dims` — [`L780`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L780)
  - `out_channels` — [`L782`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L782)
- protocol/private: `__init__`[`L778`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L778)
- used by: [`upsample`](video_autoencoder.md#UpDecoderBlock3D.upsample)

### `VideoAutoencoder`  ·  implements/extends AutoencoderKLWrapper
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py:38`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L38)
- signature: `class VideoAutoencoder(AutoencoderKLWrapper):`
- members:
  - `config(self)` — [`L118`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L118)
  - `downscale_factor(self)` — [`L145`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L145)
  - `from_config(config)` — [`L71`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L71)
  - `from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *args, **kwargs)` — [`L41`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L41)
  - `is_video_supported(self)` — [`L138`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L138) — Check if the model supports video inputs of shape (B, C, F, H, W). Otherwise, the model only supports 2D images.
  - `last_layer(self)` — [`L175`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L175)
  - `load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True)` — [`L153`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L153)
  - `to_json_string(self)` — [`L148`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L148)
- uses (calls/refs, reference-scoped): [`Decoder`](video_autoencoder.md#Decoder), [`Encoder`](video_autoencoder.md#Encoder), [`logger`](video_autoencoder.md#logger)
- used by: [`demo_video_autoencoder_forward_backward`](video_autoencoder.md#demo_video_autoencoder_forward_backward)

## Functions
- `create_video_autoencoder_config(latent_channels: int = 4)` — [`L873`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L873)
- `create_video_autoencoder_pathify4x4_config(latent_channels: int = 4)` — [`L917`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L917)
- `create_video_autoencoder_pathify4x4x4_config(latent_channels: int = 4)` — [`L897`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L897)
- `demo_video_autoencoder_forward_backward()` — [`L943`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L943)
- `patchify(x, patch_size_hw, patch_size_t=1, add_channel_padding=False)` — [`L819`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L819)
- `test_vae_patchify_unpatchify()` — [`L934`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L934)
- `unpatchify(x, patch_size_hw, patch_size_t=1, add_channel_padding=False)` — [`L851`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L851)

## Module values
- `logger` — [`L35`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/video_autoencoder.py#L35)

