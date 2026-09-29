---
title: 'Module: src/maxdiffusion/models/ltx_video/autoencoders/vae.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/autoencoders/vae.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.autoencoders.vae`/AutoencoderKLWrapper#
symbols:
  AutoencoderKLWrapper._hw_tiled_encode: _hw_tiled_encode().
  AutoencoderKLWrapper._hw_tiled_decode: _hw_tiled_decode().
  AutoencoderKLWrapper.decode: decode().
  AutoencoderKLWrapper.encode: encode().
  AutoencoderKLWrapper.latent_norm_out: latent_norm_out.
  AutoencoderKLWrapper._decode: _decode().
  AutoencoderKLWrapper.tile_sample_min_size: tile_sample_min_size.
  AutoencoderKLWrapper._encode: _encode().
  AutoencoderKLWrapper._unnormalize_latent_channels: _unnormalize_latent_channels().
  AutoencoderKLWrapper.post_quant_conv: post_quant_conv.
  AutoencoderKLWrapper.encoder: encoder.
  AutoencoderKLWrapper.use_hw_tiling: use_hw_tiling.
  AutoencoderKLWrapper.z_sample_size: z_sample_size.
  AutoencoderKLWrapper.decoder_params: decoder_params.
  AutoencoderKLWrapper.tile_latent_min_size: tile_latent_min_size.
  AutoencoderKLWrapper.enable_z_tiling: enable_z_tiling().
  AutoencoderKLWrapper.forward: forward().
  AutoencoderKLWrapper.decoder: decoder.
  AutoencoderKLWrapper.use_z_tiling: use_z_tiling.
  AutoencoderKLWrapper.tile_overlap_factor: tile_overlap_factor.
  AutoencoderKLWrapper._normalize_latent_channels: _normalize_latent_channels().
  AutoencoderKLWrapper.quant_conv: quant_conv.
  AutoencoderKLWrapper.disable_z_tiling: disable_z_tiling().
  AutoencoderKLWrapper.enable_hw_tiling: enable_hw_tiling().
  AutoencoderKLWrapper.disable_hw_tiling: disable_hw_tiling().
  AutoencoderKLWrapper.blend_v: blend_v().
  AutoencoderKLWrapper.blend_h: blend_h().
  AutoencoderKLWrapper.set_tiling_params: set_tiling_params().
  AutoencoderKLWrapper: ''
  AutoencoderKLWrapper.__init__: __init__().
  AutoencoderKLWrapper.use_quant_conv: use_quant_conv.
  AutoencoderKLWrapper.normalize_latent_channels: normalize_latent_channels.
  AutoencoderKLWrapper.dims: dims.
  AutoencoderKLWrapper.blend_z: blend_z().
---
# Module: [`src/maxdiffusion/models/ltx_video/autoencoders/vae.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py)

## Classes
### `AutoencoderKLWrapper`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/vae.py:32`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L32)
- doc: Variational Autoencoder (VAE) model with KL loss.
- signature: `class AutoencoderKLWrapper(ModelMixin, ConfigMixin):`
- members:
  - `blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int)` — [`L176`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L176)
  - `blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int)` — [`L170`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L170)
  - `blend_z(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int)` — [`L164`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L164)
  - `decode(self, z: torch.FloatTensor, return_dict: bool = True, target_shape=None, timestep: Optional[torch.Tensor] = None)` — [`L288`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L288)
  - `disable_hw_tiling(self)` — [`L121`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L121) — Disable tiling during VAE decoding along the height and width dimension.
  - `disable_z_tiling(self)` — [`L108`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L108) — Disable tiling during VAE decoding. If `use_tiling` was previously invoked, this method will go back to computing
  - `enable_hw_tiling(self)` — [`L115`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L115) — Enable tiling during VAE decoding along the height and width dimension.
  - `enable_z_tiling(self, z_sample_size: int = 8)` — [`L97`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L97) — Enable tiling during VAE decoding.
  - `encode(self, z: torch.FloatTensor, return_dict: bool = True)` — [`L224`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L224)
  - `forward(self, sample: torch.FloatTensor, sample_posterior: bool = False, return_dict: bool = True, generator: Optional[torch.Generator] = None)` — [`L332`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L332) — Args:
  - `set_tiling_params(self, sample_size: int = 512, overlap_factor: float = 0.25)` — [`L91`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L91)
  - `decoder` — [`L66`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L66)
  - `decoder_params` — [`L86`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L86)
  - `dims` — [`L83`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L83)
  - `encoder` — [`L60`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L60)
  - `latent_norm_out` — [`L76`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L76)
  - `normalize_latent_channels` — [`L62`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L62)
  - `post_quant_conv` — [`L69`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L69)
  - `quant_conv` — [`L68`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L68)
  - `tile_latent_min_size` — [`L94`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L94)
  - `tile_overlap_factor` — [`L95`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L95)
  - `tile_sample_min_size` — [`L92`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L92)
  - `use_hw_tiling` — [`L82`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L82)
  - `use_quant_conv` — [`L61`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L61)
  - `use_z_tiling` — [`L81`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L81)
  - `z_sample_size` — [`L84`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L84)
- protocol/private: `__init__`[`L47`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L47), `_decode`[`L274`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L274), `_encode`[`L268`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L268), `_hw_tiled_decode`[`L182`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L182), `_hw_tiled_encode`[`L127`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L127), `_normalize_latent_channels`[`L243`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L243), `_unnormalize_latent_channels`[`L257`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae.py#L257)
- uses (calls/refs, reference-scoped): `running_mean`, `running_var`

