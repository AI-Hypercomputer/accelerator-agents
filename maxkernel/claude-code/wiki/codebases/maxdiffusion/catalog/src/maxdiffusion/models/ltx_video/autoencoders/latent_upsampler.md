---
title: 'Module: src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.autoencoders.latent_upsampler`/
symbols:
  LatentUpsampler.forward: LatentUpsampler#forward().
  LatentUpsampler.config: LatentUpsampler#config().
  ResBlock.forward: ResBlock#forward().
  upsampled_latent: upsampled_latent.
  latent_upsampler: latent_upsampler.
  LatentUpsampler.upsampler: LatentUpsampler#upsampler.
  total_params: total_params.
  LatentUpsampler.from_pretrained: LatentUpsampler#from_pretrained().
  LatentUpsampler.res_blocks: LatentUpsampler#res_blocks.
  LatentUpsampler.post_upsample_res_blocks: LatentUpsampler#post_upsample_res_blocks.
  ResBlock: ResBlock#
  ResBlock.activation: ResBlock#activation.
  LatentUpsampler: LatentUpsampler#
  LatentUpsampler.dims: LatentUpsampler#dims.
  LatentUpsampler.temporal_upsample: LatentUpsampler#temporal_upsample.
  LatentUpsampler.initial_conv: LatentUpsampler#initial_conv.
  LatentUpsampler.initial_norm: LatentUpsampler#initial_norm.
  LatentUpsampler.initial_activation: LatentUpsampler#initial_activation.
  LatentUpsampler.final_conv: LatentUpsampler#final_conv.
  ResBlock.conv1: ResBlock#conv1.
  ResBlock.norm1: ResBlock#norm1.
  ResBlock.conv2: ResBlock#conv2.
  ResBlock.norm2: ResBlock#norm2.
  LatentUpsampler.in_channels: LatentUpsampler#in_channels.
  LatentUpsampler.mid_channels: LatentUpsampler#mid_channels.
  LatentUpsampler.num_blocks_per_stage: LatentUpsampler#num_blocks_per_stage.
  LatentUpsampler.spatial_upsample: LatentUpsampler#spatial_upsample.
  LatentUpsampler.from_config: LatentUpsampler#from_config().
  latent: latent.
  ResBlock.__init__: ResBlock#__init__().
  LatentUpsampler.__init__: LatentUpsampler#__init__().
---
# Module: [`src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py)

## Classes
### `LatentUpsampler`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py:57`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L57)
- doc: Model to spatially upsample VAE latents.
- signature: `class LatentUpsampler(ModelMixin, ConfigMixin):`
- members:
  - `config(self)` — [`L171`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L171)
  - `forward(self, latent: torch.Tensor)` — [`L118`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L118)
  - `from_config(cls, config)` — [`L161`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L161)
  - `from_pretrained(cls, pretrained_model_path: Optional[Union[str, os.PathLike]], *args, **kwargs)` — [`L183`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L183)
  - `dims` — [`L84`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L84)
  - `final_conv` — [`L116`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L116)
  - `in_channels` — [`L81`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L81)
  - `initial_activation` — [`L92`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L92)
  - `initial_conv` — [`L90`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L90)
  - `initial_norm` — [`L91`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L91)
  - `mid_channels` — [`L82`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L82)
  - `num_blocks_per_stage` — [`L83`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L83)
  - `post_upsample_res_blocks` — [`L114`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L114)
  - `res_blocks` — [`L94`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L94)
  - `spatial_upsample` — [`L85`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L85)
  - `temporal_upsample` — [`L86`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L86)
  - `upsampler` — [`L97`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L97)
- protocol/private: `__init__`[`L70`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L70)
- uses (calls/refs, reference-scoped): [`ResBlock`](latent_upsampler.md#ResBlock)
- used by: [`latent_upsampler`](latent_upsampler.md#latent_upsampler)

### `ResBlock`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py:31`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L31)
- signature: `class ResBlock(nn.Module):`
- members:
  - `forward(self, x: torch.Tensor)` — [`L46`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L46)
  - `activation` — [`L44`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L44)
  - `conv1` — [`L40`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L40)
  - `conv2` — [`L42`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L42)
  - `norm1` — [`L41`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L41)
  - `norm2` — [`L43`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L43)
- protocol/private: `__init__`[`L33`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L33)
- used by: [`post_upsample_res_blocks`](latent_upsampler.md#LatentUpsampler.post_upsample_res_blocks), [`res_blocks`](latent_upsampler.md#LatentUpsampler.res_blocks)

## Module values
- `latent` — [`L208`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L208)
- `latent_upsampler` — [`L204`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L204)
- `total_params` — [`L206`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L206)
- `upsampled_latent` — [`L209`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/latent_upsampler.py#L209)

