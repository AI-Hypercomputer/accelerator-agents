---
title: 'Module: src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx2.latent_upsampler_ltx2`/
symbols:
  LTX2LatentUpsamplerModel.final_conv: LTX2LatentUpsamplerModel#final_conv.
  LTX2LatentUpsamplerModel.__call__: LTX2LatentUpsamplerModel#__call__().
  LTX2LatentUpsamplerModel.mid_channels: LTX2LatentUpsamplerModel#mid_channels.
  LTX2LatentUpsamplerModel.pixel_shuffle: LTX2LatentUpsamplerModel#pixel_shuffle.
  LTX2LatentUpsamplerModel.initial_norm: LTX2LatentUpsamplerModel#initial_norm.
  BlurDownsample.kernel_size: BlurDownsample#kernel_size.
  LTX2LatentUpsamplerModel.upsampler: LTX2LatentUpsamplerModel#upsampler.
  ResBlock.__call__: ResBlock#__call__().
  BlurDownsample.__call__: BlurDownsample#__call__().
  LTX2LatentUpsamplerModel.upsampler_0: LTX2LatentUpsamplerModel#upsampler_0.
  LTX2LatentUpsamplerModel.rational_spatial_scale: LTX2LatentUpsamplerModel#rational_spatial_scale.
  SpatialRationalResampler.scale: SpatialRationalResampler#scale.
  LTX2LatentUpsamplerModel.num_blocks_per_stage: LTX2LatentUpsamplerModel#num_blocks_per_stage.
  LTX2LatentUpsamplerModel.initial_conv: LTX2LatentUpsamplerModel#initial_conv.
  SpatialRationalResampler.__call__: SpatialRationalResampler#__call__().
  ResBlock.conv2: ResBlock#conv2.
  LTX2LatentUpsamplerModel.spatial_upsample: LTX2LatentUpsamplerModel#spatial_upsample.
  PixelShuffleND: PixelShuffleND#
  PixelShuffleND.__call__: PixelShuffleND#__call__().
  BlurDownsample.stride: BlurDownsample#stride.
  LTX2LatentUpsamplerModel.in_channels: LTX2LatentUpsamplerModel#in_channels.
  ResBlock.dims: ResBlock#dims.
  ResBlock: ResBlock#
  LTX2LatentUpsamplerModel.temporal_upsample: LTX2LatentUpsamplerModel#temporal_upsample.
  ResBlock.conv1: ResBlock#conv1.
  ResBlock.norm1: ResBlock#norm1.
  ResBlock.norm2: ResBlock#norm2.
  BlurDownsample.kernel: BlurDownsample#kernel.
  SpatialRationalResampler.conv: SpatialRationalResampler#conv.
  SpatialRationalResampler.pixel_shuffle: SpatialRationalResampler#pixel_shuffle.
  SpatialRationalResampler.blur_down: SpatialRationalResampler#blur_down.
  ResBlock.mid_channels: ResBlock#mid_channels.
  PixelShuffleND.dims: PixelShuffleND#dims.
  PixelShuffleND.upscale_factors: PixelShuffleND#upscale_factors.
  BlurDownsample.dims: BlurDownsample#dims.
  RATIONAL_RESAMPLER_SCALE_MAPPING: RATIONAL_RESAMPLER_SCALE_MAPPING.
  ResBlock.channels: ResBlock#channels.
  SpatialRationalResampler: SpatialRationalResampler#
  LTX2LatentUpsamplerModel.dims: LTX2LatentUpsamplerModel#dims.
  BlurDownsample: BlurDownsample#
  SpatialRationalResampler.mid_channels: SpatialRationalResampler#mid_channels.
  ResBlock.__init__: ResBlock#__init__().
  PixelShuffleND.__init__: PixelShuffleND#__init__().
  BlurDownsample.__init__: BlurDownsample#__init__().
  SpatialRationalResampler.__init__: SpatialRationalResampler#__init__().
  LTX2LatentUpsamplerModel: LTX2LatentUpsamplerModel#
  LTX2LatentUpsamplerModel.__init__: LTX2LatentUpsamplerModel#__init__().
  LTX2LatentUpsamplerModel.load_config: LTX2LatentUpsamplerModel#load_config().
---
# Module: [`src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py)

## Classes
### `BlurDownsample`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py:101`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L101)
- signature: `class BlurDownsample(nnx.Module):`
- members:
  - `dims` — [`L104`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L104)
  - `kernel` — [`L116`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L116)
  - `kernel_size` — [`L106`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L106)
  - `stride` — [`L105`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L105)
- protocol/private: `__call__`[`L118`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L118), `__init__`[`L103`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L103)
- used by: [`blur_down`](latent_upsampler_ltx2.md#SpatialRationalResampler.blur_down)

### `LTX2LatentUpsamplerModel`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py:177`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L177)
- signature: `class LTX2LatentUpsamplerModel(nnx.Module):`
- members:
  - `load_config(cls, pretrained_model_name_or_path: str, subfolder: str = "", **kwargs)` — [`L264`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L264)
  - `dims` — [`L194`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L194)
  - `final_conv` — [`L222`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L222)
  - `in_channels` — [`L191`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L191)
  - `initial_conv` — [`L200`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L200)
  - `initial_norm` — [`L203`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L203)
  - `mid_channels` — [`L192`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L192)
  - `num_blocks_per_stage` — [`L193`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L193)
  - `pixel_shuffle` — [`L217`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L217)
  - `rational_spatial_scale` — [`L197`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L197)
  - `spatial_upsample` — [`L195`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L195)
  - `temporal_upsample` — [`L196`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L196)
  - `upsampler` — [`L210`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L210)
  - `upsampler_0` — [`L214`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L214)
- protocol/private: `__call__`[`L281`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L281), `__init__`[`L179`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L179)
- uses (calls/refs, reference-scoped): [`PixelShuffleND`](latent_upsampler_ltx2.md#PixelShuffleND), [`ResBlock`](latent_upsampler_ltx2.md#ResBlock), [`SpatialRationalResampler`](latent_upsampler_ltx2.md#SpatialRationalResampler)

### `PixelShuffleND`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py:68`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L68)
- signature: `class PixelShuffleND(nnx.Module):`
- members:
  - `dims` — [`L71`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L71)
  - `upscale_factors` — [`L72`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L72)
- protocol/private: `__call__`[`L74`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L74), `__init__`[`L70`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L70)
- used by: [`final_conv`](latent_upsampler_ltx2.md#LTX2LatentUpsamplerModel.final_conv), [`pixel_shuffle`](latent_upsampler_ltx2.md#LTX2LatentUpsamplerModel.pixel_shuffle), [`pixel_shuffle`](latent_upsampler_ltx2.md#SpatialRationalResampler.pixel_shuffle)

### `ResBlock`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py:37`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L37)
- signature: `class ResBlock(nnx.Module):`
- members:
  - `channels` — [`L40`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L40)
  - `conv1` — [`L47`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L47)
  - `conv2` — [`L50`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L50)
  - `dims` — [`L42`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L42)
  - `mid_channels` — [`L41`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L41)
  - `norm1` — [`L48`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L48)
  - `norm2` — [`L51`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L51)
- protocol/private: `__call__`[`L53`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L53), `__init__`[`L39`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L39)
- used by: [`final_conv`](latent_upsampler_ltx2.md#LTX2LatentUpsamplerModel.final_conv), [`pixel_shuffle`](latent_upsampler_ltx2.md#LTX2LatentUpsamplerModel.pixel_shuffle), [`initial_norm`](latent_upsampler_ltx2.md#LTX2LatentUpsamplerModel.initial_norm)

### `SpatialRationalResampler`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py:154`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L154)
- signature: `class SpatialRationalResampler(nnx.Module):`
- members:
  - `blur_down` — [`L168`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L168)
  - `conv` — [`L164`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L164)
  - `mid_channels` — [`L157`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L157)
  - `pixel_shuffle` — [`L167`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L167)
  - `scale` — [`L158`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L158)
- protocol/private: `__call__`[`L170`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L170), `__init__`[`L156`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L156)
- uses (calls/refs, reference-scoped): [`PixelShuffleND`](latent_upsampler_ltx2.md#PixelShuffleND), [`RATIONAL_RESAMPLER_SCALE_MAPPING`](latent_upsampler_ltx2.md#RATIONAL_RESAMPLER_SCALE_MAPPING), [`BlurDownsample`](latent_upsampler_ltx2.md#BlurDownsample)
- used by: [`final_conv`](latent_upsampler_ltx2.md#LTX2LatentUpsamplerModel.final_conv), [`upsampler`](latent_upsampler_ltx2.md#LTX2LatentUpsamplerModel.upsampler)

## Module values
- `RATIONAL_RESAMPLER_SCALE_MAPPING` — [`L29`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/latent_upsampler_ltx2.py#L29)

