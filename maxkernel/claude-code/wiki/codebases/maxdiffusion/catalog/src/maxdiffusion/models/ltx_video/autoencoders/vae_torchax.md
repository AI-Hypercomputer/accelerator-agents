---
title: 'Module: src/maxdiffusion/models/ltx_video/autoencoders/vae_torchax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/autoencoders/vae_torchax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.autoencoders.vae_torchax`/TorchaxCausalVideoAutoencoder#
symbols:
  TorchaxCausalVideoAutoencoder.encode: encode().
  TorchaxCausalVideoAutoencoder.decode: decode().
  TorchaxCausalVideoAutoencoder._vae_encoder_inner: _vae_encoder_inner().
  TorchaxCausalVideoAutoencoder._vae_decoder_inner: _vae_decoder_inner().
  TorchaxCausalVideoAutoencoder: ''
  TorchaxCausalVideoAutoencoder.__init__: __init__().
  TorchaxCausalVideoAutoencoder.normalize_img: normalize_img().
  TorchaxCausalVideoAutoencoder.denormalize_img: denormalize_img().
---
# Module: [`src/maxdiffusion/models/ltx_video/autoencoders/vae_torchax.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_torchax.py)

## Classes
### `TorchaxCausalVideoAutoencoder`
- def: [`src/maxdiffusion/models/ltx_video/autoencoders/vae_torchax.py:30`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_torchax.py#L30)
- signature: `class TorchaxCausalVideoAutoencoder(interop.JittableModule):`
- members:
  - `decode(self, latents: jax.Array, timestep: jax.Array, split_size: int = 1, vae_per_channel_normalize: bool = True, is_video: bool = True)` — [`L59`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_torchax.py#L59)
  - `denormalize_img(image)` — [`L110`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_torchax.py#L110)
  - `encode(self, media_items: jax.Array, split_size: int = 1, vae_per_channel_normalize: bool = True)` — [`L35`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_torchax.py#L35)
  - `normalize_img(image)` — [`L106`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_torchax.py#L106)
- protocol/private: `__init__`[`L32`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_torchax.py#L32), `_vae_decoder_inner`[`L93`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_torchax.py#L93), `_vae_encoder_inner`[`L84`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_torchax.py#L84)

