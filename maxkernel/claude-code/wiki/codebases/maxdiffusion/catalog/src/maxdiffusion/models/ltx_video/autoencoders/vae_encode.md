---
title: 'Module: src/maxdiffusion/models/ltx_video/autoencoders/vae_encode.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/autoencoders/vae_encode.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.autoencoders.vae_encode`/
symbols:
  _run_decoder: _run_decoder().
  latent_to_pixel_coords: latent_to_pixel_coords().
  vae_encode: vae_encode().
  vae_decode: vae_decode().
  get_vae_size_scale_factor: get_vae_size_scale_factor().
  un_normalize_latents: un_normalize_latents().
  latent_to_pixel_coords_from_factors: latent_to_pixel_coords_from_factors().
  normalize_latents: normalize_latents().
---
# Module: [`src/maxdiffusion/models/ltx_video/autoencoders/vae_encode.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_encode.py)

## Functions
- `_run_decoder(latents: Tensor, vae: AutoencoderKL, is_video: bool, vae_per_channel_normalize=False, timestep=None)` — [`L134`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_encode.py#L134)
- `get_vae_size_scale_factor(vae: AutoencoderKL)` — [`L169`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_encode.py#L169)
- `latent_to_pixel_coords(latent_coords: Tensor, vae: AutoencoderKL, causal_fix: bool = False)` — [`L181`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_encode.py#L181) — Converts latent coordinates to pixel coordinates by scaling them according to the VAE's
- `latent_to_pixel_coords_from_factors(latent_coords: Tensor, scale_factors: Tuple, causal_fix: bool = False)` — [`L202`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_encode.py#L202)
- `normalize_latents(latents: Tensor, vae: AutoencoderKL, vae_per_channel_normalize: bool = False)` — [`L210`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_encode.py#L210)
- `un_normalize_latents(latents: Tensor, vae: AutoencoderKL, vae_per_channel_normalize: bool = False)` — [`L219`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_encode.py#L219)
- `vae_decode(latents: Tensor, vae: AutoencoderKL, is_video: bool = True, split_size: int = 1, vae_per_channel_normalize=False, timestep=None)` — [`L104`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_encode.py#L104)
- `vae_encode(media_items: Tensor, vae: AutoencoderKL, split_size: int = 1, vae_per_channel_normalize=False)` — [`L37`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/autoencoders/vae_encode.py#L37) — Encodes media items (images or videos) into latent representations using a specified VAE model.

