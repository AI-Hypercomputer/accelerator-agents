---
title: 'Module: src/maxdiffusion/models/ltx2/ltx2_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx2/ltx2_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx2.ltx2_utils`/
symbols:
  load_transformer_weights: load_transformer_weights().
  load_vae_weights: load_vae_weights().
  load_vocoder_weights: load_vocoder_weights().
  load_connector_weights: load_connector_weights().
  load_audio_vae_weights: load_audio_vae_weights().
  load_sharded_checkpoint: load_sharded_checkpoint().
  load_upsampler_weights: load_upsampler_weights().
  _tuple_str_to_int: _tuple_str_to_int().
  get_key_and_value: get_key_and_value().
  rename_for_ltx2_transformer: rename_for_ltx2_transformer().
  rename_for_ltx2_vocoder: rename_for_ltx2_vocoder().
  rename_for_ltx2_connector: rename_for_ltx2_connector().
  rename_for_ltx2_audio_vae: rename_for_ltx2_audio_vae().
  rename_for_ltx2_upsampler: rename_for_ltx2_upsampler().
  KNOWN_UPSAMPLER_CONFIGS: KNOWN_UPSAMPLER_CONFIGS.
  adain_filter_latent: adain_filter_latent().
  tone_map_latents: tone_map_latents().
---
# Module: [`src/maxdiffusion/models/ltx2/ltx2_utils.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py)

## Functions
- `_tuple_str_to_int(in_tuple)` — [`L53`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L53)
- `adain_filter_latent(latents: jax.Array, reference_latents: jax.Array, factor: float = 1)` — [`L573`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L573) — Scales high-res latents using global channel statistics from reference latents.
- `get_key_and_value(pt_tuple_key, tensor, flax_state_dict, random_flax_state_dict, scan_layers, num_layers=48)` — [`L92`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L92)
- `load_audio_vae_weights(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, subfolder: str = "audio_vae")` — [`L456`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L456)
- `load_connector_weights(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, subfolder: str = "connectors")` — [`L378`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L378)
- `load_sharded_checkpoint(pretrained_model_name_or_path, subfolder, device, filename=None)` — [`L132`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L132) — Loads weights from a sharded safetensors checkpoint or a specific file.
- `load_transformer_weights(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, num_layers: int = 48, scan_layers: bool = True, subfolder: str = "transformer")` — [`L187`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L187)
- `load_upsampler_weights(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, subfolder: str = "latent_upsampler", dims: int = 3, filename: Optional[str] = None)` — [`L517`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L517) — Loads and ports PyTorch upsampler weights to Flax.
- `load_vae_weights(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, subfolder: str = "vae")` — [`L230`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L230)
- `load_vocoder_weights(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, subfolder: str = "vocoder")` — [`L320`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L320)
- `rename_for_ltx2_audio_vae(key)` — [`L433`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L433)
- `rename_for_ltx2_connector(key)` — [`L356`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L356)
- `rename_for_ltx2_transformer(key)` — [`L63`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L63) — Renames Diffusers LTX-2 keys to MaxDiffusion Flax LTX-2 keys.
- `rename_for_ltx2_upsampler(key)` — [`L493`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L493) — Renames PyTorch Latent Upsampler keys to match Flax LTX2LatentUpsamplerModel.
- `rename_for_ltx2_vocoder(key)` — [`L313`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L313)
- `tone_map_latents(latents: jax.Array, compression: float)` — [`L587`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L587) — Sigmoid-based compression to regularize high-variance latents.

## Module values
- `KNOWN_UPSAMPLER_CONFIGS` — [`L29`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/ltx2_utils.py#L29)

