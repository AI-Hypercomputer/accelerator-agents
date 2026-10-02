---
title: 'Module: src/maxdiffusion/loaders/lora_conversion_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/loaders/lora_conversion_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.loaders.lora_conversion_utils`/
symbols:
  _convert_kohya_flux_lora_to_diffusers._convert_to_ai_toolkit: _convert_kohya_flux_lora_to_diffusers()._convert_to_ai_toolkit().
  _convert_non_diffusers_lora_to_diffusers: _convert_non_diffusers_lora_to_diffusers().
  _convert_kohya_flux_lora_to_diffusers._convert_sd_scripts_to_ai_toolkit: _convert_kohya_flux_lora_to_diffusers()._convert_sd_scripts_to_ai_toolkit().
  _convert_kohya_flux_lora_to_diffusers._convert_to_ai_toolkit_cat: _convert_kohya_flux_lora_to_diffusers()._convert_to_ai_toolkit_cat().
  preprocess_wan_lora_dict: preprocess_wan_lora_dict().
  _convert_kohya_flux_lora_to_diffusers: _convert_kohya_flux_lora_to_diffusers().
  _convert_xlabs_flux_lora_to_diffusers: _convert_xlabs_flux_lora_to_diffusers().
  _maybe_map_sgm_blocks_to_diffusers: _maybe_map_sgm_blocks_to_diffusers().
  _convert_xlabs_flux_lora_to_diffusers.handle_qkv: _convert_xlabs_flux_lora_to_diffusers().handle_qkv().
  translate_wan_nnx_path_to_diffusers_lora: translate_wan_nnx_path_to_diffusers_lora().
  _convert_unet_lora_key: _convert_unet_lora_key().
  _convert_text_encoder_lora_key: _convert_text_encoder_lora_key().
  _get_alpha_name: _get_alpha_name().
  translate_ltx2_nnx_path_to_diffusers_lora: translate_ltx2_nnx_path_to_diffusers_lora().
---
# Module: [`src/maxdiffusion/loaders/lora_conversion_utils.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_conversion_utils.py)

## Functions
- `_convert_kohya_flux_lora_to_diffusers(state_dict)` — [`L325`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_conversion_utils.py#L325)
- `_convert_non_diffusers_lora_to_diffusers(state_dict, unet_name="unet", text_encoder_name="text_encoder")` — [`L122`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_conversion_utils.py#L122) — Converts a non-Diffusers LoRA state dict to a Diffusers compatible state dict.
- `_convert_sd_scripts_to_ai_toolkit(sds_sd)` — [`L408`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_conversion_utils.py#L408)
- `_convert_text_encoder_lora_key(key, lora_name)` — [`L281`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_conversion_utils.py#L281) — Converts a text encoder LoRA key to a Diffusers compatible key.
- `_convert_to_ai_toolkit(sds_sd, ait_sd, sds_key, ait_key)` — [`L326`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_conversion_utils.py#L326)
- `_convert_to_ai_toolkit_cat(sds_sd, ait_sd, sds_key, ait_keys, dims=None)` — [`L346`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_conversion_utils.py#L346)
- `_convert_unet_lora_key(key)` — [`L225`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_conversion_utils.py#L225) — Converts a U-Net LoRA key to a Diffusers compatible key.
- `_convert_xlabs_flux_lora_to_diffusers(old_state_dict)` — [`L517`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_conversion_utils.py#L517)
- `_get_alpha_name(lora_name_alpha, diffusers_name, alpha)` — [`L308`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_conversion_utils.py#L308) — Gets the correct alpha name for the Diffusers model.
- `_maybe_map_sgm_blocks_to_diffusers(state_dict, unet_config, delimiter="_", block_slice_pos=5)` — [`L22`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_conversion_utils.py#L22)
- `handle_qkv(sds_sd, ait_sd, sds_key, ait_keys, dims=None)` — [`L521`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_conversion_utils.py#L521)
- `preprocess_wan_lora_dict(state_dict)` — [`L613`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_conversion_utils.py#L613) — Preprocesses WAN LoRA dict to convert diff_m to modulation.diff.
- `translate_ltx2_nnx_path_to_diffusers_lora(nnx_path_str, scan_layers=False)` — [`L708`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_conversion_utils.py#L708) — Translates LTX2 NNX path to Diffusers/LoRA keys.
- `translate_wan_nnx_path_to_diffusers_lora(nnx_path_str, scan_layers=False)` — [`L627`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/loaders/lora_conversion_utils.py#L627) — Translates WAN NNX path to Diffusers/LoRA keys.

