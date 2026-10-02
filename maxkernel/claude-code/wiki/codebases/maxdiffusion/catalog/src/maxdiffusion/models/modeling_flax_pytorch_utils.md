---
title: 'Module: src/maxdiffusion/models/modeling_flax_pytorch_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/modeling_flax_pytorch_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.modeling_flax_pytorch_utils`/
symbols:
  torch2jax: torch2jax().
  rename_key: rename_key().
  validate_flax_state_dict: validate_flax_state_dict().
  rename_key_and_reshape_tensor: rename_key_and_reshape_tensor().
  convert_pytorch_state_dict_to_flax: convert_pytorch_state_dict_to_flax().
  create_flax_params_from_pytorch_state: create_flax_params_from_pytorch_state().
  convert_flux_lora_pytorch_state_dict_to_flax: convert_flux_lora_pytorch_state_dict_to_flax().
  convert_lora_pytorch_state_dict_to_flax: convert_lora_pytorch_state_dict_to_flax().
  logger: logger.
  get_network_alpha_value: get_network_alpha_value().
---
# Module: [`src/maxdiffusion/models/modeling_flax_pytorch_utils.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_pytorch_utils.py)

## Functions
- `convert_flux_lora_pytorch_state_dict_to_flax(config, pt_state_dict, params, adapter_name)` — [`L285`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_pytorch_utils.py#L285)
- `convert_lora_pytorch_state_dict_to_flax(pt_state_dict, params, network_alphas, adapter_name)` — [`L340`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_pytorch_utils.py#L340)
- `convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model, init_key=42)` — [`L368`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_pytorch_utils.py#L368)
- `create_flax_params_from_pytorch_state(pt_state_dict, unet_state_dict, text_encoder_state_dict, text_encoder_2_state_dict, network_alphas, adapter_name, is_lora=False)` — [`L195`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_pytorch_utils.py#L195)
- `get_network_alpha_value(pt_key, network_alphas)` — [`L175`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_pytorch_utils.py#L175)
- `rename_key(key)` — [`L74`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_pytorch_utils.py#L74)
- `rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor, random_flax_state_dict, scan_layers=False)` — [`L89`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_pytorch_utils.py#L89) — Rename PT weight names to corresponding Flax weight names and reshape tensor if necessary
- `torch2jax(torch_tensor: torch.Tensor)` — [`L59`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_pytorch_utils.py#L59)
- `validate_flax_state_dict(expected_pytree: dict, new_pytree: dict)` — [`L33`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_pytorch_utils.py#L33) — expected_pytree: dict - a pytree that comes from initializing the model.

## Module values
- `logger` — [`L30`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_pytorch_utils.py#L30)

