---
title: 'Module: recurrentgemma/conversion.py'
type: catalog
provenance: extracted
module: recurrentgemma/conversion.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.conversion`/
symbols:
  flax_params_to_pytorch_state_dict: flax_params_to_pytorch_state_dict().
  flatten_nested_dict: flatten_nested_dict().
  jax_array_to_torch_tensor: jax_array_to_torch_tensor().
  pytorch_state_dict_to_flax_params: pytorch_state_dict_to_flax_params().
  torch_tensor_to_jax_array: torch_tensor_to_jax_array().
---
# Module: [`recurrentgemma/conversion.py`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion.py)

## Functions
- `flatten_nested_dict(nested_dict: Mapping[str, Any], prefix: str = "")` — [`L49`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion.py#L49) — Recursively flattens a nested dictionary.
- `flax_params_to_pytorch_state_dict(params: Mapping[str, Any])` — [`L65`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion.py#L65) — Converts a Flax params dict to a PyTorch state dict.
- `jax_array_to_torch_tensor(x: jax.Array)` — [`L26`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion.py#L26) — Converts a JAX array to a PyTorch Tensor.
- `pytorch_state_dict_to_flax_params(state_dict: Mapping[str, torch.Tensor])` — [`L86`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion.py#L86) — Converts a PyTorch state dict to a Flax params dict.
- `torch_tensor_to_jax_array(x: torch.Tensor)` — [`L38`](../../../../../raw/code/recurrentgemma/recurrentgemma/conversion.py#L38) — Converts a PyTorch Tensor to a JAX array.

