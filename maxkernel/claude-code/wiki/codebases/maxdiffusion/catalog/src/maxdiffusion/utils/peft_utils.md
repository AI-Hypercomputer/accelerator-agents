---
title: 'Module: src/maxdiffusion/utils/peft_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/utils/peft_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.utils.peft_utils`/
symbols:
  recurse_remove_peft_layers: recurse_remove_peft_layers().
  check_peft_version: check_peft_version().
  scale_lora_layers: scale_lora_layers().
  unscale_lora_layers: unscale_lora_layers().
  get_peft_kwargs: get_peft_kwargs().
  get_adapter_name: get_adapter_name().
  set_adapter_layers: set_adapter_layers().
  set_weights_and_activate_adapters: set_weights_and_activate_adapters().
---
# Module: [`src/maxdiffusion/utils/peft_utils.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/peft_utils.py)

## Functions
- `check_peft_version(min_version: str)` — [`L198`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/peft_utils.py#L198) — Checks if the version of PEFT is compatible.
- `get_adapter_name(model)` — [`L153`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/peft_utils.py#L153)
- `get_peft_kwargs(rank_dict, network_alpha_dict, peft_state_dict, is_unet=True)` — [`L111`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/peft_utils.py#L111)
- `recurse_remove_peft_layers(model)` — [`L25`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/peft_utils.py#L25)
- `scale_lora_layers(model, weight)` — [`L77`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/peft_utils.py#L77) — Adjust the weightage given to the LoRA layers of the model.
- `set_adapter_layers(model, enabled=True)` — [`L162`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/peft_utils.py#L162)
- `set_weights_and_activate_adapters(model, adapter_names, weights)` — [`L174`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/peft_utils.py#L174)
- `unscale_lora_layers(model)` — [`L94`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/peft_utils.py#L94) — Removes the previously passed weight given to the LoRA layers of the model.

