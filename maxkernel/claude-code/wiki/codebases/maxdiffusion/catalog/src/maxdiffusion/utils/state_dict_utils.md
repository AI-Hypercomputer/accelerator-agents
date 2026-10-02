---
title: 'Module: src/maxdiffusion/utils/state_dict_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/utils/state_dict_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.utils.state_dict_utils`/
symbols:
  PEFT_STATE_DICT_MAPPINGS: PEFT_STATE_DICT_MAPPINGS.
  DIFFUSERS_STATE_DICT_MAPPINGS: DIFFUSERS_STATE_DICT_MAPPINGS.
  convert_state_dict_to_peft: convert_state_dict_to_peft().
  convert_state_dict_to_diffusers: convert_state_dict_to_diffusers().
  StateDictType: StateDictType#
  convert_unet_state_dict_to_peft: convert_unet_state_dict_to_peft().
  convert_state_dict: convert_state_dict().
  StateDictType.DIFFUSERS_OLD: StateDictType#DIFFUSERS_OLD.
  StateDictType.PEFT: StateDictType#PEFT.
  StateDictType.DIFFUSERS: StateDictType#DIFFUSERS.
  KEYS_TO_ALWAYS_REPLACE: KEYS_TO_ALWAYS_REPLACE.
  UNET_TO_DIFFUSERS: UNET_TO_DIFFUSERS.
  DIFFUSERS_TO_PEFT: DIFFUSERS_TO_PEFT.
  DIFFUSERS_OLD_TO_PEFT: DIFFUSERS_OLD_TO_PEFT.
  PEFT_TO_DIFFUSERS: PEFT_TO_DIFFUSERS.
  DIFFUSERS_OLD_TO_DIFFUSERS: DIFFUSERS_OLD_TO_DIFFUSERS.
---
# Module: [`src/maxdiffusion/utils/state_dict_utils.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py)

## Classes
### `StateDictType`  ·  implements/extends Enum
- def: [`src/maxdiffusion/utils/state_dict_utils.py:20`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L20)
- doc: The mode to use when converting state dicts.
- signature: `class StateDictType(enum.Enum):`
- members:
  - `DIFFUSERS` — [`L28`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L28)
  - `DIFFUSERS_OLD` — [`L25`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L25)
  - `PEFT` — [`L27`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L27)
- used by: [`DIFFUSERS_STATE_DICT_MAPPINGS`](state_dict_utils.md#DIFFUSERS_STATE_DICT_MAPPINGS), [`PEFT_STATE_DICT_MAPPINGS`](state_dict_utils.md#PEFT_STATE_DICT_MAPPINGS), [`convert_state_dict_to_diffusers`](state_dict_utils.md#convert_state_dict_to_diffusers), [`convert_state_dict_to_peft`](state_dict_utils.md#convert_state_dict_to_peft)

## Functions
- `convert_state_dict(state_dict, mapping)` — [`L110`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L110) — Simply iterates over the state dict and replaces the patterns in `mapping` with the corresponding values.
- `convert_state_dict_to_diffusers(state_dict, original_type=None, **kwargs)` — [`L170`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L170) — Converts a state dict to new diffusers format. The state dict can be from previous diffusers format
- `convert_state_dict_to_peft(state_dict, original_type=None, **kwargs)` — [`L143`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L143) — Converts a state dict to the PEFT format The state dict can be from previous diffusers format (`OLD_DIFFUSERS`), or
- `convert_unet_state_dict_to_peft(state_dict)` — [`L217`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L217) — Converts a state dict from UNet format to diffusers format - i.e. by removing some keys

## Module values
- `DIFFUSERS_OLD_TO_DIFFUSERS` — [`L84`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L84)
- `DIFFUSERS_OLD_TO_PEFT` — [`L60`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L60)
- `DIFFUSERS_STATE_DICT_MAPPINGS` — [`L100`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L100)
- `DIFFUSERS_TO_PEFT` — [`L47`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L47)
- `KEYS_TO_ALWAYS_REPLACE` — [`L105`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L105)
- `PEFT_STATE_DICT_MAPPINGS` — [`L95`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L95)
- `PEFT_TO_DIFFUSERS` — [`L73`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L73)
- `UNET_TO_DIFFUSERS` — [`L33`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/state_dict_utils.py#L33)

