---
title: 'Module: src/maxtext/utils/accelerator_to_spec_map.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/accelerator_to_spec_map.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.accelerator_to_spec_map`/
symbols:
  SystemCharacteristics: SystemCharacteristics#
  UserFacingNameToSystemCharacteristics: UserFacingNameToSystemCharacteristics.
  get_system_characteristics: get_system_characteristics().
  SystemCharacteristics.platform: SystemCharacteristics#platform.
  SystemCharacteristics.topology_name: SystemCharacteristics#topology_name.
  SystemCharacteristics.chip_config_name: SystemCharacteristics#chip_config_name.
  SystemCharacteristics.chips_per_host_bounds: SystemCharacteristics#chips_per_host_bounds.
  SystemCharacteristics.devices_per_slice: SystemCharacteristics#devices_per_slice.
  SystemCharacteristics.wrap: SystemCharacteristics#wrap.
---
# Module: [`src/maxtext/utils/accelerator_to_spec_map.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/accelerator_to_spec_map.py)

## Classes
### `SystemCharacteristics`
- def: [`src/maxtext/utils/accelerator_to_spec_map.py:25`](../../../../../../../raw/code/maxtext/src/maxtext/utils/accelerator_to_spec_map.py#L25)
- signature: `class SystemCharacteristics:`
- members:
  - `chip_config_name` — [`L28`](../../../../../../../raw/code/maxtext/src/maxtext/utils/accelerator_to_spec_map.py#L28)
  - `chips_per_host_bounds` — [`L29`](../../../../../../../raw/code/maxtext/src/maxtext/utils/accelerator_to_spec_map.py#L29)
  - `devices_per_slice` — [`L30`](../../../../../../../raw/code/maxtext/src/maxtext/utils/accelerator_to_spec_map.py#L30)
  - `platform` — [`L26`](../../../../../../../raw/code/maxtext/src/maxtext/utils/accelerator_to_spec_map.py#L26)
  - `topology_name` — [`L27`](../../../../../../../raw/code/maxtext/src/maxtext/utils/accelerator_to_spec_map.py#L27)
  - `wrap` — [`L31`](../../../../../../../raw/code/maxtext/src/maxtext/utils/accelerator_to_spec_map.py#L31)
- used by: [`UserFacingNameToSystemCharacteristics`](accelerator_to_spec_map.md#UserFacingNameToSystemCharacteristics)

## Functions
- `get_system_characteristics(user_facing_name)` — [`L272`](../../../../../../../raw/code/maxtext/src/maxtext/utils/accelerator_to_spec_map.py#L272)

## Module values
- `UserFacingNameToSystemCharacteristics` — [`L34`](../../../../../../../raw/code/maxtext/src/maxtext/utils/accelerator_to_spec_map.py#L34)

