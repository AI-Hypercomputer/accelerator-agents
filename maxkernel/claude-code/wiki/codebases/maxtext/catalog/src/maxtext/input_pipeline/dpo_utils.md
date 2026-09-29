---
title: 'Module: src/maxtext/input_pipeline/dpo_utils.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/dpo_utils.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.dpo_utils`/DPODataFormatting#
symbols:
  DPODataFormatting.map: map().
  DPODataFormatting.data_column_names: data_column_names.
  DPODataFormatting._pad: _pad().
  DPODataFormatting.pad_id: pad_id.
  DPODataFormatting.max_prompt_length: max_prompt_length.
  DPODataFormatting.max_target_length: max_target_length.
  DPODataFormatting: ''
---
# Module: [`src/maxtext/input_pipeline/dpo_utils.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/dpo_utils.py)

## Classes
### `DPODataFormatting`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/dpo_utils.py:23`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/dpo_utils.py#L23)
- doc: Prepares DPO data.
- signature: `class DPODataFormatting(grain.MapTransform):`
- members:
  - `_pad(self, x, length, left=False)` — [`L93`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/dpo_utils.py#L93) — Pads or trims an array to a specific length.
  - `map(self, element)` — [`L34`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/dpo_utils.py#L34) — Apply the dataset transformations for DPO.
  - `data_column_names` — [`L31`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/dpo_utils.py#L31)
  - `max_prompt_length` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/dpo_utils.py#L32)
  - `max_target_length` — [`L30`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/dpo_utils.py#L30)
  - `pad_id` — [`L29`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/dpo_utils.py#L29)

