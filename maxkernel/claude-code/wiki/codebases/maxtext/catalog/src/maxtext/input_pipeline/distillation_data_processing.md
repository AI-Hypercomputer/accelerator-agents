---
title: 'Module: src/maxtext/input_pipeline/distillation_data_processing.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/distillation_data_processing.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.distillation_data_processing`/
symbols:
  process_dataset: process_dataset().
  filter_dataset: filter_dataset().
  InputRequest: InputRequest#
  map_to_prompt_completion: map_to_prompt_completion().
  extract_content: extract_content().
  InputRequest.prompt: InputRequest#prompt.
  InputRequest.prompt_token_ids: InputRequest#prompt_token_ids.
  InputRequest.actual_completion: InputRequest#actual_completion.
  InputRequest.max_output_tokens: InputRequest#max_output_tokens.
  load_dataset: load_dataset().
---
# Module: [`src/maxtext/input_pipeline/distillation_data_processing.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/distillation_data_processing.py)

## Classes
### `InputRequest`
- def: [`src/maxtext/input_pipeline/distillation_data_processing.py:31`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/distillation_data_processing.py#L31)
- signature: `class InputRequest:`
- members:
  - `actual_completion` — [`L34`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/distillation_data_processing.py#L34)
  - `max_output_tokens` — [`L35`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/distillation_data_processing.py#L35)
  - `prompt` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/distillation_data_processing.py#L32)
  - `prompt_token_ids` — [`L33`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/distillation_data_processing.py#L33)
- used by: [`filter_dataset`](distillation_data_processing.md#filter_dataset)

## Functions
- `extract_content(example, data_column_names)` — [`L60`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/distillation_data_processing.py#L60) — example = {
- `filter_dataset(config, dataset, tokenizer)` — [`L113`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/distillation_data_processing.py#L113) — Filter out samples from the dataset.
- `load_dataset(config)` — [`L100`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/distillation_data_processing.py#L100) — Loads dataset from Hugging Face.
- `map_to_prompt_completion(example)` — [`L38`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/distillation_data_processing.py#L38) — example = {
- `process_dataset(config, dataset)` — [`L77`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/distillation_data_processing.py#L77) — Pipeline for preprocessing dataset.

