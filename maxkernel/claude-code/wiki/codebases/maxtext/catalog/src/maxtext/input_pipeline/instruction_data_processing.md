---
title: 'Module: src/maxtext/input_pipeline/instruction_data_processing.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/instruction_data_processing.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.instruction_data_processing`/
symbols:
  convert_to_conversational_format: convert_to_conversational_format().
  math_qa_formatting: math_qa_formatting().
  load_formatter: load_formatter().
  get_template_placeholders: get_template_placeholders().
  load_data_template_from_file: load_data_template_from_file().
  extract_reasoning_and_answer: extract_reasoning_and_answer().
  load_formatter.formatter: load_formatter().formatter().
  load_chat_template_from_file: load_chat_template_from_file().
---
# Module: [`src/maxtext/input_pipeline/instruction_data_processing.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/instruction_data_processing.py)

## Functions
- `convert_to_conversational_format(dataset, data_columns, formatting_func_path=None, formatting_func_kwargs=None)` — [`L178`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/instruction_data_processing.py#L178) — Converts instruction dataset to conversational format.
- `extract_reasoning_and_answer(text, separator)` — [`L81`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/instruction_data_processing.py#L81)
- `formatter(dataset, dataset_features)` — [`L164`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/instruction_data_processing.py#L164)
- `get_template_placeholders(template)` — [`L75`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/instruction_data_processing.py#L75) — Dynamically extracts the format keys (placeholders) from a template string.
- `load_chat_template_from_file(template_path)` — [`L47`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/instruction_data_processing.py#L47) — Loads a chat template from a file.
- `load_data_template_from_file(template_path)` — [`L25`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/instruction_data_processing.py#L25) — Loads a data template from a file.
- `load_formatter(formatting_func_path, **kwargs)` — [`L155`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/instruction_data_processing.py#L155) — Loads a formatter function from a given path.
- `math_qa_formatting(example, template_config=None)` — [`L88`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/instruction_data_processing.py#L88) — Maps question-answer pairs to conversational format.

