---
title: 'Module: src/maxtext/input_pipeline/data_processing_utils.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/data_processing_utils.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.data_processing_utils`/
symbols:
  parse_and_keep_features: parse_and_keep_features().
  get_tokenizer_and_pad_id: get_tokenizer_and_pad_id().
  validate_and_configure_sft_columns: validate_and_configure_sft_columns().
  get_local_batch_size: get_local_batch_size().
  format_and_batch: format_and_batch().
  apply_multiprocessing_and_prefetch: apply_multiprocessing_and_prefetch().
---
# Module: [`src/maxtext/input_pipeline/data_processing_utils.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/data_processing_utils.py)

## Functions
- `apply_multiprocessing_and_prefetch(dataset, config, grain_worker_count, grain_per_worker_buffer_size)` — [`L141`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/data_processing_utils.py#L141) — Applies multiprocessing and prefetching configurations to the dataset.
- `format_and_batch(dataset, config, batch_size, pad_id, data_columns, tokenizer_model, shift=True)` — [`L81`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/data_processing_utils.py#L81) — Packs or pads the dataset, batches it, and optionally shifts tokens for next-token prediction.
- `get_local_batch_size(config)` — [`L67`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/data_processing_utils.py#L67) — Computes local batch size based on process count and expansion factor.
- `get_tokenizer_and_pad_id(config)` — [`L38`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/data_processing_utils.py#L38) — Builds tokenizer and extracts pad_id safely.
- `parse_and_keep_features(dataset, config, data_columns, tokenize)` — [`L28`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/data_processing_utils.py#L28) — Parse arrayrecord features or keep specified columns for other formats.
- `validate_and_configure_sft_columns(data_columns, tokenizer_model, chat_template=None)` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/data_processing_utils.py#L56) — Validates SFT data columns and configures the tokenizer chat template.

