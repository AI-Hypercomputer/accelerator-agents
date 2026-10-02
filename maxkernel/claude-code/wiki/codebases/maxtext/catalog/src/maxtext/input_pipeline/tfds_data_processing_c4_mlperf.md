---
title: 'Module: src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.tfds_data_processing_c4_mlperf`/
symbols:
  preprocess_eval_dataset: preprocess_eval_dataset().
  AUTOTUNE: AUTOTUNE.
  preprocess_train_dataset: preprocess_train_dataset().
  split_tokens: split_tokens().
  rekey: rekey().
  make_c4_mlperf_train_iterator: make_c4_mlperf_train_iterator().
  make_c4_mlperf_eval_iterator: make_c4_mlperf_eval_iterator().
  reduce_concat_tokens: reduce_concat_tokens().
  _pad_to_batch_size: _pad_to_batch_size().
  split_tokens_to_targets_length: split_tokens_to_targets_length().
  format_fn: format_fn().
  get_dataset: get_dataset().
  _shift_left_and_pad: _shift_left_and_pad().
  rekey._rekey: rekey()._rekey().
  reduce_concat_tokens._my_fn: reduce_concat_tokens()._my_fn().
  split_tokens._split_tokens: split_tokens()._split_tokens().
  split_tokens._strip_padding: split_tokens()._strip_padding().
  _pad_to_batch_size._get_num_examples: _pad_to_batch_size()._get_num_examples().
  _pad_to_batch_size._add_pad: _pad_to_batch_size()._add_pad().
---
# Module: [`src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py)

## Functions
- `_add_pad(x)` — [`L194`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L194)
- `_get_num_examples(ds: tf.data.Dataset)` — [`L172`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L172)
- `_my_fn(x)` — [`L106`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L106)
- `_pad_to_batch_size(ds: tf.data.Dataset, batch_size: int, num_examples: None | int = None)` — [`L160`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L160) — Pad unevenly distributed eval data in each shard with new entries to multiples of batch size.
- `_rekey(x, key_map=None)` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L66) — Replace the feature keys according to the mapping in `key_map`.
- `_shift_left_and_pad(tensor, pad_val)` — [`L44`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L44) — Shift the input to the left with pad_val
- `_split_tokens(x)` — [`L133`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L133) — Split one token sequence into multiple multiple.
- `_strip_padding(x)` — [`L146`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L146)
- `format_fn(x, eos_id: int = 1, pad_id: int = 0)` — [`L233`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L233) — Format function for c4_mlperf.
- `get_dataset(dataset_name: str, split: str, dataloading_host_index: int, dataloading_host_count: int, enable_data_shuffling: bool = False, data_shuffle_seed: int = 0, shard_in_read: bool = False)` — [`L202`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L202) — Load and return a dataset of examples.
- `make_c4_mlperf_eval_iterator(config: ml_collections.ConfigDict, global_mesh, process_indices)` — [`L351`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L351) — Make eval iterator of customized C4 dataset for mlperf gpt3 training.
- `make_c4_mlperf_train_iterator(config: ml_collections.ConfigDict, global_mesh, process_indices)` — [`L320`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L320) — Make train iterator of customized C4 dataset for mlperf gpt3 training.
- `preprocess_eval_dataset(eval_ds: tf.data.Dataset, sp_tokenizer, eval_global_batch_size_to_load: int, max_target_length: int, num_examples: None | int = None, is_tokenized_dataset: bool = True)` — [`L276`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L276) — Preprocess the evaluation dataset.
- `preprocess_train_dataset(train_ds: tf.data.Dataset, sp_tokenizer, train_global_batch_size_to_load: int, max_target_length: int, shuffle_buffer_size: int, data_shuffle_seed: int)` — [`L245`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L245) — Preprocess the training dataset.
- `reduce_concat_tokens(dataset, feature_key="targets", batch_size=128)` — [`L87`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L87) — Token-preprocessor to concatenate multiple unrelated documents.
- `rekey(ds, key_map=None)` — [`L63`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L63) — normalization with key mapping
- `split_tokens(dataset, max_tokens_per_segment=128, feature_key="targets")` — [`L115`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L115) — Split examples into multiple examples each.
- `split_tokens_to_targets_length(dataset, sequence_length)` — [`L156`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L156)

## Module values
- `AUTOTUNE` — [`L36`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing_c4_mlperf.py#L36)

