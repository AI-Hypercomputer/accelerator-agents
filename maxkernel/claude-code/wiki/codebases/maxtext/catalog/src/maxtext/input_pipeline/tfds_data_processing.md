---
title: 'Module: src/maxtext/input_pipeline/tfds_data_processing.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/tfds_data_processing.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.tfds_data_processing`/
symbols:
  preprocessing_pipeline: preprocessing_pipeline().
  make_tfds_train_iterator: make_tfds_train_iterator().
  make_tfds_eval_iterator: make_tfds_eval_iterator().
  get_datasets: get_datasets().
  AUTOTUNE: AUTOTUNE.
---
# Module: [`src/maxtext/input_pipeline/tfds_data_processing.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing.py)

## Functions
- `get_datasets(dataset_name, data_split, shuffle_files, shuffle_seed, dataloading_host_index, dataloading_host_count, dataset_path=None)` — [`L41`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing.py#L41) — Load a TFDS dataset.
- `make_tfds_eval_iterator(config: ml_collections.ConfigDict, global_mesh, process_indices_eval)` — [`L264`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing.py#L264) — load eval dataset, preprocess and return iterators
- `make_tfds_train_iterator(config: ml_collections.ConfigDict, global_mesh, process_indices_train)` — [`L195`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing.py#L195) — load dataset, preprocess and return iterators
- `preprocessing_pipeline(dataset, tokenizer_path, tokenizer_type: str, global_batch_size: int, max_target_length: int, data_column_names, shuffle: bool = False, data_shuffle_seed=0, tokenize: bool = True, add_bos: bool = True, add_eos: bool = True, num_epochs: None | int = 1, pack_examples: bool = True, shuffle_buffer_size: int = 1024, shift: bool = True, drop_remainder: bool = True, prefetch_size=tf.data.experimental.AUTOTUNE, hf_access_token: str = "")` — [`L87`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing.py#L87) — pipeline for preprocessing TFDS dataset.

## Module values
- `AUTOTUNE` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tfds_data_processing.py#L32)

