---
title: 'Module: src/maxtext/input_pipeline/grain_data_processing.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/grain_data_processing.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.grain_data_processing`/
symbols:
  get_datasets: get_datasets().
  _get_pipeline_fn: _get_pipeline_fn().
  make_grain_train_iterator: make_grain_train_iterator().
  sft_preprocessing_pipeline: sft_preprocessing_pipeline().
  get_datasets.create_dataset_from_pattern: get_datasets().create_dataset_from_pattern().
  make_grain_eval_iterator: make_grain_eval_iterator().
  _apply_mapdataset_transforms: _apply_mapdataset_transforms().
  find_data_files: find_data_files().
  _make_elastic_iterator: _make_elastic_iterator().
  pretrain_preprocessing_pipeline: pretrain_preprocessing_pipeline().
  dpo_preprocessing_pipeline: dpo_preprocessing_pipeline().
  _format_chat_template_grain: _format_chat_template_grain().
  _tokenize_sft_chunks: _tokenize_sft_chunks().
---
# Module: [`src/maxtext/input_pipeline/grain_data_processing.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_data_processing.py)

## Functions
- `_apply_mapdataset_transforms(dataset, shuffle, shuffle_seed, num_epoch, dataloading_host_index, dataloading_host_count, grain_num_threads, grain_prefetch_buffer_size, elastic=False)` — [`L52`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_data_processing.py#L52) — Apply standard shuffle, repeat, shard, and iter conversion transforms.
- `_format_chat_template_grain(element, data_columns, tokenizer_model)` — [`L309`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_data_processing.py#L309) — Grain-compatible mapping function to format raw columns into conversational messages.
- `_get_pipeline_fn(config)` — [`L394`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_data_processing.py#L394) — Returns the appropriate preprocessing pipeline function based on config.
- `_make_elastic_iterator(dataset, config, preprocessing_fn, shard_index=None, shard_count=None, mp_opts=None)` — [`L403`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_data_processing.py#L403) — Applies preprocessing_fn then wraps the result with ElasticIterator.
- `_tokenize_sft_chunks(element, text_column_name, tokenizer_model)` — [`L334`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_data_processing.py#L334) — Tokenize each chunk individually without truncating.
- `create_dataset_from_pattern(pattern)` — [`L103`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_data_processing.py#L103)
- `dpo_preprocessing_pipeline(dataset, config, data_columns, tokenize, grain_worker_count, grain_per_worker_buffer_size)` — [`L270`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_data_processing.py#L270) — Use grain to pre-process the dataset and return iterators for dpo fine-tuning
- `find_data_files(data_file_pattern)` — [`L39`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_data_processing.py#L39) — Find data files matching the pattern.
- `get_datasets(data_file_pattern, data_file_type, shuffle, shuffle_seed, shuffle_buffer_size, num_epoch, dataloading_host_index, dataloading_host_count, grain_worker_count, grain_num_threads, grain_prefetch_buffer_size, grain_data_source_max_workers, mixture_config_path=None, elastic=False)` — [`L84`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_data_processing.py#L84) — Load dataset from array_record files for using with grain
- `make_grain_eval_iterator(config: ml_collections.ConfigDict, global_mesh, process_indices)` — [`L525`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_data_processing.py#L525) — Load, preprocess dataset and return iterators
- `make_grain_train_iterator(config: ml_collections.ConfigDict, global_mesh, process_indices)` — [`L424`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_data_processing.py#L424) — Load, preprocess dataset and return iterators
- `pretrain_preprocessing_pipeline(dataset, config, data_columns, tokenize, grain_worker_count, grain_per_worker_buffer_size)` — [`L223`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_data_processing.py#L223) — Use grain pipeline to pre-process the dataset and return iterators for pretrain
- `sft_preprocessing_pipeline(dataset, config, data_columns, tokenize, grain_worker_count, grain_per_worker_buffer_size)` — [`L341`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_data_processing.py#L341) — Use grain pipeline to pre-process the dataset and return iterators for sft fine-tuning

