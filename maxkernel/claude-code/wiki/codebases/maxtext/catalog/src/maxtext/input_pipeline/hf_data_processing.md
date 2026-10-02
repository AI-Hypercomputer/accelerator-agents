---
title: 'Module: src/maxtext/input_pipeline/hf_data_processing.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/hf_data_processing.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.hf_data_processing`/
symbols:
  make_hf_train_iterator: make_hf_train_iterator().
  make_hf_eval_iterator: make_hf_eval_iterator().
  vision_sft_preprocessing_pipeline: vision_sft_preprocessing_pipeline().
  preprocessing_pipeline: preprocessing_pipeline().
  _get_pad_id: _get_pad_id().
---
# Module: [`src/maxtext/input_pipeline/hf_data_processing.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/hf_data_processing.py)

## Functions
- `_get_pad_id(tokenizer)` — [`L35`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/hf_data_processing.py#L35)
- `make_hf_eval_iterator(config: ml_collections.ConfigDict, global_mesh, process_indices_eval)` — [`L453`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/hf_data_processing.py#L453) — Make Hugging Face evaluation iterator. Load and preprocess eval dataset: and return iterator.
- `make_hf_train_iterator(config: ml_collections.ConfigDict, global_mesh, process_indices_train)` — [`L391`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/hf_data_processing.py#L391) — Load, preprocess dataset and return iterators
- `preprocessing_pipeline(dataloading_host_index, dataloading_host_count, global_mesh, dataset, config, data_column_names, tokenize, tokenizer_path, hf_access_token, global_batch_size, max_target_length, shuffle, data_shuffle_seed, chat_template_path="", add_bos=True, add_eos=True, packing=True, shift=True, num_threads=1, drop_remainder=True, generate_padding_batch=False, use_dpo=False, use_sft=None, use_tunix_gradient_accumulation=False, num_microbatches=1, sft_train_on_completion_only=True, grain_worker_count=1, max_segments_per_seq=None, num_epoch=1, chat_template: Optional[str] = None, formatting_func_path: Optional[str] = None, formatting_func_kwargs: Optional[dict] = None)` — [`L195`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/hf_data_processing.py#L195) — pipeline for preprocessing HF dataset
- `vision_sft_preprocessing_pipeline(dataset, config, dataloading_host_index, dataloading_host_count, global_mesh, text_columns, image_column, global_batch_size)` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/hf_data_processing.py#L45) — pipeline for multimodal SFT with HF dataset

