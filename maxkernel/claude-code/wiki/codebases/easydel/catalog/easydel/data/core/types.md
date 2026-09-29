---
title: 'Module: easydel/data/core/types.py'
type: catalog
provenance: extracted
module: easydel/data/core/types.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.core.types`/
symbols:
  DatasetType.infer_from_path: DatasetType#infer_from_path().
  BaseDatasetInform.__post_init__: BaseDatasetInform#__post_init__().
  BaseDatasetInform: BaseDatasetInform#
  TextDatasetInform: TextDatasetInform#
  DatasetMixture.load_mixture: DatasetMixture#load_mixture().
  VisualDatasetInform: VisualDatasetInform#
  DatasetMixture.informs: DatasetMixture#informs.
  DatasetType: DatasetType#
  DatasetMixture.cache_dir: DatasetMixture#cache_dir.
  DatasetMixture: DatasetMixture#
  BaseDatasetInform.data_files: BaseDatasetInform#data_files.
  DatasetMixture.batch_size: DatasetMixture#batch_size.
  DatasetMixture.seed: DatasetMixture#seed.
  DatasetType.from_string: DatasetType#from_string().
  DatasetMixture.shuffle_buffer_size: DatasetMixture#shuffle_buffer_size.
  DatasetMixture.mixture_weights: DatasetMixture#mixture_weights.
  DatasetMixture.__post_init__: DatasetMixture#__post_init__().
  DatasetMixture.save_mixture: DatasetMixture#save_mixture().
  DatasetMixture.pack_seq_length: DatasetMixture#pack_seq_length.
  DatasetMixture.pack_eos_token_id: DatasetMixture#pack_eos_token_id.
  DatasetMixture.pack_shuffle: DatasetMixture#pack_shuffle.
  DatasetMixture.pack_shuffle_buffer_factor: DatasetMixture#pack_shuffle_buffer_factor.
  DatasetMixture.tokenize_callback: DatasetMixture#tokenize_callback.
  DatasetLoadError: DatasetLoadError#
  DatasetType.JSON: DatasetType#JSON.
  DatasetType.PARQUET: DatasetType#PARQUET.
  DatasetType.CSV: DatasetType#CSV.
  DatasetType.ARROW: DatasetType#ARROW.
  DatasetType.TSV: DatasetType#TSV.
  DatasetType.TXT: DatasetType#TXT.
  BaseDatasetInform.dataset_split_name: BaseDatasetInform#dataset_split_name.
  BaseDatasetInform.format_callback: BaseDatasetInform#format_callback.
  BaseDatasetInform.format_fields: BaseDatasetInform#format_fields.
  BaseDatasetInform.get_str_type: BaseDatasetInform#get_str_type().
  TextDatasetInform.content_field: TextDatasetInform#content_field.
  DatasetMixture.streaming: DatasetMixture#streaming.
  DatasetMixture.text_target_field: DatasetMixture#text_target_field.
  DatasetMixture._dict_from_json_file: DatasetMixture#_dict_from_json_file().
  DatasetMixture.to_json_string: DatasetMixture#to_json_string().
  DatasetMixture.build: DatasetMixture#build().
  DatasetType.HF: DatasetType#HF.
  BaseDatasetInform.num_rows: BaseDatasetInform#num_rows.
  BaseDatasetInform.split: BaseDatasetInform#split.
  TextDatasetInform.additional_fields: TextDatasetInform#additional_fields.
  TextDatasetInform.preprocessing_fn: TextDatasetInform#preprocessing_fn.
  VisualDatasetInform.pixel_field: VisualDatasetInform#pixel_field.
  VisualDatasetInform.content_field: VisualDatasetInform#content_field.
  VisualDatasetInform.image_size: VisualDatasetInform#image_size.
  VisualDatasetInform.preprocessing_fn: VisualDatasetInform#preprocessing_fn.
  DatasetMixture.image_target_field: DatasetMixture#image_target_field.
  DatasetMixture.pack_tokens: DatasetMixture#pack_tokens.
  DatasetMixture.tokens_field_name: DatasetMixture#tokens_field_name.
  DatasetMixture.dask_storage_options: DatasetMixture#dask_storage_options.
  DatasetMixture.pack_on_the_fly: DatasetMixture#pack_on_the_fly.
  DatasetMixture.prefetch_workers: DatasetMixture#prefetch_workers.
  DatasetMixture.prefetch_buffer_size: DatasetMixture#prefetch_buffer_size.
  DatasetMixture.cloud_max_retries: DatasetMixture#cloud_max_retries.
  DatasetMixture.cloud_retry_delay: DatasetMixture#cloud_retry_delay.
  DatasetMixture.cache_remote_files: DatasetMixture#cache_remote_files.
  DatasetMixture.cache_expiry_seconds: DatasetMixture#cache_expiry_seconds.
  DatasetMixture.block_mixture: DatasetMixture#block_mixture.
  DatasetMixture.mixture_block_size: DatasetMixture#mixture_block_size.
  DatasetMixture.stop_strategy: DatasetMixture#stop_strategy.
---
# Module: [`easydel/data/core/types.py`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py)

## Classes
### `BaseDatasetInform`
- def: [`easydel/data/core/types.py:90`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L90)
- doc: Base configuration class for dataset information.
- signature: `class BaseDatasetInform:`
- members:
  - `__post_init__(self)` — [`L114`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L114) — Validate and auto-detect dataset type from file extension if not provided.
  - `get_str_type(self)` — [`L150`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L150) — Get string representation of dataset type.
  - `data_files` — [`L107`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L107)
  - `dataset_split_name` — [`L109`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L109)
  - `format_callback` — [`L111`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L111)
  - `format_fields` — [`L112`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L112)
  - `num_rows` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L108)
  - `split` — [`L110`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L110)
- uses (calls/refs, reference-scoped): [`infer_from_path`](types.md#DatasetType.infer_from_path), [`TextDatasetInform`](types.md#TextDatasetInform), [`VisualDatasetInform`](types.md#VisualDatasetInform), [`DatasetType`](types.md#DatasetType), [`from_string`](types.md#DatasetType.from_string)
- used by: [`build_dataset`](../execution/pipeline.md#build_dataset), [`TextDatasetInform`](types.md#TextDatasetInform), [`VisualDatasetInform`](types.md#VisualDatasetInform)

### `DatasetLoadError`  ·  implements/extends Exception
- def: [`easydel/data/core/types.py:378`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L378)
- doc: Exception raised when dataset loading fails.
- signature: `class DatasetLoadError(Exception):`

### `DatasetMixture`
- def: [`easydel/data/core/types.py:199`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L199)
- doc: Configuration for mixing multiple datasets with various strategies.
- signature: `class DatasetMixture:`
- members:
  - `__post_init__(self)` — [`L294`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L294) — Ensure cache directory exists, converting string paths to ePath.
  - `_dict_from_json_file(cls, json_file: str | os.PathLike)` — [`L301`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L301) — Load dictionary from JSON file.
  - `build(self)` — [`L354`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L354) — Build the dataset using this mixture configuration.
  - `load_mixture(cls, json_file: str | os.PathLike)` — [`L324`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L324) — Load DatasetMixture configuration from JSON file.
  - `save_mixture(self, json_file_path: str | os.PathLike)` — [`L345`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L345) — Save DatasetMixture configuration to JSON file.
  - `to_json_string(self)` — [`L314`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L314) — Serialize configuration to JSON string.
  - `batch_size` — [`L263`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L263)
  - `block_mixture` — [`L288`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L288)
  - `cache_dir` — [`L259`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L259)
  - `cache_expiry_seconds` — [`L286`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L286)
  - `cache_remote_files` — [`L285`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L285)
  - `cloud_max_retries` — [`L283`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L283)
  - `cloud_retry_delay` — [`L284`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L284)
  - `dask_storage_options` — [`L273`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L273)
  - `image_target_field` — [`L262`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L262)
  - `informs` — [`L258`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L258)
  - `mixture_block_size` — [`L289`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L289)
  - `mixture_weights` — [`L292`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L292)
  - `pack_eos_token_id` — [`L270`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L270)
  - `pack_on_the_fly` — [`L275`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L275)
  - `pack_seq_length` — [`L269`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L269)
  - `pack_shuffle` — [`L271`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L271)
  - `pack_shuffle_buffer_factor` — [`L272`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L272)
  - `pack_tokens` — [`L267`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L267)
  - `prefetch_buffer_size` — [`L280`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L280)
  - `prefetch_workers` — [`L279`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L279)
  - `seed` — [`L265`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L265)
  - `shuffle_buffer_size` — [`L264`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L264)
  - `stop_strategy` — [`L290`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L290)
  - `streaming` — [`L260`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L260)
  - `text_target_field` — [`L261`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L261)
  - `tokenize_callback` — [`L276`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L276)
  - `tokens_field_name` — [`L268`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L268)
- uses (calls/refs, reference-scoped): [`TextDatasetInform`](types.md#TextDatasetInform), [`VisualDatasetInform`](types.md#VisualDatasetInform), [`get_cache_dir`](../../utils/helpers.md#get_cache_dir)
- used by: [`build_dataset`](../execution/pipeline.md#build_dataset), [`build_dataset`](../../infra/elarge/builders.md#build_dataset)

### `DatasetType`  ·  implements/extends StrEnum
- def: [`easydel/data/core/types.py:35`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L35)
- doc: Enumeration of supported dataset file formats.
- signature: `class DatasetType(StrEnum):`
- members:
  - `from_string(cls, value: str)` — [`L51`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L51) — Convert string to DatasetType enum value.
  - `infer_from_path(cls, path: str)` — [`L66`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L66) — Infer dataset type from file path extension.
  - `ARROW` — [`L45`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L45)
  - `CSV` — [`L44`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L44)
  - `HF` — [`L46`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L46)
  - `JSON` — [`L42`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L42)
  - `PARQUET` — [`L43`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L43)
  - `TSV` — [`L47`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L47)
  - `TXT` — [`L48`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L48)
- used by: [`__post_init__`](types.md#BaseDatasetInform.__post_init__), [`BaseDatasetInform`](types.md#BaseDatasetInform)

### `TextDatasetInform`  ·  implements/extends BaseDatasetInform
- def: [`easydel/data/core/types.py:163`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L163)
- doc: Configuration for text-only datasets.
- signature: `class TextDatasetInform(BaseDatasetInform):`
- members:
  - `additional_fields` — [`L175`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L175)
  - `content_field` — [`L174`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L174)
  - `preprocessing_fn` — [`L176`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L176)
- uses (calls/refs, reference-scoped): [`BaseDatasetInform`](types.md#BaseDatasetInform)
- used by: [`build_dataset`](../execution/pipeline.md#build_dataset), [`to_data_mixture_kwargs`](../../infra/elarge/builders.md#to_data_mixture_kwargs), [`BaseDatasetInform`](types.md#BaseDatasetInform), [`load_mixture`](types.md#DatasetMixture.load_mixture), [`informs`](types.md#DatasetMixture.informs)

### `VisualDatasetInform`  ·  implements/extends BaseDatasetInform
- def: [`easydel/data/core/types.py:180`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L180)
- doc: Configuration for visual/multimodal datasets.
- signature: `class VisualDatasetInform(BaseDatasetInform):`
- members:
  - `content_field` — [`L193`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L193)
  - `image_size` — [`L194`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L194)
  - `pixel_field` — [`L192`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L192)
  - `preprocessing_fn` — [`L195`](../../../../../../../raw/code/EasyDeL/easydel/data/core/types.py#L195)
- uses (calls/refs, reference-scoped): [`BaseDatasetInform`](types.md#BaseDatasetInform)
- used by: [`to_data_mixture_kwargs`](../../infra/elarge/builders.md#to_data_mixture_kwargs), [`BaseDatasetInform`](types.md#BaseDatasetInform), [`load_mixture`](types.md#DatasetMixture.load_mixture), [`informs`](types.md#DatasetMixture.informs)

