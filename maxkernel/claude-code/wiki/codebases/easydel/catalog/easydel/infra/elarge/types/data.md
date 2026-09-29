---
title: 'Module: easydel/infra/elarge/types/data.py'
type: catalog
provenance: extracted
module: easydel/infra/elarge/types/data.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.elarge.types.data`/
symbols:
  TextDatasetInformCfg: TextDatasetInformCfg#
  VisualDatasetInformCfg: VisualDatasetInformCfg#
  DataMixtureCfg: DataMixtureCfg#
  DatasetMixtureCfg.informs: DatasetMixtureCfg#informs.
  DatasetMixtureCfg: DatasetMixtureCfg#
  DataMixtureCfg.tokenization: DataMixtureCfg#tokenization.
  DataMixtureCfg.save: DataMixtureCfg#save.
  TokenizationCfg: TokenizationCfg#
  DatasetSaveCfg: DatasetSaveCfg#
  TextDatasetInformCfg.data_files: TextDatasetInformCfg#data_files.
  TextDatasetInformCfg.dataset_split_name: TextDatasetInformCfg#dataset_split_name.
  TextDatasetInformCfg.split: TextDatasetInformCfg#split.
  TextDatasetInformCfg.content_field: TextDatasetInformCfg#content_field.
  TextDatasetInformCfg.additional_fields: TextDatasetInformCfg#additional_fields.
  TextDatasetInformCfg.num_rows: TextDatasetInformCfg#num_rows.
  TextDatasetInformCfg.format_callback: TextDatasetInformCfg#format_callback.
  TextDatasetInformCfg.format_fields: TextDatasetInformCfg#format_fields.
  TextDatasetInformCfg.preprocessing_fn: TextDatasetInformCfg#preprocessing_fn.
  VisualDatasetInformCfg.data_files: VisualDatasetInformCfg#data_files.
  VisualDatasetInformCfg.dataset_split_name: VisualDatasetInformCfg#dataset_split_name.
  VisualDatasetInformCfg.split: VisualDatasetInformCfg#split.
  VisualDatasetInformCfg.pixel_field: VisualDatasetInformCfg#pixel_field.
  VisualDatasetInformCfg.content_field: VisualDatasetInformCfg#content_field.
  VisualDatasetInformCfg.image_size: VisualDatasetInformCfg#image_size.
  VisualDatasetInformCfg.num_rows: VisualDatasetInformCfg#num_rows.
  VisualDatasetInformCfg.format_callback: VisualDatasetInformCfg#format_callback.
  VisualDatasetInformCfg.format_fields: VisualDatasetInformCfg#format_fields.
  VisualDatasetInformCfg.preprocessing_fn: VisualDatasetInformCfg#preprocessing_fn.
  TokenizationCfg.tokenizer: TokenizationCfg#tokenizer.
  TokenizationCfg.max_length: TokenizationCfg#max_length.
  TokenizationCfg.truncation: TokenizationCfg#truncation.
  TokenizationCfg.padding: TokenizationCfg#padding.
  TokenizationCfg.add_special_tokens: TokenizationCfg#add_special_tokens.
  TokenizationCfg.return_attention_mask: TokenizationCfg#return_attention_mask.
  TokenizationCfg.text_field: TokenizationCfg#text_field.
  TokenizationCfg.output_field: TokenizationCfg#output_field.
  TokenizationCfg.num_proc: TokenizationCfg#num_proc.
  TokenizationCfg.batched: TokenizationCfg#batched.
  TokenizationCfg.batch_size: TokenizationCfg#batch_size.
  TokenizationCfg.remove_columns: TokenizationCfg#remove_columns.
  TokenizationCfg.keep_in_memory: TokenizationCfg#keep_in_memory.
  DatasetSaveCfg.output_path: DatasetSaveCfg#output_path.
  DatasetSaveCfg.format: DatasetSaveCfg#format.
  DatasetSaveCfg.num_shards: DatasetSaveCfg#num_shards.
  DatasetSaveCfg.compression: DatasetSaveCfg#compression.
  DatasetSaveCfg.max_shard_size: DatasetSaveCfg#max_shard_size.
  DatasetSaveCfg.overwrite: DatasetSaveCfg#overwrite.
  DatasetSaveCfg.push_to_hub: DatasetSaveCfg#push_to_hub.
  DatasetSaveCfg.hub_repo_id: DatasetSaveCfg#hub_repo_id.
  DatasetSaveCfg.hub_private: DatasetSaveCfg#hub_private.
  DatasetSaveCfg.hub_token: DatasetSaveCfg#hub_token.
  DatasetMixtureCfg.cache_dir: DatasetMixtureCfg#cache_dir.
  DatasetMixtureCfg.streaming: DatasetMixtureCfg#streaming.
  DatasetMixtureCfg.text_target_field: DatasetMixtureCfg#text_target_field.
  DatasetMixtureCfg.image_target_field: DatasetMixtureCfg#image_target_field.
  DatasetMixtureCfg.batch_size: DatasetMixtureCfg#batch_size.
  DatasetMixtureCfg.shuffle_buffer_size: DatasetMixtureCfg#shuffle_buffer_size.
  DatasetMixtureCfg.seed: DatasetMixtureCfg#seed.
  DatasetMixtureCfg.pack_tokens: DatasetMixtureCfg#pack_tokens.
  DatasetMixtureCfg.tokens_field_name: DatasetMixtureCfg#tokens_field_name.
  DatasetMixtureCfg.pack_seq_length: DatasetMixtureCfg#pack_seq_length.
  DatasetMixtureCfg.pack_eos_token_id: DatasetMixtureCfg#pack_eos_token_id.
  DatasetMixtureCfg.pack_shuffle: DatasetMixtureCfg#pack_shuffle.
  DatasetMixtureCfg.pack_shuffle_buffer_factor: DatasetMixtureCfg#pack_shuffle_buffer_factor.
  DatasetMixtureCfg.dask_storage_options: DatasetMixtureCfg#dask_storage_options.
  DatasetMixtureCfg.pack_on_the_fly: DatasetMixtureCfg#pack_on_the_fly.
  DatasetMixtureCfg.tokenize_callback: DatasetMixtureCfg#tokenize_callback.
  DatasetMixtureCfg.prefetch_workers: DatasetMixtureCfg#prefetch_workers.
  DatasetMixtureCfg.prefetch_buffer_size: DatasetMixtureCfg#prefetch_buffer_size.
  DatasetMixtureCfg.cloud_max_retries: DatasetMixtureCfg#cloud_max_retries.
  DatasetMixtureCfg.cloud_retry_delay: DatasetMixtureCfg#cloud_retry_delay.
  DatasetMixtureCfg.cache_remote_files: DatasetMixtureCfg#cache_remote_files.
  DatasetMixtureCfg.cache_expiry_seconds: DatasetMixtureCfg#cache_expiry_seconds.
  DatasetMixtureCfg.block_mixture: DatasetMixtureCfg#block_mixture.
  DatasetMixtureCfg.mixture_block_size: DatasetMixtureCfg#mixture_block_size.
  DatasetMixtureCfg.stop_strategy: DatasetMixtureCfg#stop_strategy.
  DatasetMixtureCfg.mixture_weights: DatasetMixtureCfg#mixture_weights.
  DataMixtureCfg.use_sharded_source: DataMixtureCfg#use_sharded_source.
  DataMixtureCfg.use_fast_loader: DataMixtureCfg#use_fast_loader.
  DataMixtureCfg.num_workers: DataMixtureCfg#num_workers.
  DataMixtureCfg.prefetch_size: DataMixtureCfg#prefetch_size.
  DataMixtureCfg.enable_caching: DataMixtureCfg#enable_caching.
---
# Module: [`easydel/infra/elarge/types/data.py`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py)

## Classes
### `DataMixtureCfg`  ·  implements/extends DatasetMixtureCfg
- def: [`easydel/infra/elarge/types/data.py:245`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L245)
- doc: Extended dataset mixture configuration with EasyDeL-specific extras.
- signature: `class DataMixtureCfg(DatasetMixtureCfg, total=False):`
- members:
  - `enable_caching` — [`L272`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L272)
  - `num_workers` — [`L270`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L270)
  - `prefetch_size` — [`L271`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L271)
  - `save` — [`L263`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L263)
  - `tokenization` — [`L260`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L260)
  - `use_fast_loader` — [`L269`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L269)
  - `use_sharded_source` — [`L266`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L266)
- uses (calls/refs, reference-scoped): [`DatasetMixtureCfg`](data.md#DatasetMixtureCfg), [`DatasetSaveCfg`](data.md#DatasetSaveCfg), [`TokenizationCfg`](data.md#TokenizationCfg)
- used by: [`DatasetMixtureCfg`](data.md#DatasetMixtureCfg), [`mixture`](root.md#eLMConfig.mixture)

### `DatasetMixtureCfg`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/data.py:164`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L164)
- doc: Configuration for mixing multiple datasets together.
- signature: `class DatasetMixtureCfg(TypedDict, total=False):`
- members:
  - `batch_size` — [`L211`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L211)
  - `block_mixture` — [`L239`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L239)
  - `cache_dir` — [`L207`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L207)
  - `cache_expiry_seconds` — [`L236`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L236)
  - `cache_remote_files` — [`L235`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L235)
  - `cloud_max_retries` — [`L233`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L233)
  - `cloud_retry_delay` — [`L234`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L234)
  - `dask_storage_options` — [`L222`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L222)
  - `image_target_field` — [`L210`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L210)
  - `informs` — [`L206`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L206)
  - `mixture_block_size` — [`L240`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L240)
  - `mixture_weights` — [`L242`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L242)
  - `pack_eos_token_id` — [`L219`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L219)
  - `pack_on_the_fly` — [`L225`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L225)
  - `pack_seq_length` — [`L218`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L218)
  - `pack_shuffle` — [`L220`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L220)
  - `pack_shuffle_buffer_factor` — [`L221`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L221)
  - `pack_tokens` — [`L216`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L216)
  - `prefetch_buffer_size` — [`L230`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L230)
  - `prefetch_workers` — [`L229`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L229)
  - `seed` — [`L213`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L213)
  - `shuffle_buffer_size` — [`L212`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L212)
  - `stop_strategy` — [`L241`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L241)
  - `streaming` — [`L208`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L208)
  - `text_target_field` — [`L209`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L209)
  - `tokenize_callback` — [`L226`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L226)
  - `tokens_field_name` — [`L217`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L217)
- uses (calls/refs, reference-scoped): [`TextDatasetInformCfg`](data.md#TextDatasetInformCfg), [`VisualDatasetInformCfg`](data.md#VisualDatasetInformCfg), [`DataMixtureCfg`](data.md#DataMixtureCfg)
- used by: [`DataMixtureCfg`](data.md#DataMixtureCfg)

### `DatasetSaveCfg`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/data.py:132`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L132)
- doc: Configuration for saving processed datasets to disk or hub.
- signature: `class DatasetSaveCfg(TypedDict, total=False):`
- members:
  - `compression` — [`L155`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L155)
  - `format` — [`L153`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L153)
  - `hub_private` — [`L160`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L160)
  - `hub_repo_id` — [`L159`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L159)
  - `hub_token` — [`L161`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L161)
  - `max_shard_size` — [`L156`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L156)
  - `num_shards` — [`L154`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L154)
  - `output_path` — [`L152`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L152)
  - `overwrite` — [`L157`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L157)
  - `push_to_hub` — [`L158`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L158)
- used by: [`save`](data.md#DataMixtureCfg.save)

### `TextDatasetInformCfg`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/data.py:28`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L28)
- doc: Configuration for a single text dataset source in the data pipeline.
- signature: `class TextDatasetInformCfg(TypedDict, total=False):`
- members:
  - `additional_fields` — [`L56`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L56)
  - `content_field` — [`L55`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L55)
  - `data_files` — [`L52`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L52)
  - `dataset_split_name` — [`L53`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L53)
  - `format_callback` — [`L58`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L58)
  - `format_fields` — [`L59`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L59)
  - `num_rows` — [`L57`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L57)
  - `preprocessing_fn` — [`L60`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L60)
  - `split` — [`L54`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L54)
- uses (calls/refs, reference-scoped): [`DatasetTypeLike`](aliases.md#DatasetTypeLike)
- used by: [`set_mixture`](../model.md#eLargeModel.set_mixture), [`informs`](data.md#DatasetMixtureCfg.informs)

### `TokenizationCfg`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/data.py:95`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L95)
- doc: Configuration for tokenizing dataset text into model-ready token IDs.
- signature: `class TokenizationCfg(TypedDict, total=False):`
- members:
  - `add_special_tokens` — [`L121`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L121)
  - `batch_size` — [`L127`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L127)
  - `batched` — [`L126`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L126)
  - `keep_in_memory` — [`L129`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L129)
  - `max_length` — [`L118`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L118)
  - `num_proc` — [`L125`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L125)
  - `output_field` — [`L124`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L124)
  - `padding` — [`L120`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L120)
  - `remove_columns` — [`L128`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L128)
  - `return_attention_mask` — [`L122`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L122)
  - `text_field` — [`L123`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L123)
  - `tokenizer` — [`L117`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L117)
  - `truncation` — [`L119`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L119)
- used by: [`tokenization`](data.md#DataMixtureCfg.tokenization)

### `VisualDatasetInformCfg`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/types/data.py:63`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L63)
- doc: Configuration for a single visual/image dataset source.
- signature: `class VisualDatasetInformCfg(TypedDict, total=False):`
- members:
  - `content_field` — [`L87`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L87)
  - `data_files` — [`L83`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L83)
  - `dataset_split_name` — [`L84`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L84)
  - `format_callback` — [`L90`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L90)
  - `format_fields` — [`L91`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L91)
  - `image_size` — [`L88`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L88)
  - `num_rows` — [`L89`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L89)
  - `pixel_field` — [`L86`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L86)
  - `preprocessing_fn` — [`L92`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L92)
  - `split` — [`L85`](../../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/types/data.py#L85)
- uses (calls/refs, reference-scoped): [`DatasetTypeLike`](aliases.md#DatasetTypeLike)
- used by: [`set_mixture`](../model.md#eLargeModel.set_mixture), [`informs`](data.md#DatasetMixtureCfg.informs)

