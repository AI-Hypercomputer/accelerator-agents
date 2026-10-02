---
title: 'Module: easydel/data/core/config.py'
type: catalog
provenance: extracted
module: easydel/data/core/config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.core.config`/
symbols:
  merge_tokenizer_config: merge_tokenizer_config().
  PipelineConfig.validate: PipelineConfig#validate().
  TokenizerConfig: TokenizerConfig#
  DatasetConfig: DatasetConfig#
  TokenizerConfig.to_dict: TokenizerConfig#to_dict().
  TokenizeStageConfig: TokenizeStageConfig#
  SaveStageConfig: SaveStageConfig#
  PipelineConfig.datasets: PipelineConfig#datasets.
  WeightSchedulePoint.weights: WeightSchedulePoint#weights.
  MixStageConfig: MixStageConfig#
  PackStageConfig: PackStageConfig#
  LoadStageConfig: LoadStageConfig#
  PipelineConfig: PipelineConfig#
  DatasetConfig.get_tokenizer_config: DatasetConfig#get_tokenizer_config().
  PipelineConfig.get_dataset_by_name: PipelineConfig#get_dataset_by_name().
  WeightSchedulePoint.step: WeightSchedulePoint#step.
  PipelineConfig.mix: PipelineConfig#mix.
  TokenizerConfig.name_or_path: TokenizerConfig#name_or_path.
  DatasetConfig.tokenizer: DatasetConfig#tokenizer.
  DatasetConfig.name: DatasetConfig#name.
  MixStageConfig.weight_schedule: MixStageConfig#weight_schedule.
  get_dataset_name: get_dataset_name().
  PipelineConfig.tokenize: PipelineConfig#tokenize.
  PipelineConfig.save: PipelineConfig#save.
  WeightSchedulePoint: WeightSchedulePoint#
  MixStageConfig.weights: MixStageConfig#weights.
  PipelineConfig.__post_init__: PipelineConfig#__post_init__().
  PipelineConfig.pack: PipelineConfig#pack.
  PipelineConfig.load: PipelineConfig#load.
  TokenizerConfig.max_length: TokenizerConfig#max_length.
  SourceStageConfig: SourceStageConfig#
  CacheStageConfig: CacheStageConfig#
  RayConfig: RayConfig#
  ObservabilityConfig: ObservabilityConfig#
  PipelineConfig.source: PipelineConfig#source.
  PipelineConfig.cache: PipelineConfig#cache.
  TokenizerConfig.truncation: TokenizerConfig#truncation.
  TokenizerConfig.padding: TokenizerConfig#padding.
  TokenizerConfig.add_special_tokens: TokenizerConfig#add_special_tokens.
  TokenizerConfig.return_attention_mask: TokenizerConfig#return_attention_mask.
  DatasetConfig.data_files: DatasetConfig#data_files.
  DatasetConfig.save_format: DatasetConfig#save_format.
  DatasetConfig.content_field: DatasetConfig#content_field.
  PipelineConfig.default_tokenizer: PipelineConfig#default_tokenizer.
  DatasetConfig.__post_init__: DatasetConfig#__post_init__().
  WeightSchedulePoint.__post_init__: WeightSchedulePoint#__post_init__().
  MixStageConfig.__post_init__: MixStageConfig#__post_init__().
  PipelineConfig.ray: PipelineConfig#ray.
  PipelineConfig.observability: PipelineConfig#observability.
  TokenizerConfig.trust_remote_code: TokenizerConfig#trust_remote_code.
  DatasetConfig.split: DatasetConfig#split.
  DatasetConfig.dataset_split_name: DatasetConfig#dataset_split_name.
  DatasetConfig.save_path: DatasetConfig#save_path.
  DatasetConfig.additional_fields: DatasetConfig#additional_fields.
  DatasetConfig.format_callback: DatasetConfig#format_callback.
  DatasetConfig.format_fields: DatasetConfig#format_fields.
  TokenizeStageConfig.default_tokenizer: TokenizeStageConfig#default_tokenizer.
  MixStageConfig.block_size: MixStageConfig#block_size.
  PackStageConfig.strategy: PackStageConfig#strategy.
  LoadStageConfig.batch_size: LoadStageConfig#batch_size.
  SaveStageConfig.enabled: SaveStageConfig#enabled.
  SaveStageConfig.format: SaveStageConfig#format.
  SaveStageConfig.hub_repo_id: SaveStageConfig#hub_repo_id.
  PipelineConfig.seed: PipelineConfig#seed.
  DatasetConfig.tokenizer_kwargs: DatasetConfig#tokenizer_kwargs.
  TokenizeStageConfig.max_length: TokenizeStageConfig#max_length.
  MixStageConfig.weight_schedule_type: MixStageConfig#weight_schedule_type.
  MixStageConfig.stop_strategy: MixStageConfig#stop_strategy.
  MixStageConfig.seed: MixStageConfig#seed.
  PackStageConfig.enabled: PackStageConfig#enabled.
  PackStageConfig.seq_length: PackStageConfig#seq_length.
  PackStageConfig.eos_token_id: PackStageConfig#eos_token_id.
  PackStageConfig.pad_token_id: PackStageConfig#pad_token_id.
  PackStageConfig.num_packers: PackStageConfig#num_packers.
  PackStageConfig.include_segment_ids: PackStageConfig#include_segment_ids.
  PackStageConfig.shuffle_packed: PackStageConfig#shuffle_packed.
  PackStageConfig.shuffle_buffer_factor: PackStageConfig#shuffle_buffer_factor.
  LoadStageConfig.prefetch_enabled: LoadStageConfig#prefetch_enabled.
  LoadStageConfig.prefetch_workers: LoadStageConfig#prefetch_workers.
  LoadStageConfig.prefetch_buffer_size: LoadStageConfig#prefetch_buffer_size.
  LoadStageConfig.shuffle_buffer_size: LoadStageConfig#shuffle_buffer_size.
  LoadStageConfig.drop_last: LoadStageConfig#drop_last.
  SaveStageConfig.output_dir: SaveStageConfig#output_dir.
  SaveStageConfig.num_shards: SaveStageConfig#num_shards.
  SaveStageConfig.compression: SaveStageConfig#compression.
  SaveStageConfig.max_shard_size: SaveStageConfig#max_shard_size.
  SaveStageConfig.overwrite: SaveStageConfig#overwrite.
  SaveStageConfig.push_to_hub: SaveStageConfig#push_to_hub.
  SaveStageConfig.hub_private: SaveStageConfig#hub_private.
  SaveStageConfig.hub_token: SaveStageConfig#hub_token.
  PipelineConfig.streaming: PipelineConfig#streaming.
  DatasetConfig.num_rows: DatasetConfig#num_rows.
  DatasetConfig.cache_path: DatasetConfig#cache_path.
  DatasetConfig.cache_enabled: DatasetConfig#cache_enabled.
  DatasetConfig.shard_column: DatasetConfig#shard_column.
  DatasetConfig.num_shards: DatasetConfig#num_shards.
  SourceStageConfig.streaming: SourceStageConfig#streaming.
  SourceStageConfig.cloud_max_retries: SourceStageConfig#cloud_max_retries.
  SourceStageConfig.cloud_retry_delay: SourceStageConfig#cloud_retry_delay.
  SourceStageConfig.dask_storage_options: SourceStageConfig#dask_storage_options.
  TokenizeStageConfig.batch_size: TokenizeStageConfig#batch_size.
  TokenizeStageConfig.num_workers: TokenizeStageConfig#num_workers.
  TokenizeStageConfig.cache_tokenized: TokenizeStageConfig#cache_tokenized.
  TokenizeStageConfig.remove_columns: TokenizeStageConfig#remove_columns.
  CacheStageConfig.enabled: CacheStageConfig#enabled.
  CacheStageConfig.cache_type: CacheStageConfig#cache_type.
  CacheStageConfig.cache_dir: CacheStageConfig#cache_dir.
  CacheStageConfig.memory_cache_size: CacheStageConfig#memory_cache_size.
  CacheStageConfig.disk_cache_expiry: CacheStageConfig#disk_cache_expiry.
  CacheStageConfig.compression: CacheStageConfig#compression.
  CacheStageConfig.hash_fn: CacheStageConfig#hash_fn.
  LoadStageConfig.prefetch_to_device: LoadStageConfig#prefetch_to_device.
  RayConfig.enabled: RayConfig#enabled.
  RayConfig.num_workers: RayConfig#num_workers.
  RayConfig.resources_per_worker: RayConfig#resources_per_worker.
  RayConfig.use_gpu: RayConfig#use_gpu.
  RayConfig.object_store_memory: RayConfig#object_store_memory.
  ObservabilityConfig.progress_enabled: ObservabilityConfig#progress_enabled.
  ObservabilityConfig.progress_type: ObservabilityConfig#progress_type.
  ObservabilityConfig.metrics_enabled: ObservabilityConfig#metrics_enabled.
  ObservabilityConfig.log_level: ObservabilityConfig#log_level.
  ObservabilityConfig.log_interval: ObservabilityConfig#log_interval.
---
# Module: [`easydel/data/core/config.py`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py)

## Classes
### `CacheStageConfig`
- def: [`easydel/data/core/config.py:192`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L192)
- doc: Configuration for the multi-layer caching stage (TreeCache-style).
- signature: `class CacheStageConfig:`
- members:
  - `cache_dir` — [`L207`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L207)
  - `cache_type` — [`L206`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L206)
  - `compression` — [`L210`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L210)
  - `disk_cache_expiry` — [`L209`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L209)
  - `enabled` — [`L205`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L205)
  - `hash_fn` — [`L211`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L211)
  - `memory_cache_size` — [`L208`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L208)
- used by: [`cache`](config.md#PipelineConfig.cache)

### `DatasetConfig`
- def: [`easydel/data/core/config.py:71`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L71)
- doc: Configuration for a single dataset in the pipeline.
- signature: `class DatasetConfig:`
- members:
  - `__post_init__(self)` — [`L131`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L131) — Validate configuration after initialization.
  - `get_tokenizer_config(self)` — [`L136`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L136) — Get tokenizer configuration, normalizing string to TokenizerConfig.
  - `additional_fields` — [`L123`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L123)
  - `cache_enabled` — [`L115`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L115)
  - `cache_path` — [`L114`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L114)
  - `content_field` — [`L122`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L122)
  - `data_files` — [`L98`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L98)
  - `dataset_split_name` — [`L107`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L107)
  - `format_callback` — [`L124`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L124)
  - `format_fields` — [`L125`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L125)
  - `name` — [`L101`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L101)
  - `num_rows` — [`L106`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L106)
  - `num_shards` — [`L129`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L129)
  - `save_format` — [`L119`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L119)
  - `save_path` — [`L118`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L118)
  - `shard_column` — [`L128`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L128)
  - `split` — [`L105`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L105)
  - `tokenizer` — [`L110`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L110)
  - `tokenizer_kwargs` — [`L111`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L111)
- uses (calls/refs, reference-scoped): [`TokenizerConfig`](config.md#TokenizerConfig), [`name_or_path`](config.md#TokenizerConfig.name_or_path)
- used by: [`process`](../execution/save.md#SaveStage.process), [`tokenize_and_save`](../execution/pipeline.md#tokenize_and_save), [`from_config`](../execution/pipeline.md#Pipeline.from_config), [`process`](../transforms/tokenize.md#TokenizeStage.process), [`create_source`](../sources/base.md#create_source), [`tokenize_dataset_config`](../transforms/tokenize.md#tokenize_dataset_config), [`source`](../execution/pipeline.md#Pipeline.source), [`merge_tokenizer_config`](config.md#merge_tokenizer_config), [`validate`](config.md#PipelineConfig.validate), [`create_pipeline`](../execution/pipeline.md#create_pipeline), [`datasets`](config.md#PipelineConfig.datasets), [`get_dataset_by_name`](config.md#PipelineConfig.get_dataset_by_name), [`get_dataset_name`](config.md#get_dataset_name), [`__post_init__`](config.md#PipelineConfig.__post_init__)

### `LoadStageConfig`
- def: [`easydel/data/core/config.py:294`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L294)
- doc: Configuration for the data loading stage.
- signature: `class LoadStageConfig:`
- members:
  - `batch_size` — [`L307`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L307)
  - `drop_last` — [`L312`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L312)
  - `prefetch_buffer_size` — [`L310`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L310)
  - `prefetch_enabled` — [`L308`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L308)
  - `prefetch_to_device` — [`L313`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L313)
  - `prefetch_workers` — [`L309`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L309)
  - `shuffle_buffer_size` — [`L311`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L311)
- used by: [`from_config`](../execution/pipeline.md#Pipeline.from_config), [`process`](../execution/loader.md#LoadStage.process), [`load`](../execution/pipeline.md#Pipeline.load), [`_stage_config`](../execution/loader.md#LoadStage._stage_config), [`__init__`](../execution/loader.md#LoadStage.__init__), [`load`](config.md#PipelineConfig.load)

### `MixStageConfig`
- def: [`easydel/data/core/config.py:237`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L237)
- doc: Configuration for the dataset mixing stage.
- signature: `class MixStageConfig:`
- members:
  - `__post_init__(self)` — [`L258`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L258) — Validate weights if provided.
  - `block_size` — [`L254`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L254)
  - `seed` — [`L256`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L256)
  - `stop_strategy` — [`L255`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L255)
  - `weight_schedule` — [`L252`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L252)
  - `weight_schedule_type` — [`L253`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L253)
  - `weights` — [`L251`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L251)
- uses (calls/refs, reference-scoped): [`WeightSchedulePoint`](config.md#WeightSchedulePoint)
- used by: [`from_config`](../execution/pipeline.md#Pipeline.from_config), [`process`](../transforms/mixture.md#MixStage.process), [`mix`](../execution/pipeline.md#Pipeline.mix), [`validate`](config.md#PipelineConfig.validate), [`_stage_config`](../transforms/mixture.md#MixStage._stage_config), [`mix`](config.md#PipelineConfig.mix), [`__init__`](../transforms/mixture.md#MixStage.__init__)

### `ObservabilityConfig`
- def: [`easydel/data/core/config.py:367`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L367)
- doc: Configuration for pipeline observability.
- signature: `class ObservabilityConfig:`
- members:
  - `log_interval` — [`L382`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L382)
  - `log_level` — [`L381`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L381)
  - `metrics_enabled` — [`L380`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L380)
  - `progress_enabled` — [`L378`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L378)
  - `progress_type` — [`L379`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L379)
- used by: [`observability`](config.md#PipelineConfig.observability)

### `PackStageConfig`
- def: [`easydel/data/core/config.py:267`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L267)
- doc: Configuration for the token packing stage.
- signature: `class PackStageConfig:`
- members:
  - `enabled` — [`L282`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L282)
  - `eos_token_id` — [`L284`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L284)
  - `include_segment_ids` — [`L288`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L288)
  - `num_packers` — [`L287`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L287)
  - `pad_token_id` — [`L285`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L285)
  - `seq_length` — [`L283`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L283)
  - `shuffle_buffer_factor` — [`L290`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L290)
  - `shuffle_packed` — [`L289`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L289)
  - `strategy` — [`L286`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L286)
- used by: [`from_config`](../execution/pipeline.md#Pipeline.from_config), [`process`](../transforms/pack.md#PackStage.process), [`pack`](../execution/pipeline.md#Pipeline.pack), [`_stage_config`](../transforms/pack.md#PackStage._stage_config), [`__init__`](../transforms/pack.md#PackStage.__init__), [`pack`](config.md#PipelineConfig.pack)

### `PipelineConfig`
- def: [`easydel/data/core/config.py:386`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L386)
- doc: Main configuration for the data pipeline.
- signature: `class PipelineConfig:`
- members:
  - `__post_init__(self)` — [`L440`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L440) — Validate configuration after initialization.
  - `get_dataset_by_name(self, name: str)` — [`L450`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L450) — Get a dataset configuration by its unique name.
  - `validate(self)` — [`L464`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L464) — Validate the full pipeline configuration.
  - `cache` — [`L432`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L432)
  - `datasets` — [`L422`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L422)
  - `default_tokenizer` — [`L425`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L425)
  - `load` — [`L435`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L435)
  - `mix` — [`L433`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L433)
  - `observability` — [`L438`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L438)
  - `pack` — [`L434`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L434)
  - `ray` — [`L437`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L437)
  - `save` — [`L436`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L436)
  - `seed` — [`L427`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L427)
  - `source` — [`L430`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L430)
  - `streaming` — [`L426`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L426)
  - `tokenize` — [`L431`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L431)
- uses (calls/refs, reference-scoped): [`DatasetConfig`](config.md#DatasetConfig), [`TokenizeStageConfig`](config.md#TokenizeStageConfig), [`SaveStageConfig`](config.md#SaveStageConfig), [`LoadStageConfig`](config.md#LoadStageConfig), [`MixStageConfig`](config.md#MixStageConfig), [`PackStageConfig`](config.md#PackStageConfig), [`weights`](config.md#WeightSchedulePoint.weights), [`step`](config.md#WeightSchedulePoint.step), [`name`](config.md#DatasetConfig.name), [`weight_schedule`](config.md#MixStageConfig.weight_schedule), [`weights`](config.md#MixStageConfig.weights), [`CacheStageConfig`](config.md#CacheStageConfig), [`ObservabilityConfig`](config.md#ObservabilityConfig), [`RayConfig`](config.md#RayConfig), [`SourceStageConfig`](config.md#SourceStageConfig)
- used by: [`process`](../execution/save.md#SaveStage.process), [`tokenize_and_save`](../execution/pipeline.md#tokenize_and_save), [`from_config`](../execution/pipeline.md#Pipeline.from_config), [`process`](../transforms/tokenize.md#TokenizeStage.process), [`mix`](../execution/pipeline.md#Pipeline.mix), [`save`](../execution/pipeline.md#Pipeline.save), [`tokenize`](../execution/pipeline.md#Pipeline.tokenize), [`load`](../execution/pipeline.md#Pipeline.load), [`pack`](../execution/pipeline.md#Pipeline.pack), [`source`](../execution/pipeline.md#Pipeline.source), [`_context`](../execution/pipeline.md#Pipeline._context), [`create_pipeline`](../execution/pipeline.md#create_pipeline), [`config`](protocols.md#PipelineContext.config), [`__init__`](../execution/pipeline.md#Pipeline.__init__)

### `RayConfig`
- def: [`easydel/data/core/config.py:348`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L348)
- doc: Configuration for Ray distributed preprocessing.
- signature: `class RayConfig:`
- members:
  - `enabled` — [`L359`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L359)
  - `num_workers` — [`L360`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L360)
  - `object_store_memory` — [`L363`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L363)
  - `resources_per_worker` — [`L361`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L361)
  - `use_gpu` — [`L362`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L362)
- used by: [`ray`](config.md#PipelineConfig.ray)

### `SaveStageConfig`
- def: [`easydel/data/core/config.py:317`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L317)
- doc: Configuration for the save stage.
- signature: `class SaveStageConfig:`
- members:
  - `compression` — [`L338`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L338)
  - `enabled` — [`L334`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L334)
  - `format` — [`L336`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L336)
  - `hub_private` — [`L343`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L343)
  - `hub_repo_id` — [`L342`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L342)
  - `hub_token` — [`L344`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L344)
  - `max_shard_size` — [`L339`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L339)
  - `num_shards` — [`L337`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L337)
  - `output_dir` — [`L335`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L335)
  - `overwrite` — [`L340`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L340)
  - `push_to_hub` — [`L341`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L341)
- used by: [`process`](../execution/save.md#SaveStage.process), [`tokenize_and_save`](../execution/pipeline.md#tokenize_and_save), [`from_config`](../execution/pipeline.md#Pipeline.from_config), [`save`](../execution/pipeline.md#Pipeline.save), [`_push_to_hub`](../execution/save.md#SaveStage._push_to_hub), [`_stage_config`](../execution/save.md#SaveStage._stage_config), [`save`](config.md#PipelineConfig.save), [`__init__`](../execution/save.md#SaveStage.__init__)

### `SourceStageConfig`
- def: [`easydel/data/core/config.py:154`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L154)
- doc: Configuration for the source loading stage.
- signature: `class SourceStageConfig:`
- members:
  - `cloud_max_retries` — [`L165`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L165)
  - `cloud_retry_delay` — [`L166`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L166)
  - `dask_storage_options` — [`L167`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L167)
  - `streaming` — [`L164`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L164)
- used by: [`source`](config.md#PipelineConfig.source)

### `TokenizeStageConfig`
- def: [`easydel/data/core/config.py:171`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L171)
- doc: Configuration for the tokenization stage.
- signature: `class TokenizeStageConfig:`
- members:
  - `batch_size` — [`L185`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L185)
  - `cache_tokenized` — [`L187`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L187)
  - `default_tokenizer` — [`L183`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L183)
  - `max_length` — [`L184`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L184)
  - `num_workers` — [`L186`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L186)
  - `remove_columns` — [`L188`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L188)
- used by: [`tokenize_and_save`](../execution/pipeline.md#tokenize_and_save), [`from_config`](../execution/pipeline.md#Pipeline.from_config), [`tokenize`](../execution/pipeline.md#Pipeline.tokenize), [`merge_tokenizer_config`](config.md#merge_tokenizer_config), [`tokenize`](config.md#PipelineConfig.tokenize), [`__init__`](../transforms/tokenize.md#TokenizeStage.__init__), [`_stage_config`](../transforms/tokenize.md#TokenizeStage._stage_config)

### `TokenizerConfig`
- def: [`easydel/data/core/config.py:33`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L33)
- doc: Configuration for a tokenizer.
- signature: `class TokenizerConfig:`
- members:
  - `to_dict(self)` — [`L54`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L54) — Convert tokenizer configuration to a dictionary of keyword arguments.
  - `add_special_tokens` — [`L50`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L50)
  - `max_length` — [`L47`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L47)
  - `name_or_path` — [`L46`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L46)
  - `padding` — [`L49`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L49)
  - `return_attention_mask` — [`L51`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L51)
  - `truncation` — [`L48`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L48)
  - `trust_remote_code` — [`L52`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L52)
- used by: [`process`](../transforms/tokenize.md#TokenizeStage.process), [`tokenize_dataset_config`](../transforms/tokenize.md#tokenize_dataset_config), [`merge_tokenizer_config`](config.md#merge_tokenizer_config), [`batched_tokenize_iterator`](../transforms/tokenize.md#batched_tokenize_iterator), [`tokenize_batch`](../transforms/tokenize.md#TokenizerManager.tokenize_batch), [`tokenize_text`](../transforms/tokenize.md#TokenizerManager.tokenize_text), [`get_tokenizer`](../transforms/tokenize.md#TokenizerManager.get_tokenizer), [`__repr__`](../transforms/tokenize.md#TokenizedShardedSource.__repr__), [`get_tokenizer_config`](config.md#DatasetConfig.get_tokenizer_config), [`_make_cache_key`](../transforms/tokenize.md#TokenizerManager._make_cache_key), [`tokenizer`](config.md#DatasetConfig.tokenizer), [`__init__`](../transforms/tokenize.md#TokenizedShardedSource.__init__)

### `WeightSchedulePoint`
- def: [`easydel/data/core/config.py:215`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L215)
- doc: A single checkpoint in a dynamic weight schedule for dataset mixing.
- signature: `class WeightSchedulePoint:`
- members:
  - `__post_init__(self)` — [`L229`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L229) — Validate weights sum to 1.
  - `step` — [`L226`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L226)
  - `weights` — [`L227`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L227)
- used by: [`validate`](config.md#PipelineConfig.validate), [`_schedule`](../transforms/mixture.md#WeightScheduler._schedule), [`get_weights`](../transforms/mixture.md#WeightScheduler.get_weights), [`_dataset_names`](../transforms/mixture.md#WeightScheduler._dataset_names), [`weight_schedule`](config.md#MixStageConfig.weight_schedule), [`__init__`](../transforms/mixture.md#WeightScheduler.__init__)

## Functions
- `get_dataset_name(ds_cfg: DatasetConfig, index: int)` — [`L493`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L493) — Get a unique name for a dataset configuration.
- `merge_tokenizer_config(ds_cfg: DatasetConfig, global_tokenizer: str | None, stage_cfg: TokenizeStageConfig)` — [`L506`](../../../../../../../raw/code/EasyDeL/easydel/data/core/config.py#L506) — Merge tokenizer configuration from multiple sources.

