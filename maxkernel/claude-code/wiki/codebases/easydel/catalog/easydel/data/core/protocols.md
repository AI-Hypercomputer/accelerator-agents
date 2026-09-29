---
title: 'Module: easydel/data/core/protocols.py'
type: catalog
provenance: extracted
module: easydel/data/core/protocols.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.core.protocols`/
symbols:
  ShardedDataSource: ShardedDataSource#
  ShardedDataSource.open_shard: ShardedDataSource#open_shard().
  ShardedDataSource.shard_names: ShardedDataSource#shard_names().
  ShardedDataSource.num_shards: ShardedDataSource#num_shards().
  ShardedDataSource.open_shard_at_row: ShardedDataSource#open_shard_at_row().
  BaseStage: BaseStage#
  PipelineContext: PipelineContext#
  ShardedDataSource.get_shard_info: ShardedDataSource#get_shard_info().
  BaseStage.process: BaseStage#process().
  ShardInfo: ShardInfo#
  ShardedDataSource.iter_shards: ShardedDataSource#iter_shards().
  BaseStage.name: BaseStage#name().
  ResumeState.to_dict: ResumeState#to_dict().
  T: T.
  ShardInfo.num_rows: ShardInfo#num_rows.
  T_co: T_co.
  MappedShardedDataSource.open_shard: MappedShardedDataSource#open_shard().
  AsyncDataset: AsyncDataset#
  ShardedDataSource.transform: ShardedDataSource#transform().
  PipelineContext.config: PipelineContext#config.
  MappedShardedDataSource: MappedShardedDataSource#
  ShardedDataSource.map: ShardedDataSource#map().
  PipelineContext.cache_manager: PipelineContext#cache_manager().
  AsyncDataset.aget: AsyncDataset#aget().
  BaseStage._update_metric: BaseStage#_update_metric().
  ShardedDataSource.filter: ShardedDataSource#filter().
  MappedShardedDataSource.__init__: MappedShardedDataSource#__init__().
  MappedShardedDataSource.shard_names: MappedShardedDataSource#shard_names().
  MappedShardedDataSource.num_shards: MappedShardedDataSource#num_shards().
  AsyncDataset.abatch: AsyncDataset#abatch().
  AsyncDataset.get: AsyncDataset#get().
  AsyncDataset.__iter__: AsyncDataset#__iter__().
  BaseStage.__init__: BaseStage#__init__().
  BaseStage._get_dataset_config: BaseStage#_get_dataset_config().
  ShardInfo.shard_id: ShardInfo#shard_id.
  ShardInfo.shard_name: ShardInfo#shard_name.
  MappedShardedDataSource._source: MappedShardedDataSource#_source.
  PipelineContext.seed: PipelineContext#seed.
  PipelineContext.metrics: PipelineContext#metrics.
  AsyncDatasetProtocol: AsyncDatasetProtocol#
  PipelineContext._tokenizers: PipelineContext#_tokenizers.
  PipelineContext._cache_manager: PipelineContext#_cache_manager.
  ResumeState: ResumeState#
  ShardedDataSource.rename_fields: ShardedDataSource#rename_fields().
  ShardedDataSource.select_fields: ShardedDataSource#select_fields().
  ShardedDataSource.drop_fields: ShardedDataSource#drop_fields().
  ShardedDataSource.apply_chat_template: ShardedDataSource#apply_chat_template().
  MappedShardedDataSource.__len__: MappedShardedDataSource#__len__().
  AsyncDatasetProtocol.aget: AsyncDatasetProtocol#aget().
  AsyncDatasetProtocol.__aiter__: AsyncDatasetProtocol#__aiter__().
  AsyncDataset.__aiter__: AsyncDataset#__aiter__().
  AsyncDataset.get_output_sharding: AsyncDataset#get_output_sharding().
  AsyncDataset.is_exhausted: AsyncDataset#is_exhausted().
  PipelineStage.process: PipelineStage#process().
  BaseStage.get_metrics: BaseStage#get_metrics().
  PipelineContext.get_tokenizer: PipelineContext#get_tokenizer().
  PipelineContext.update_step: PipelineContext#update_step().
  PipelineContext.update_epoch: PipelineContext#update_epoch().
  PipelineContext.record_metric: PipelineContext#record_metric().
  PipelineContext.get_metrics: PipelineContext#get_metrics().
  ResumeState.from_dict: ResumeState#from_dict().
  BaseStage._metrics: BaseStage#_metrics.
  ShardInfo.byte_size: ShardInfo#byte_size.
  ShardInfo.url: ShardInfo#url.
  ShardInfo.checksum: ShardInfo#checksum.
  MappedShardedDataSource._fn: MappedShardedDataSource#_fn.
  AsyncDataset._collect: AsyncDataset#_collect().
  PipelineStage: PipelineStage#
  PipelineContext.step: PipelineContext#step.
  PipelineContext.epoch: PipelineContext#epoch.
  ResumeState.shard_index: ResumeState#shard_index.
  ResumeState.row_index: ResumeState#row_index.
  ResumeState.epoch: ResumeState#epoch.
  ResumeState.step: ResumeState#step.
  ResumeState.dataset_states: ResumeState#dataset_states.
  DatasetLike: DatasetLike.
  ShardedDataSource.__len__: ShardedDataSource#__len__().
  AsyncDatasetProtocol.get_output_sharding: AsyncDatasetProtocol#get_output_sharding().
  AsyncDatasetProtocol.is_exhausted: AsyncDatasetProtocol#is_exhausted().
  PipelineStage.name: PipelineStage#name().
  PipelineStage.validate_config: PipelineStage#validate_config().
  BaseStage._config: BaseStage#_config.
  BaseStage.validate_config: BaseStage#validate_config().
---
# Module: [`easydel/data/core/protocols.py`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py)

## Classes
### `AsyncDataset`  ·  implements/extends ABC, Generic
- def: [`easydel/data/core/protocols.py:363`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L363)
- doc: Async-first dataset with dual sync/async interface.
- signature: `class AsyncDataset(ABC, Generic[T]):`
- members:
  - `__aiter__(self)` — [`L378`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L378) — Async iteration over the dataset.
  - `__iter__(self)` — [`L410`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L410) — Sync iteration (wraps async).
  - `abatch(self, indices: Sequence[int])` — [`L382`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L382) — Get multiple items concurrently.
  - `aget(self, index: int)` — [`L373`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L373) — Asynchronously get item at index.
  - `get(self, index: int)` — [`L398`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L398) — Synchronously get item at index.
  - `get_output_sharding(self)` — [`L388`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L388) — Return sharding specification for output batches.
  - `is_exhausted(self)` — [`L393`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L393) — Check if dataset iteration is complete.
- protocol/private: `_collect`[`L414`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L414)
- uses (calls/refs, reference-scoped): [`T`](protocols.md#T), [`AsyncDataLoader`](../execution/loader.md#AsyncDataLoader), [`get_output_sharding`](../execution/loader.md#AsyncDataLoader.get_output_sharding), [`is_exhausted`](../execution/loader.md#AsyncDataLoader.is_exhausted), [`aget`](../execution/loader.md#AsyncDataLoader.aget)
- used by: [`AsyncDataLoader`](../execution/loader.md#AsyncDataLoader)

### `AsyncDatasetProtocol`  ·  implements/extends Protocol
- def: [`easydel/data/core/protocols.py:339`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L339)
- doc: Protocol for async datasets that integrate with JAX's execution model.
- signature: `class AsyncDatasetProtocol(Protocol[T_co]):`
- members:
  - `__aiter__(self)` — [`L346`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L346) — Async iteration over the dataset.
  - `aget(self, index: int)` — [`L342`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L342) — Asynchronously get item at index.
  - `get_output_sharding(self)` — [`L350`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L350) — Return the sharding specification for output batches.
  - `is_exhausted(self)` — [`L358`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L358) — Check if dataset iteration is complete.
- uses (calls/refs, reference-scoped): [`T_co`](protocols.md#T_co)

### `BaseStage`  ·  implements/extends ABC
- def: [`easydel/data/core/protocols.py:459`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L459)
- doc: Base class for pipeline stages with common functionality.
- signature: `class BaseStage(ABC):`
- members:
  - `__init__(self, config: dict | None = None)` — [`L467`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L467) — Initialize BaseStage.
  - `_get_dataset_config(self, dataset_id: str, context: PipelineContext)` — [`L514`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L514) — Get configuration for a specific dataset.
  - `_update_metric(self, key: str, value: Any)` — [`L510`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L510) — Update a metric value.
  - `get_metrics(self)` — [`L502`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L502) — Return a copy of all recorded stage metrics.
  - `name(self)` — [`L478`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L478) — Stage name.
  - `process(self, data: Any, context: PipelineContext)` — [`L483`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L483) — Process data through this stage.
  - `validate_config(self, config: dict)` — [`L491`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L491) — Validate stage configuration. Override for custom validation.
- protocol/private: `_config`[`L473`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L473), `_metrics`[`L474`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L474)
- uses (calls/refs, reference-scoped): [`process`](../execution/save.md#SaveStage.process), [`process`](../transforms/tokenize.md#TokenizeStage.process), [`process`](../transforms/pack.md#PackStage.process), [`process`](../transforms/mixture.md#MixStage.process), [`process`](../execution/loader.md#LoadStage.process), [`PipelineContext`](protocols.md#PipelineContext), [`config`](protocols.md#PipelineContext.config), [`LoadStage`](../execution/loader.md#LoadStage), [`MixStage`](../transforms/mixture.md#MixStage), [`PackStage`](../transforms/pack.md#PackStage), [`SaveStage`](../execution/save.md#SaveStage), [`TokenizeStage`](../transforms/tokenize.md#TokenizeStage), [`name`](../execution/loader.md#LoadStage.name), [`name`](../execution/save.md#SaveStage.name), [`name`](../transforms/mixture.md#MixStage.name), [`name`](../transforms/pack.md#PackStage.name), [`name`](../transforms/tokenize.md#TokenizeStage.name)
- used by: [`process`](../execution/save.md#SaveStage.process), [`process`](../transforms/tokenize.md#TokenizeStage.process), [`LoadStage`](../execution/loader.md#LoadStage), [`MixStage`](../transforms/mixture.md#MixStage), [`PackStage`](../transforms/pack.md#PackStage), [`SaveStage`](../execution/save.md#SaveStage), [`TokenizeStage`](../transforms/tokenize.md#TokenizeStage), [`__init__`](../execution/loader.md#LoadStage.__init__), [`__init__`](../execution/save.md#SaveStage.__init__), [`__init__`](../transforms/mixture.md#MixStage.__init__), [`__init__`](../transforms/pack.md#PackStage.__init__), [`__init__`](../transforms/tokenize.md#TokenizeStage.__init__)

### `MappedShardedDataSource`
- def: [`easydel/data/core/protocols.py:300`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L300)
- members:
  - `__init__(self, source: ShardedDataSource, fn: tp.Callable[[Any], T])` — [`L308`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L308) — Initialize MappedShardedDataSource.
  - `__len__(self)` — [`L333`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L333) — Return length of underlying source.
  - `num_shards(self)` — [`L326`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L326)
  - `open_shard(self, shard_name: str)` — [`L329`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L329)
  - `shard_names(self)` — [`L323`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L323)
- protocol/private: `_fn`[`L320`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L320), `_source`[`L319`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L319)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](protocols.md#ShardedDataSource), [`open_shard`](protocols.md#ShardedDataSource.open_shard), [`shard_names`](protocols.md#ShardedDataSource.shard_names), [`num_shards`](protocols.md#ShardedDataSource.num_shards), [`T`](protocols.md#T)
- used by: [`open_shard`](protocols.md#ShardedDataSource.open_shard), [`shard_names`](protocols.md#ShardedDataSource.shard_names), [`num_shards`](protocols.md#ShardedDataSource.num_shards), [`map`](protocols.md#ShardedDataSource.map)

### `PipelineContext`
- def: [`easydel/data/core/protocols.py:529`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L529)
- members:
  - `cache_manager(self)` — [`L604`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L604) — Get cache manager (lazy initialization).
  - `get_metrics(self)` — [`L595`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L595) — Get all recorded metrics across all stages.
  - `get_tokenizer(self, name_or_path: str)` — [`L549`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L549) — Get or create a tokenizer, cached for reuse across stages.
  - `record_metric(self, stage: str, key: str, value: Any)` — [`L583`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L583) — Record a metric for a specific pipeline stage.
  - `update_epoch(self, epoch: int)` — [`L575`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L575) — Update the current epoch counter.
  - `update_step(self, step: int)` — [`L567`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L567) — Update the current pipeline step counter.
  - `config` — [`L539`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L539)
  - `epoch` — [`L542`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L542)
  - `metrics` — [`L543`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L543)
  - `seed` — [`L540`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L540)
  - `step` — [`L541`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L541)
- protocol/private: `_cache_manager`[`L547`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L547), `_tokenizers`[`L546`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L546)
- uses (calls/refs, reference-scoped): [`PipelineConfig`](config.md#PipelineConfig), [`TreeCacheManager`](../execution/cache.md#TreeCacheManager)
- used by: [`process`](../execution/save.md#SaveStage.process), [`process`](../transforms/tokenize.md#TokenizeStage.process), [`process`](../transforms/pack.md#PackStage.process), [`process`](../transforms/mixture.md#MixStage.process), [`process`](../execution/loader.md#LoadStage.process), [`_context`](../execution/pipeline.md#Pipeline._context), [`process`](protocols.md#BaseStage.process), [`_get_dataset_config`](protocols.md#BaseStage._get_dataset_config), [`get_context`](../execution/pipeline.md#Pipeline.get_context), [`process`](protocols.md#PipelineStage.process)

### `PipelineStage`  ·  implements/extends Protocol
- def: [`easydel/data/core/protocols.py:430`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L430)
- doc: Protocol for all pipeline stages.
- signature: `class PipelineStage(Protocol):`
- members:
  - `name(self)` — [`L434`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L434) — Stage name for logging and metrics.
  - `process(self, data: Any, context: PipelineContext)` — [`L438`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L438) — Process data through this stage.
  - `validate_config(self, config: dict)` — [`L454`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L454) — Validate stage configuration.
- uses (calls/refs, reference-scoped): [`PipelineContext`](protocols.md#PipelineContext)

### `ResumeState`
- def: [`easydel/data/core/protocols.py:616`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L616)
- doc: State for resuming pipeline iteration.
- signature: `class ResumeState:`
- members:
  - `from_dict(cls, data: dict)` — [`L642`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L642) — Deserialize from dictionary.
  - `to_dict(self)` — [`L631`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L631) — Serialize to dictionary.
  - `dataset_states` — [`L629`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L629)
  - `epoch` — [`L627`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L627)
  - `row_index` — [`L626`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L626)
  - `shard_index` — [`L625`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L625)
  - `step` — [`L628`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L628)

### `ShardInfo`
- def: [`easydel/data/core/protocols.py:50`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L50)
- doc: Metadata about a data shard.
- signature: `class ShardInfo:`
- members:
  - `byte_size` — [`L65`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L65)
  - `checksum` — [`L67`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L67)
  - `num_rows` — [`L64`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L64)
  - `shard_id` — [`L62`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L62)
  - `shard_name` — [`L63`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L63)
  - `url` — [`L66`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L66)
- uses (calls/refs, reference-scoped): [`ParquetShardInfo`](../sources/base.md#ParquetShardInfo)
- used by: [`get_shard_info`](../transforms/source.md#LimitedShardedSource.get_shard_info), [`get_shard_info`](../sources/base.md#ParquetShardedSource.get_shard_info), [`_sample_prompts_from_sharded_source`](../../trainers/base_trainer.md#BaseTrainer._sample_prompts_from_sharded_source), [`get_shard_info`](protocols.md#ShardedDataSource.get_shard_info), [`_get_known_shard_size`](../transforms/source.md#LimitedShardedSource._get_known_shard_size), [`get_shard_info`](../sources/hf_wrapper.md#HFDatasetShardedSource.get_shard_info), [`_sum_known_shard_rows`](../../trainers/base_trainer.md#BaseTrainer._sum_known_shard_rows), [`__len__`](../sources/base.md#ParquetShardedSource.__len__), [`ParquetShardInfo`](../sources/base.md#ParquetShardInfo)

### `ShardedDataSource`  ·  implements/extends ABC, Generic
- def: [`easydel/data/core/protocols.py:70`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L70)
- doc: Abstract base class for shard-based data sources.
- signature: `class ShardedDataSource(ABC, Generic[T_co]):`
- members:
  - `__len__(self)` — [`L181`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L181) — Return total number of examples across all shards.
  - `apply_chat_template(self, tokenizer: tp.Any, messages_field: str = "messages", output_field: str = "text", **kwargs)` — [`L268`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L268) — Apply chat template to convert messages to formatted text.
  - `drop_fields(self, fields: list[str])` — [`L247`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L247) — Drop specified fields.
  - `filter(self, predicate: tp.Callable[[T_co], bool])` — [`L158`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L158) — Filter examples by predicate (lazy).
  - `get_shard_info(self, shard_name: str)` — [`L101`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L101) — Get metadata about a specific shard. Optional.
  - `iter_shards(self, shard_indices: Sequence[int] | None = None, start_shard: int = 0, start_row: int = 0)` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L120) — Iterate over shards with optional resume support.
  - `map(self, fn: tp.Callable[[T_co], T])` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L151) — Apply a function to each example (lazy).
  - `num_shards(self)` — [`L92`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L92) — Return total number of shards.
  - `open_shard(self, shard_name: str)` — [`L97`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L97) — Open a shard and return an iterator over its examples.
  - `open_shard_at_row(self, shard_name: str, row: int)` — [`L105`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L105) — Open a shard and skip to a specific row for resumption.
  - `rename_fields(self, mapping: dict[str, str])` — [`L205`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L205) — Rename fields in examples.
  - `select_fields(self, fields: list[str])` — [`L226`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L226) — Select only specified fields.
  - `shard_names(self)` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L87) — Return list of shard identifiers (URLs or paths).
  - `transform(self, transform: tp.Any)` — [`L189`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L189) — Apply a transform or chain of transforms.
- uses (calls/refs, reference-scoped): [`open_shard`](../transforms/pack.md#PackedShardedSource.open_shard), [`get_shard_info`](../transforms/source.md#LimitedShardedSource.get_shard_info), [`get_shard_info`](../sources/base.md#ParquetShardedSource.get_shard_info), [`open_shard`](../transforms/mixture.md#MixedShardedSource.open_shard), [`ShardInfo`](protocols.md#ShardInfo), [`get_shard_info`](../sources/hf_wrapper.md#HFDatasetShardedSource.get_shard_info), [`T`](protocols.md#T), [`PackedShardedSource`](../transforms/pack.md#PackedShardedSource), [`T_co`](protocols.md#T_co), [`open_shard`](protocols.md#MappedShardedDataSource.open_shard), [`open_shard`](../sources/base.md#CompositeShardedSource.open_shard), [`open_shard`](../transforms/source.md#TransformedShardedSource.open_shard), [`open_shard_at_row`](../sources/base.md#CompositeShardedSource.open_shard_at_row), [`open_shard_at_row`](../transforms/source.md#TransformedShardedSource.open_shard_at_row), [`HuggingFaceShardedSource`](../sources/base.md#HuggingFaceShardedSource), [`LimitedShardedSource`](../transforms/source.md#LimitedShardedSource), [`open_shard`](../sources/base.md#CsvShardedSource.open_shard), [`open_shard`](../sources/base.md#JsonShardedSource.open_shard), [`MappedShardedDataSource`](protocols.md#MappedShardedDataSource), [`ArrowShardedSource`](../sources/base.md#ArrowShardedSource), [`CsvShardedSource`](../sources/base.md#CsvShardedSource), [`JsonShardedSource`](../sources/base.md#JsonShardedSource), [`MixedShardedSource`](../transforms/mixture.md#MixedShardedSource), [`ParquetShardedSource`](../sources/base.md#ParquetShardedSource), [`TextShardedSource`](../sources/base.md#TextShardedSource), [`open_shard`](../sources/base.md#ParquetShardedSource.open_shard), [`open_shard`](../sources/hf_wrapper.md#HFDatasetShardedSource.open_shard), [`open_shard`](../transforms/source.md#LimitedShardedSource.open_shard), [`open_shard`](../transforms/tokenize.md#TokenizedShardedSource.open_shard), [`open_shard_at_row`](../sources/base.md#HuggingFaceShardedSource.open_shard_at_row), [`open_shard_at_row`](../sources/hf_wrapper.md#HFDatasetShardedSource.open_shard_at_row), [`open_shard_at_row`](../transforms/source.md#LimitedShardedSource.open_shard_at_row), [`open_shard_at_row`](../transforms/tokenize.md#TokenizedShardedSource.open_shard_at_row), [`HFDatasetShardedSource`](../sources/hf_wrapper.md#HFDatasetShardedSource), [`TokenizedShardedSource`](../transforms/tokenize.md#TokenizedShardedSource), [`TransformedShardedSource`](../transforms/source.md#TransformedShardedSource), [`open_shard`](../sources/base.md#ArrowShardedSource.open_shard), [`open_shard`](../sources/base.md#TextShardedSource.open_shard), [`get_shard_info`](../transforms/source.md#TransformedShardedSource.get_shard_info), [`num_shards`](protocols.md#MappedShardedDataSource.num_shards)  (+35 more)
- used by: [`_train_source`](../../trainers/base_trainer.md#BaseTrainer._train_source), [`process`](../execution/save.md#SaveStage.process), [`__init__`](../../trainers/base_trainer.md#BaseTrainer.__init__), [`_resolve_step_count`](../../trainers/base_trainer.md#BaseTrainer._resolve_step_count), [`build_trainer`](../../infra/elarge/model.md#eLargeModel.build_trainer), [`process`](../transforms/tokenize.md#TokenizeStage.process), [`_is_pretokenized`](../../trainers/base_trainer.md#BaseTrainer._is_pretokenized), [`create_source`](../sources/base.md#create_source), [`__init__`](../../trainers/group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.__init__), [`process`](../transforms/pack.md#PackStage.process), [`__init__`](../../trainers/proximal_policy_optimization_trainer/ppo_trainer.md#PPOTrainer.__init__), [`open_shard`](../transforms/pack.md#PackedShardedSource.open_shard), [`__init__`](../../trainers/rlvr_trainer/rlvr_trainer.md#RLVRTrainer.__init__), [`get_shard_info`](../transforms/source.md#LimitedShardedSource.get_shard_info), [`process`](../transforms/mixture.md#MixStage.process), [`train`](../../infra/elarge/model.md#eLargeModel.train), [`tokenize_dataset_config`](../transforms/tokenize.md#tokenize_dataset_config), [`process`](../execution/loader.md#LoadStage.process), [`mix`](../execution/pipeline.md#Pipeline.mix), [`build_sharded_source`](../../infra/elarge/builders.md#build_sharded_source), [`save`](../execution/pipeline.md#Pipeline.save), [`tokenize`](../execution/pipeline.md#Pipeline.tokenize), [`__init__`](../../trainers/kto_trainer/kto_trainer.md#KTOTrainer.__init__), [`load`](../execution/pipeline.md#Pipeline.load), [`pack`](../execution/pipeline.md#Pipeline.pack), [`write`](../execution/save.md#ArrowWriter.write), [`write`](../execution/save.md#JsonlWriter.write), [`write`](../execution/save.md#ParquetWriter.write), [`__init__`](../../trainers/supervised_fine_tuning_trainer/sft_trainer.md#SFTTrainer.__init__), [`_train_density_ratio_classifier`](../../trainers/binary_classifier_optimization_trainer/bco_trainer.md#BCOTrainer._train_density_ratio_classifier), [`__init__`](../../trainers/agentic_moshpit/agentic_moshpit_trainer.md#AgenticMoshPitTrainer.__init__), [`__init__`](../../trainers/sparse_distillation_trainer/sparse_distillation_trainer.md#SparseDistillationTrainer.__init__), [`_create_source_from_inform`](../../infra/elarge/builders.md#_create_source_from_inform), [`__init__`](../../trainers/seq_kd_trainer/seq_kd_trainer.md#SeqKDTrainer.__init__), [`pretokenize`](../execution/pipeline.md#pretokenize), [`__init__`](../../trainers/binary_classifier_optimization_trainer/bco_trainer.md#BCOTrainer.__init__), [`__init__`](../../trainers/contrastive_preference_optimization_trainer/cpo_trainer.md#CPOTrainer.__init__), [`__init__`](../../trainers/odds_ratio_preference_optimization_trainer/orpo_trainer.md#ORPOTrainer.__init__), [`__init__`](../../trainers/on_policy_distillation_trainer/on_policy_distillation_trainer.md#OnPolicyDistillationTrainer.__init__), [`__init__`](../../trainers/reward_trainer/reward_trainer.md#RewardTrainer.__init__)  (+83 more)

## Module values
- `DatasetLike` — [`L46`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L46)
- `T` — [`L44`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L44)
- `T_co` — [`L45`](../../../../../../../raw/code/EasyDeL/easydel/data/core/protocols.py#L45)

