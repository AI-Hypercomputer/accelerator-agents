---
title: 'Module: easydel/data/execution/save.py'
type: catalog
provenance: extracted
module: easydel/data/execution/save.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.execution.save`/
symbols:
  SaveStage.process: SaveStage#process().
  ParquetWriter.write: ParquetWriter#write().
  ArrowWriter.write: ArrowWriter#write().
  JsonlWriter.write: JsonlWriter#write().
  SaveStage._push_to_hub: SaveStage#_push_to_hub().
  create_writer: create_writer().
  save_dataset: save_dataset().
  DatasetWriter.write: DatasetWriter#write().
  SaveStage._stage_config: SaveStage#_stage_config.
  WriteStats: WriteStats#
  DatasetWriter: DatasetWriter#
  ParquetWriter.flush_shard: ParquetWriter#flush_shard().
  JsonlWriter.flush_shard: JsonlWriter#flush_shard().
  ArrowWriter.flush_shard: ArrowWriter#flush_shard().
  save_iterator: save_iterator().
  logger: logger.
  WriteStats.num_examples: WriteStats#num_examples.
  WriteStats.num_shards: WriteStats#num_shards.
  WriteStats.output_paths: WriteStats#output_paths.
  SaveStage: SaveStage#
  ParquetWriter: ParquetWriter#
  ArrowWriter: ArrowWriter#
  JsonlWriter: JsonlWriter#
  SaveStage.__init__: SaveStage#__init__().
  WriteStats.total_bytes: WriteStats#total_bytes.
  save_iterator.IteratorSource: save_iterator().IteratorSource#
  DatasetWriter.output_path: DatasetWriter#output_path.
  DatasetWriter.max_shard_size: DatasetWriter#max_shard_size.
  DatasetWriter.num_shards: DatasetWriter#num_shards.
  WriteStats.__post_init__: WriteStats#__post_init__().
  save_iterator.IteratorSource.open_shard: save_iterator().IteratorSource#open_shard().
  parse_size: parse_size().
  DatasetWriter.compression: DatasetWriter#compression.
  IteratorSource._it: IteratorSource#_it.
  DatasetWriter.__init__: DatasetWriter#__init__().
  DatasetWriter.overwrite: DatasetWriter#overwrite.
  SaveStage.name: SaveStage#name().
  save_iterator.IteratorSource.__init__: save_iterator().IteratorSource#__init__().
  save_iterator.IteratorSource.shard_names: save_iterator().IteratorSource#shard_names().
  save_iterator.IteratorSource.num_shards: save_iterator().IteratorSource#num_shards().
---
# Module: [`easydel/data/execution/save.py`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py)

## Classes
### `ArrowWriter`  ·  implements/extends DatasetWriter
- def: [`easydel/data/execution/save.py:220`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L220)
- doc: Writer for Apache Arrow IPC (Feather) format.
- signature: `class ArrowWriter(DatasetWriter):`
- members:
  - `flush_shard()` — [`L247`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L247)
  - `write(self, source: ShardedDataSource)` — [`L226`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L226) — Write all data from source to sharded Arrow IPC files.
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`WriteStats`](save.md#WriteStats), [`DatasetWriter`](save.md#DatasetWriter), [`logger`](save.md#logger), [`num_examples`](save.md#WriteStats.num_examples), [`num_shards`](save.md#WriteStats.num_shards), [`output_paths`](save.md#WriteStats.output_paths), [`total_bytes`](save.md#WriteStats.total_bytes), [`max_shard_size`](save.md#DatasetWriter.max_shard_size), [`num_shards`](save.md#DatasetWriter.num_shards), [`output_path`](save.md#DatasetWriter.output_path)
- used by: [`create_writer`](save.md#create_writer), [`write`](save.md#DatasetWriter.write), [`DatasetWriter`](save.md#DatasetWriter)

### `DatasetWriter`
- def: [`easydel/data/execution/save.py:104`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L104)
- doc: Base class for dataset writers.
- signature: `class DatasetWriter:`
- members:
  - `__init__(self, output_path: str, max_shard_size: int = 500 * 1024 * 1024, num_shards: int | None = None, compression: str | None = None, overwrite: bool = False)` — [`L112`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L112) — Initialize DatasetWriter.
  - `write(self, source: ShardedDataSource)` — [`L135`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L135) — Write a sharded source to output.
  - `compression` — [`L132`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L132)
  - `max_shard_size` — [`L130`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L130)
  - `num_shards` — [`L131`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L131)
  - `output_path` — [`L129`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L129)
  - `overwrite` — [`L133`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L133)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`write`](save.md#ArrowWriter.write), [`write`](save.md#JsonlWriter.write), [`write`](save.md#ParquetWriter.write), [`WriteStats`](save.md#WriteStats), [`ArrowWriter`](save.md#ArrowWriter), [`JsonlWriter`](save.md#JsonlWriter), [`ParquetWriter`](save.md#ParquetWriter)
- used by: [`process`](save.md#SaveStage.process), [`write`](save.md#ArrowWriter.write), [`write`](save.md#JsonlWriter.write), [`write`](save.md#ParquetWriter.write), [`create_writer`](save.md#create_writer), [`save_dataset`](save.md#save_dataset), [`flush_shard`](save.md#JsonlWriter.flush_shard), [`flush_shard`](save.md#ParquetWriter.flush_shard), [`ArrowWriter`](save.md#ArrowWriter), [`JsonlWriter`](save.md#JsonlWriter), [`ParquetWriter`](save.md#ParquetWriter)

### `IteratorSource`  ·  implements/extends ShardedDataSource
- def: [`easydel/data/execution/save.py:576`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L576)
- doc: Adapter wrapping a plain iterator as a single-shard ShardedDataSource.
- signature: `class IteratorSource(ShardedDataSource[dict]):`
- members:
  - `num_shards(self)` — [`L586`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L586)
  - `open_shard(self, shard_name)` — [`L589`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L589)
  - `shard_names(self)` — [`L583`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L583)
- protocol/private: `__init__`[`L579`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L579), `_it`[`L580`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L580)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource)
- used by: [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`num_shards`](../core/protocols.md#ShardedDataSource.num_shards), [`save_iterator`](save.md#save_iterator)

### `JsonlWriter`  ·  implements/extends DatasetWriter
- def: [`easydel/data/execution/save.py:289`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L289)
- doc: Writer for JSON Lines format.
- signature: `class JsonlWriter(DatasetWriter):`
- members:
  - `flush_shard()` — [`L315`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L315)
  - `write(self, source: ShardedDataSource)` — [`L296`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L296) — Write all data from source to sharded JSONL files.
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`WriteStats`](save.md#WriteStats), [`DatasetWriter`](save.md#DatasetWriter), [`logger`](save.md#logger), [`num_examples`](save.md#WriteStats.num_examples), [`num_shards`](save.md#WriteStats.num_shards), [`output_paths`](save.md#WriteStats.output_paths), [`total_bytes`](save.md#WriteStats.total_bytes), [`max_shard_size`](save.md#DatasetWriter.max_shard_size), [`num_shards`](save.md#DatasetWriter.num_shards), [`output_path`](save.md#DatasetWriter.output_path), [`compression`](save.md#DatasetWriter.compression)
- used by: [`create_writer`](save.md#create_writer), [`write`](save.md#DatasetWriter.write), [`DatasetWriter`](save.md#DatasetWriter)

### `ParquetWriter`  ·  implements/extends DatasetWriter
- def: [`easydel/data/execution/save.py:147`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L147)
- doc: Writer for Apache Parquet columnar format.
- signature: `class ParquetWriter(DatasetWriter):`
- members:
  - `flush_shard()` — [`L176`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L176)
  - `write(self, source: ShardedDataSource)` — [`L154`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L154) — Write all data from source to sharded Parquet files.
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`WriteStats`](save.md#WriteStats), [`DatasetWriter`](save.md#DatasetWriter), [`logger`](save.md#logger), [`num_examples`](save.md#WriteStats.num_examples), [`num_shards`](save.md#WriteStats.num_shards), [`output_paths`](save.md#WriteStats.output_paths), [`total_bytes`](save.md#WriteStats.total_bytes), [`max_shard_size`](save.md#DatasetWriter.max_shard_size), [`num_shards`](save.md#DatasetWriter.num_shards), [`output_path`](save.md#DatasetWriter.output_path), [`compression`](save.md#DatasetWriter.compression)
- used by: [`create_writer`](save.md#create_writer), [`write`](save.md#DatasetWriter.write), [`DatasetWriter`](save.md#DatasetWriter)

### `SaveStage`  ·  implements/extends BaseStage
- def: [`easydel/data/execution/save.py:410`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L410)
- doc: Pipeline stage for saving processed datasets to disk.
- signature: `class SaveStage(BaseStage):`
- members:
  - `__init__(self, config: SaveStageConfig | None = None)` — [`L418`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L418) — Initialize SaveStage.
  - `_push_to_hub(self, local_path: str, ds_name: str, stats: WriteStats)` — [`L486`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L486) — Push saved dataset to HuggingFace Hub.
  - `name(self)` — [`L428`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L428)
  - `process(self, data: dict[str, ShardedDataSource], context: PipelineContext)` — [`L431`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L431) — Process datasets through save stage.
- protocol/private: `_stage_config`[`L425`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L425)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`BaseStage`](../core/protocols.md#BaseStage), [`PipelineContext`](../core/protocols.md#PipelineContext), [`create_writer`](save.md#create_writer), [`write`](save.md#DatasetWriter.write), [`WriteStats`](save.md#WriteStats), [`SaveStageConfig`](../core/config.md#SaveStageConfig), [`get_dataset_by_name`](../core/config.md#PipelineConfig.get_dataset_by_name), [`config`](../core/protocols.md#PipelineContext.config), [`logger`](save.md#logger), [`num_examples`](save.md#WriteStats.num_examples), [`num_shards`](save.md#WriteStats.num_shards), [`output_paths`](save.md#WriteStats.output_paths), [`_update_metric`](../core/protocols.md#BaseStage._update_metric), [`__init__`](../core/protocols.md#BaseStage.__init__), [`save_format`](../core/config.md#DatasetConfig.save_format), [`enabled`](../core/config.md#SaveStageConfig.enabled), [`format`](../core/config.md#SaveStageConfig.format), [`hub_repo_id`](../core/config.md#SaveStageConfig.hub_repo_id), [`save_path`](../core/config.md#DatasetConfig.save_path), [`compression`](../core/config.md#SaveStageConfig.compression), [`hub_private`](../core/config.md#SaveStageConfig.hub_private), [`hub_token`](../core/config.md#SaveStageConfig.hub_token), [`max_shard_size`](../core/config.md#SaveStageConfig.max_shard_size), [`num_shards`](../core/config.md#SaveStageConfig.num_shards), [`output_dir`](../core/config.md#SaveStageConfig.output_dir), [`overwrite`](../core/config.md#SaveStageConfig.overwrite), [`push_to_hub`](../core/config.md#SaveStageConfig.push_to_hub)
- used by: [`save`](pipeline.md#Pipeline.save), [`BaseStage`](../core/protocols.md#BaseStage), [`process`](../core/protocols.md#BaseStage.process), [`name`](../core/protocols.md#BaseStage.name)

### `WriteStats`
- def: [`easydel/data/execution/save.py:84`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L84)
- doc: Statistics from a dataset write operation.
- signature: `class WriteStats:`
- members:
  - `num_examples` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L94)
  - `num_shards` — [`L95`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L95)
  - `output_paths` — [`L97`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L97)
  - `total_bytes` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L96)
- protocol/private: `__post_init__`[`L99`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L99)
- used by: [`process`](save.md#SaveStage.process), [`write`](save.md#ArrowWriter.write), [`write`](save.md#JsonlWriter.write), [`write`](save.md#ParquetWriter.write), [`pretokenize`](pipeline.md#pretokenize), [`_push_to_hub`](save.md#SaveStage._push_to_hub), [`save_dataset`](save.md#save_dataset), [`write`](save.md#DatasetWriter.write), [`flush_shard`](save.md#JsonlWriter.flush_shard), [`flush_shard`](save.md#ParquetWriter.flush_shard), [`flush_shard`](save.md#ArrowWriter.flush_shard), [`save_iterator`](save.md#save_iterator)

## Functions
- `create_writer(output_path: str, format: str = "parquet", max_shard_size: str | int = "500MB", num_shards: int | None = None, compression: str | None = None, overwrite: bool = False)` — [`L359`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L359) — Create a dataset writer for the specified format.
- `parse_size(size: str | int)` — [`L42`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L42) — Parse a size string (e.g., '500MB') to bytes.
- `save_dataset(source: ShardedDataSource, output_path: str, format: str = "parquet", max_shard_size: str | int = "500MB", num_shards: int | None = None, compression: str | None = None)` — [`L522`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L522) — Save a sharded source to disk.
- `save_iterator(iterator: Iterator[dict], output_path: str, format: str = "parquet", max_shard_size: str | int = "500MB", compression: str | None = None)` — [`L555`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L555) — Save an iterator of examples to disk.

## Module values
- `logger` — [`L39`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/save.py#L39)

