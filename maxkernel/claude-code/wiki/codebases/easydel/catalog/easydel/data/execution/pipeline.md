---
title: 'Module: easydel/data/execution/pipeline.py'
type: catalog
provenance: extracted
module: easydel/data/execution/pipeline.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.execution.pipeline`/
symbols:
  build_dataset: build_dataset().
  tokenize_and_save: tokenize_and_save().
  Pipeline.from_config: Pipeline#from_config().
  Pipeline.mix: Pipeline#mix().
  Pipeline.tokenize: Pipeline#tokenize().
  Pipeline.save: Pipeline#save().
  Pipeline.pack: Pipeline#pack().
  Pipeline.load: Pipeline#load().
  pretokenize: pretokenize().
  Pipeline.source: Pipeline#source().
  Pipeline._data: Pipeline#_data.
  Pipeline._context: Pipeline#_context.
  create_pipeline: create_pipeline().
  Pipeline: Pipeline#
  Pipeline.build: Pipeline#build().
  Pipeline._ensure_data: Pipeline#_ensure_data().
  Pipeline._stages: Pipeline#_stages.
  logger: logger.
  PipelineDataValue: PipelineDataValue.
  Pipeline._config: Pipeline#_config.
  Pipeline.get_context: Pipeline#get_context().
  Pipeline.__init__: Pipeline#__init__().
  Pipeline.get_data: Pipeline#get_data().
  Pipeline.get_stages: Pipeline#get_stages().
  build_dataset.rename_fields: build_dataset().rename_fields().
  build_dataset.to_target: build_dataset().to_target().
---
# Module: [`easydel/data/execution/pipeline.py`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py)

## Classes
### `Pipeline`
- def: [`easydel/data/execution/pipeline.py:60`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L60)
- doc: Fluent API for building data processing pipelines.
- signature: `class Pipeline:`
- members:
  - `__init__(self, config: PipelineConfig)` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L82) — Initialize Pipeline.
  - `_ensure_data(self)` — [`L284`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L284) — Ensure data has been loaded by calling source() first.
  - `build(self)` — [`L245`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L245) — Build and return the final data iterator.
  - `from_config(cls, config: PipelineConfig | dict)` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L94) — Create a pipeline from configuration.
  - `get_context(self)` — [`L268`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L268) — Get the pipeline context.
  - `get_data(self)` — [`L260`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L260) — Get the current pipeline data.
  - `get_stages(self)` — [`L276`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L276) — Get the list of applied stages.
  - `load(self, config: LoadStageConfig | None = None)` — [`L227`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L227) — Create data loaders with batching and prefetching.
  - `mix(self, config: MixStageConfig | None = None)` — [`L162`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L162) — Mix multiple datasets into one.
  - `pack(self, config: PackStageConfig | None = None)` — [`L187`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L187) — Pack sequences into fixed-length chunks.
  - `save(self, config: SaveStageConfig | None = None)` — [`L207`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L207) — Save datasets to disk.
  - `source(self)` — [`L121`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L121) — Load datasets from their sources.
  - `tokenize(self, config: TokenizeStageConfig | None = None)` — [`L142`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L142) — Apply tokenization to all datasets.
- protocol/private: `_config`[`L88`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L88), `_context`[`L89`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L89), `_data`[`L90`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L90), `_stages`[`L91`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L91)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`process`](save.md#SaveStage.process), [`process`](../transforms/tokenize.md#TokenizeStage.process), [`create_source`](../sources/base.md#create_source), [`process`](../transforms/pack.md#PackStage.process), [`process`](../transforms/mixture.md#MixStage.process), [`process`](loader.md#LoadStage.process), [`PipelineContext`](../core/protocols.md#PipelineContext), [`DatasetConfig`](../core/config.md#DatasetConfig), [`TokenizeStageConfig`](../core/config.md#TokenizeStageConfig), [`SaveStageConfig`](../core/config.md#SaveStageConfig), [`datasets`](../core/config.md#PipelineConfig.datasets), [`LoadStageConfig`](../core/config.md#LoadStageConfig), [`MixStageConfig`](../core/config.md#MixStageConfig), [`PackStageConfig`](../core/config.md#PackStageConfig), [`PipelineConfig`](../core/config.md#PipelineConfig), [`AsyncDataLoader`](loader.md#AsyncDataLoader), [`config`](../core/protocols.md#PipelineContext.config), [`mix`](../core/config.md#PipelineConfig.mix), [`name`](../core/config.md#DatasetConfig.name), [`LoadStage`](loader.md#LoadStage), [`MixStage`](../transforms/mixture.md#MixStage), [`PackStage`](../transforms/pack.md#PackStage), [`SaveStage`](save.md#SaveStage), [`TokenizeStage`](../transforms/tokenize.md#TokenizeStage), [`PipelineDataValue`](pipeline.md#PipelineDataValue), [`logger`](pipeline.md#logger), [`save`](../core/config.md#PipelineConfig.save), [`tokenize`](../core/config.md#PipelineConfig.tokenize), [`load`](../core/config.md#PipelineConfig.load), [`pack`](../core/config.md#PipelineConfig.pack), [`seed`](../core/protocols.md#PipelineContext.seed), [`cache`](../core/config.md#PipelineConfig.cache), [`source`](../core/config.md#PipelineConfig.source), [`default_tokenizer`](../core/config.md#PipelineConfig.default_tokenizer), [`seed`](../core/config.md#PipelineConfig.seed), [`streaming`](../core/config.md#PipelineConfig.streaming)
- used by: [`tokenize_and_save`](pipeline.md#tokenize_and_save), [`create_pipeline`](pipeline.md#create_pipeline)

## Functions
- `build_dataset(mixture: DatasetMixture)` — [`L450`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L450) — Build a unified dataset from a DatasetMixture configuration.
- `create_pipeline(datasets: list[DatasetConfig | dict], default_tokenizer: str | None = None, **kwargs)` — [`L294`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L294) — Create a pipeline from a list of dataset configurations.
- `pretokenize(source: ShardedDataSource, transform: tp.Any, output_path: str, output_format: str = "parquet", max_shard_size: str = "500MB", compression: str | None = "snappy", num_proc: int | None = None, show_progress: bool = True)` — [`L353`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L353) — Pretokenize a data source using a trainer transform and save to disk.
- `rename_fields(ex, _mapping=mapping_local)` — [`L504`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L504)
- `to_target(ex, _content_field=content_field, _addl=addl_fields, _target=content_target)` — [`L537`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L537)
- `tokenize_and_save(data_files: str | os.PathLike | list[str | os.PathLike], tokenizer: str, output_path: str, output_format: str = "parquet", max_length: int = 2048)` — [`L318`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L318) — Tokenize a dataset and save to disk.

## Module values
- `PipelineDataValue` — [`L57`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L57)
- `logger` — [`L56`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/pipeline.py#L56)

