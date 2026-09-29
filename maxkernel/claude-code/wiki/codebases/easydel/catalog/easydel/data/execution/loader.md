---
title: 'Module: easydel/data/execution/loader.py'
type: catalog
provenance: extracted
module: easydel/data/execution/loader.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.execution.loader`/
symbols:
  AsyncDataLoader._create_iterator: AsyncDataLoader#_create_iterator().
  LoadStage.process: LoadStage#process().
  PrefetchIterator._prefetch_worker: PrefetchIterator#_prefetch_worker().
  AsyncDataLoader: AsyncDataLoader#
  LoadStage._stage_config: LoadStage#_stage_config.
  create_data_iterator: create_data_iterator().
  PrefetchIterator._start: PrefetchIterator#_start().
  AsyncDataLoader.iter_examples: AsyncDataLoader#iter_examples().
  PrefetchIterator.__next__: PrefetchIterator#__next__().
  AsyncDataLoader.__aiter__: AsyncDataLoader#__aiter__().
  LoadStage: LoadStage#
  batch_iterator: batch_iterator().
  PrefetchIterator.close: PrefetchIterator#close().
  ShardingSpec.apply: ShardingSpec#apply().
  LoadStage.__init__: LoadStage#__init__().
  collate_batch: collate_batch().
  PrefetchIterator._buffer: PrefetchIterator#_buffer.
  PrefetchIterator._worker: PrefetchIterator#_worker.
  AsyncDataLoader._shuffle_stream: AsyncDataLoader#_shuffle_stream().
  PrefetchIterator: PrefetchIterator#
  preshard_batch: preshard_batch().
  AsyncDataLoader._sharding_map: AsyncDataLoader#_sharding_map.
  AsyncDataLoader.__init__: AsyncDataLoader#__init__().
  AsyncDataLoader.get_output_sharding: AsyncDataLoader#get_output_sharding().
  AsyncDataLoader.is_exhausted: AsyncDataLoader#is_exhausted().
  AsyncDataLoader.__iter__: AsyncDataLoader#__iter__().
  PrefetchIterator._stop_event: PrefetchIterator#_stop_event.
  PrefetchIterator._sentinel: PrefetchIterator#_sentinel.
  PrefetchIterator._started: PrefetchIterator#_started.
  ShardingSpec: ShardingSpec#
  ShardingSpec.mesh: ShardingSpec#mesh.
  AsyncDataLoader._source: AsyncDataLoader#_source.
  AsyncDataLoader._shuffle_buffer_size: AsyncDataLoader#_shuffle_buffer_size.
  AsyncDataLoader._seed: AsyncDataLoader#_seed.
  AsyncDataLoader._exhausted: AsyncDataLoader#_exhausted.
  logger: logger.
  PrefetchIterator._source: PrefetchIterator#_source.
  ShardingSpec.partition_specs: ShardingSpec#partition_specs.
  AsyncDataLoader._batch_size: AsyncDataLoader#_batch_size.
  AsyncDataLoader._prefetch_enabled: AsyncDataLoader#_prefetch_enabled.
  AsyncDataLoader._prefetch_workers: AsyncDataLoader#_prefetch_workers.
  AsyncDataLoader._prefetch_buffer_size: AsyncDataLoader#_prefetch_buffer_size.
  AsyncDataLoader._drop_last: AsyncDataLoader#_drop_last.
  AsyncDataLoader.get_next: AsyncDataLoader#get_next().
  create_data_iterator._batched: create_data_iterator()._batched().
  create_data_iterator.prefetch_worker: create_data_iterator().prefetch_worker().
  create_data_iterator._gen: create_data_iterator()._gen().
  PrefetchIterator.__init__: PrefetchIterator#__init__().
  PrefetchIterator.__iter__: PrefetchIterator#__iter__().
  AsyncDataLoader.aget: AsyncDataLoader#aget().
  LoadStage.name: LoadStage#name().
---
# Module: [`easydel/data/execution/loader.py`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py)

## Classes
### `AsyncDataLoader`  ·  implements/extends AsyncDataset
- def: [`easydel/data/execution/loader.py:238`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L238)
- doc: Async-first data loader with prefetching and optional sharding.
- signature: `class AsyncDataLoader(AsyncDataset[dict]):`
- members:
  - `__aiter__(self)` — [`L287`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L287) — Async iteration over batches.
  - `__init__(self, source: ShardedDataSource, batch_size: int = 8, prefetch_enabled: bool = True, prefetch_workers: int = 2, prefetch_buffer_size: int = 4, shuffle_buffer_size: int | None = None, drop_last: bool = True, sharding_map: Mapping[str, NamedSharding] | None = None, seed: int | None = None)` — [`L247`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L247) — Initialize AsyncDataLoader.
  - `__iter__(self)` — [`L374`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L374) — Synchronous iteration interface.
  - `_create_iterator(self)` — [`L312`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L312) — Create the base iterator with batching and prefetching.
  - `_shuffle_stream(self, stream: Iterator[dict], buffer_size: int)` — [`L342`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L342) — Shuffle a stream using reservoir sampling.
  - `aget(self, _index: int)` — [`L283`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L283) — Async get is not supported for streaming sources.
  - `get_next()` — [`L296`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L296)
  - `get_output_sharding(self)` — [`L365`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L365) — Return the sharding specification for output batches.
  - `is_exhausted(self)` — [`L370`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L370) — Check if the dataset iteration is complete.
  - `iter_examples()` — [`L316`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L316)
- protocol/private: `_batch_size`[`L273`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L273), `_drop_last`[`L278`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L278), `_exhausted`[`L281`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L281), `_prefetch_buffer_size`[`L276`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L276), `_prefetch_enabled`[`L274`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L274), `_prefetch_workers`[`L275`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L275), `_seed`[`L280`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L280), `_sharding_map`[`L279`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L279), `_shuffle_buffer_size`[`L277`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L277), `_source`[`L272`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L272)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`AsyncDataset`](../core/protocols.md#AsyncDataset), [`batch_iterator`](loader.md#batch_iterator), [`PrefetchIterator`](loader.md#PrefetchIterator), [`preshard_batch`](loader.md#preshard_batch)
- used by: [`process`](loader.md#LoadStage.process), [`build`](pipeline.md#Pipeline.build), [`AsyncDataset`](../core/protocols.md#AsyncDataset), [`aget`](../core/protocols.md#AsyncDataset.aget), [`PipelineDataValue`](pipeline.md#PipelineDataValue), [`get_output_sharding`](../core/protocols.md#AsyncDataset.get_output_sharding), [`is_exhausted`](../core/protocols.md#AsyncDataset.is_exhausted)

### `LoadStage`  ·  implements/extends BaseStage
- def: [`easydel/data/execution/loader.py:379`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L379)
- doc: Pipeline stage for creating data loaders from sources.
- signature: `class LoadStage(BaseStage):`
- members:
  - `__init__(self, config: LoadStageConfig | None = None)` — [`L386`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L386) — Initialize LoadStage.
  - `name(self)` — [`L396`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L396)
  - `process(self, data: dict[str, ShardedDataSource], context: PipelineContext)` — [`L399`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L399) — Create data loaders from sources.
- protocol/private: `_stage_config`[`L393`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L393)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`BaseStage`](../core/protocols.md#BaseStage), [`PipelineContext`](../core/protocols.md#PipelineContext), [`LoadStageConfig`](../core/config.md#LoadStageConfig), [`AsyncDataLoader`](loader.md#AsyncDataLoader), [`__init__`](../core/protocols.md#BaseStage.__init__), [`seed`](../core/protocols.md#PipelineContext.seed), [`batch_size`](../core/config.md#LoadStageConfig.batch_size), [`drop_last`](../core/config.md#LoadStageConfig.drop_last), [`logger`](loader.md#logger), [`prefetch_buffer_size`](../core/config.md#LoadStageConfig.prefetch_buffer_size), [`prefetch_enabled`](../core/config.md#LoadStageConfig.prefetch_enabled), [`prefetch_workers`](../core/config.md#LoadStageConfig.prefetch_workers), [`shuffle_buffer_size`](../core/config.md#LoadStageConfig.shuffle_buffer_size)
- used by: [`load`](pipeline.md#Pipeline.load), [`BaseStage`](../core/protocols.md#BaseStage), [`process`](../core/protocols.md#BaseStage.process), [`name`](../core/protocols.md#BaseStage.name)

### `PrefetchIterator`
- def: [`easydel/data/execution/loader.py:99`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L99)
- doc: Iterator with thread-based prefetching.
- signature: `class PrefetchIterator:`
- members:
  - `__init__(self, source: Iterator, buffer_size: int = 4, num_workers: int = 2)` — [`L106`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L106) — Initialize PrefetchIterator.
  - `_prefetch_worker(self)` — [`L127`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L127) — Background worker that prefetches items.
  - `_start(self)` — [`L139`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L139) — Start the prefetch worker.
  - `close(self)` — [`L166`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L166) — Stop the prefetch worker thread and clean up resources.
- protocol/private: `__iter__`[`L149`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L149), `__next__`[`L152`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L152), `_buffer`[`L121`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L121), `_sentinel`[`L123`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L123), `_source`[`L120`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L120), `_started`[`L125`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L125), `_stop_event`[`L122`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L122), `_worker`[`L124`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L124)
- used by: [`_create_iterator`](loader.md#AsyncDataLoader._create_iterator)

### `ShardingSpec`
- def: [`easydel/data/execution/loader.py:174`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L174)
- doc: Specification for sharding batch arrays across JAX devices.
- signature: `class ShardingSpec:`
- members:
  - `apply(self, batch: dict[str, np.ndarray])` — [`L185`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L185) — Apply sharding to a batch.
  - `mesh` — [`L182`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L182)
  - `partition_specs` — [`L183`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L183)

## Functions
- `_batched(it, bs)` — [`L469`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L469)
- `_gen()` — [`L506`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L506)
- `batch_iterator(source: Iterator[dict], batch_size: int, drop_last: bool = True)` — [`L73`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L73) — Create batches from an iterator.
- `collate_batch(examples: list[dict])` — [`L47`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L47) — Collate a list of examples into a batch.
- `create_data_iterator(dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False, prefetch: bool = True, prefetch_workers: int = 2, prefetch_buffer: int = 4, shuffle_buffer: int | None = None, seed: int | None = None)` — [`L432`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L432) — Create an efficient data iterator with optional batching and prefetching.
- `prefetch_worker(source_iter, buf, stop_evt)` — [`L488`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L488)
- `preshard_batch(batch: dict[str, np.ndarray], sharding_map: Mapping[str, NamedSharding] | None)` — [`L209`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L209) — Pre-shard a batch according to sharding specifications.

## Module values
- `logger` — [`L44`](../../../../../../../raw/code/EasyDeL/easydel/data/execution/loader.py#L44)

