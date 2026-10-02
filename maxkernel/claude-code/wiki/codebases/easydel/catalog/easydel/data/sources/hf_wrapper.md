---
title: 'Module: easydel/data/sources/hf_wrapper.py'
type: catalog
provenance: extracted
module: easydel/data/sources/hf_wrapper.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.sources.hf_wrapper`/
symbols:
  HFDatasetShardedSource.get_shard_info: HFDatasetShardedSource#get_shard_info().
  HFDatasetShardedSource._length: HFDatasetShardedSource#_length.
  wrap_hf_dataset: wrap_hf_dataset().
  HFDatasetShardedSource._is_iterable: HFDatasetShardedSource#_is_iterable.
  HFDatasetShardedSource.open_shard: HFDatasetShardedSource#open_shard().
  HFDatasetShardedSource.open_shard_at_row: HFDatasetShardedSource#open_shard_at_row().
  HFDatasetShardedSource: HFDatasetShardedSource#
  HFDatasetShardedSource._dataset: HFDatasetShardedSource#_dataset.
  HFDatasetShardedSource.__repr__: HFDatasetShardedSource#__repr__().
  HFDatasetShardedSource._to_example: HFDatasetShardedSource#_to_example().
  HFDatasetShardedSource.shard_names: HFDatasetShardedSource#shard_names().
  HFDatasetShardedSource.is_streaming: HFDatasetShardedSource#is_streaming().
  HFDatasetShardedSource.estimated_length: HFDatasetShardedSource#estimated_length().
  HFDatasetShardedSource.__len__: HFDatasetShardedSource#__len__().
  HFDatasetShardedSource._name: HFDatasetShardedSource#_name.
  HFDatasetShardedSource._check_is_iterable: HFDatasetShardedSource#_check_is_iterable().
  HFDatasetShardedSource.__init__: HFDatasetShardedSource#__init__().
  HFDatasetShardedSource.num_shards: HFDatasetShardedSource#num_shards().
---
# Module: [`easydel/data/sources/hf_wrapper.py`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py)

## Classes
### `HFDatasetShardedSource`  ·  implements/extends ShardedDataSource
- def: [`easydel/data/sources/hf_wrapper.py:33`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L33)
- doc: Wraps HuggingFace Dataset/IterableDataset as ShardedDataSource.
- signature: `class HFDatasetShardedSource(ShardedDataSource[dict]):`
- members:
  - `__init__(self, dataset: Dataset | IterableDataset, name: str | None = None)` — [`L54`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L54) — Initialize HFDatasetShardedSource.
  - `__len__(self)` — [`L181`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L181) — Return number of examples in the dataset.
  - `_check_is_iterable(dataset)` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L78) — Check if the dataset is a streaming IterableDataset.
  - `_to_example(value: tp.Any)` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L96) — Normalize a dataset element to a plain dictionary example.
  - `estimated_length(self)` — [`L177`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L177) — Return estimated number of examples, if known.
  - `get_shard_info(self, shard_name: str)` — [`L156`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L156) — Get metadata about the shard.
  - `is_streaming(self)` — [`L172`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L172) — Check if this is a streaming (IterableDataset) source.
  - `num_shards(self)` — [`L114`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L114) — Return number of shards. HF datasets are treated as single shard.
  - `open_shard(self, shard_name: str)` — [`L118`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L118) — Open the shard and return iterator over examples.
  - `open_shard_at_row(self, shard_name: str, row: int)` — [`L136`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L136) — Open shard starting at a specific row.
  - `shard_names(self)` — [`L110`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L110) — Return shard names. HF datasets are treated as single shard.
- protocol/private: `__repr__`[`L191`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L191), `_dataset`[`L65`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L65), `_is_iterable`[`L67`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L67), `_length`[`L68`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L68), `_name`[`L66`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L66)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`ShardInfo`](../core/protocols.md#ShardInfo), [`num_rows`](../core/protocols.md#ShardInfo.num_rows), [`shard_id`](../core/protocols.md#ShardInfo.shard_id), [`shard_name`](../core/protocols.md#ShardInfo.shard_name)
- used by: [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`num_shards`](../core/protocols.md#ShardedDataSource.num_shards), [`open_shard_at_row`](../core/protocols.md#ShardedDataSource.open_shard_at_row), [`get_shard_info`](../core/protocols.md#ShardedDataSource.get_shard_info), [`wrap_hf_dataset`](hf_wrapper.md#wrap_hf_dataset)

## Functions
- `wrap_hf_dataset(dataset: Dataset | IterableDataset | ShardedDataSource)` — [`L197`](../../../../../../../raw/code/EasyDeL/easydel/data/sources/hf_wrapper.py#L197) — Wrap a HuggingFace dataset as ShardedDataSource if needed.

