---
title: 'Module: easydel/data/transforms/source.py'
type: catalog
provenance: extracted
module: easydel/data/transforms/source.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.transforms.source`/
symbols:
  LimitedShardedSource.get_shard_info: LimitedShardedSource#get_shard_info().
  LimitedShardedSource._resolve_prefix_until: LimitedShardedSource#_resolve_prefix_until().
  LimitedShardedSource._get_shard_limit: LimitedShardedSource#_get_shard_limit().
  LimitedShardedSource._get_known_shard_size: LimitedShardedSource#_get_known_shard_size().
  TransformedShardedSource.open_shard: TransformedShardedSource#open_shard().
  TransformedShardedSource.open_shard_at_row: TransformedShardedSource#open_shard_at_row().
  LimitedShardedSource.__len__: LimitedShardedSource#__len__().
  LimitedShardedSource: LimitedShardedSource#
  LimitedShardedSource._shard_names: LimitedShardedSource#_shard_names.
  TransformedShardedSource.__init__: TransformedShardedSource#__init__().
  LimitedShardedSource._remaining_before_index: LimitedShardedSource#_remaining_before_index.
  LimitedShardedSource.open_shard: LimitedShardedSource#open_shard().
  LimitedShardedSource.open_shard_at_row: LimitedShardedSource#open_shard_at_row().
  TransformedShardedSource._source: TransformedShardedSource#_source.
  TransformedShardedSource._transform: TransformedShardedSource#_transform.
  TransformedShardedSource: TransformedShardedSource#
  LimitedShardedSource._source: LimitedShardedSource#_source.
  LimitedShardedSource._count_shard_rows_up_to: LimitedShardedSource#_count_shard_rows_up_to().
  TransformedShardedSource.shard_names: TransformedShardedSource#shard_names().
  TransformedShardedSource.num_shards: TransformedShardedSource#num_shards().
  TransformedShardedSource.get_shard_info: TransformedShardedSource#get_shard_info().
  TransformedShardedSource.__repr__: TransformedShardedSource#__repr__().
  LimitedShardedSource._resolved_prefix_rows: LimitedShardedSource#_resolved_prefix_rows.
  LimitedShardedSource.__repr__: LimitedShardedSource#__repr__().
  LimitedShardedSource._shard_name_to_index: LimitedShardedSource#_shard_name_to_index.
  LimitedShardedSource._exact_shard_sizes: LimitedShardedSource#_exact_shard_sizes.
  LimitedShardedSource._max_rows: LimitedShardedSource#_max_rows.
  LimitedShardedSource._resolved_prefix_count: LimitedShardedSource#_resolved_prefix_count.
  TransformedShardedSource.__len__: TransformedShardedSource#__len__().
  LimitedShardedSource.__init__: LimitedShardedSource#__init__().
  LimitedShardedSource.shard_names: LimitedShardedSource#shard_names().
  LimitedShardedSource.num_shards: LimitedShardedSource#num_shards().
---
# Module: [`easydel/data/transforms/source.py`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py)

## Classes
### `LimitedShardedSource`  ·  implements/extends ShardedDataSource
- def: [`easydel/data/transforms/source.py:124`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L124)
- doc: ShardedDataSource wrapper that exposes at most `max_rows` examples.
- signature: `class LimitedShardedSource(ShardedDataSource[dict]):`
- members:
  - `_count_shard_rows_up_to(self, shard_name: str, limit: int)` — [`L146`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L146) — Count rows in *shard_name*, stopping early once we exceed *limit*.
  - `get_shard_info(self, shard_name: str)` — [`L224`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L224)
  - `num_shards(self)` — [`L212`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L212)
  - `open_shard(self, shard_name: str)` — [`L215`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L215)
  - `open_shard_at_row(self, shard_name: str, row: int)` — [`L218`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L218)
  - `shard_names(self)` — [`L209`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L209)
- protocol/private: `__init__`[`L136`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L136), `__len__`[`L254`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L254), `__repr__`[`L258`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L258), `_exact_shard_sizes`[`L141`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L141), `_get_known_shard_size`[`L163`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L163), `_get_shard_limit`[`L195`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L195), `_max_rows`[`L138`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L138), `_remaining_before_index`[`L144`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L144), `_resolve_prefix_until`[`L173`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L173), `_resolved_prefix_count`[`L143`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L143), `_resolved_prefix_rows`[`L142`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L142), `_shard_name_to_index`[`L140`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L140), `_shard_names`[`L139`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L139), `_source`[`L137`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L137)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`open_shard_at_row`](../core/protocols.md#ShardedDataSource.open_shard_at_row), [`get_shard_info`](../core/protocols.md#ShardedDataSource.get_shard_info), [`ShardInfo`](../core/protocols.md#ShardInfo), [`num_rows`](../core/protocols.md#ShardInfo.num_rows), [`shard_id`](../core/protocols.md#ShardInfo.shard_id), [`shard_name`](../core/protocols.md#ShardInfo.shard_name), [`byte_size`](../core/protocols.md#ShardInfo.byte_size), [`checksum`](../core/protocols.md#ShardInfo.checksum), [`url`](../core/protocols.md#ShardInfo.url)
- used by: [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`num_shards`](../core/protocols.md#ShardedDataSource.num_shards), [`open_shard_at_row`](../core/protocols.md#ShardedDataSource.open_shard_at_row), [`_create_source_from_inform`](../../infra/elarge/builders.md#_create_source_from_inform), [`get_shard_info`](../core/protocols.md#ShardedDataSource.get_shard_info)

### `TransformedShardedSource`  ·  implements/extends ShardedDataSource
- def: [`easydel/data/transforms/source.py:33`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L33)
- doc: ShardedDataSource wrapper that applies transforms during iteration.
- signature: `class TransformedShardedSource(ShardedDataSource[dict]):`
- members:
  - `__init__(self, source: ShardedDataSource[dict], transform: Transform | ExpandTransform)` — [`L48`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L48) — Initialize TransformedShardedSource.
  - `__len__(self)` — [`L113`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L113) — Return length of underlying source.
  - `get_shard_info(self, shard_name: str)` — [`L109`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L109) — Get shard info from underlying source.
  - `num_shards(self)` — [`L63`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L63) — Return number of shards from underlying source.
  - `open_shard(self, shard_name: str)` — [`L67`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L67) — Open a shard and apply transforms during iteration.
  - `open_shard_at_row(self, shard_name: str, row: int)` — [`L86`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L86) — Open a shard at a specific row and apply transforms.
  - `shard_names(self)` — [`L59`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L59) — Return shard names from underlying source.
- protocol/private: `__repr__`[`L120`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L120), `_source`[`L55`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L55), `_transform`[`L56`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/source.py#L56)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`Transform`](base.md#Transform), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`num_shards`](../core/protocols.md#ShardedDataSource.num_shards), [`open_shard_at_row`](../core/protocols.md#ShardedDataSource.open_shard_at_row), [`get_shard_info`](../core/protocols.md#ShardedDataSource.get_shard_info), [`ExpandTransform`](base.md#ExpandTransform)
- used by: [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`num_shards`](../core/protocols.md#ShardedDataSource.num_shards), [`open_shard_at_row`](../core/protocols.md#ShardedDataSource.open_shard_at_row), [`pretokenize`](../execution/pipeline.md#pretokenize), [`get_shard_info`](../core/protocols.md#ShardedDataSource.get_shard_info)

