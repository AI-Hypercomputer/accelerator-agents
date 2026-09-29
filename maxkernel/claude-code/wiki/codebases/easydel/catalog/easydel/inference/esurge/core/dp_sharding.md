---
title: 'Module: easydel/inference/esurge/core/dp_sharding.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/core/dp_sharding.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.core.dp_sharding`/
symbols:
  pages_per_dp_shard: pages_per_dp_shard().
  dp_shard_page_bounds: dp_shard_page_bounds().
  dp_shard_for_page_id: dp_shard_for_page_id().
  usable_pages_count: usable_pages_count().
---
# Module: [`easydel/inference/esurge/core/dp_sharding.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/dp_sharding.py)

## Functions
- `dp_shard_for_page_id(page_id: int, pages_per_shard: int, dp_size: int)` — [`L47`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/dp_sharding.py#L47) — Map a non-null page ID to a DP shard index.
- `dp_shard_page_bounds(shard_index: int, pages_per_shard: int)` — [`L55`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/dp_sharding.py#L55) — Return inclusive-exclusive page-ID bounds for a DP shard.
- `pages_per_dp_shard(num_pages: int, data_parallel_size: int | None)` — [`L31`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/dp_sharding.py#L31) — Return pages per DP shard, or ``None`` when not evenly partitionable.
- `usable_pages_count(num_pages: int)` — [`L26`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/core/dp_sharding.py#L26) — Return the number of allocatable pages (excluding null page 0).

