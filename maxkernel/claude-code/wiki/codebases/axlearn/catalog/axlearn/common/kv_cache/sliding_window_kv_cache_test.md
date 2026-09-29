---
title: 'Module: axlearn/common/kv_cache/sliding_window_kv_cache_test.py'
type: catalog
provenance: extracted
module: axlearn/common/kv_cache/sliding_window_kv_cache_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.kv_cache.sliding_window_kv_cache_test`/SlidingWindowKVCacheTest#
symbols:
  SlidingWindowKVCacheTest: ''
  SlidingWindowKVCacheTest.test_sliding_window_kv_cache: test_sliding_window_kv_cache().
  SlidingWindowKVCacheTest.test_segment_ids: test_segment_ids().
  SlidingWindowKVCacheTest.test_init_states_kv_partition_spec: test_init_states_kv_partition_spec().
---
# Module: [`axlearn/common/kv_cache/sliding_window_kv_cache_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/sliding_window_kv_cache_test.py)

## Classes
### `SlidingWindowKVCacheTest`
- def: [`axlearn/common/kv_cache/sliding_window_kv_cache_test.py:16`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/sliding_window_kv_cache_test.py#L16)
- doc: Tests SlidingWindowKVCache.
- signature: `class SlidingWindowKVCacheTest(TestCase):`
- members:
  - `test_init_states_kv_partition_spec(self)` — [`L116`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/sliding_window_kv_cache_test.py#L116) — Verify init_states applies kv_partition_spec sharding to key/value and batch-shards
  - `test_segment_ids(self)` — [`L74`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/sliding_window_kv_cache_test.py#L74) — segment_ids=[0,1,1,1,0]: leading/trailing padding must not enter the ring buffer.
  - `test_sliding_window_kv_cache(self, cached_kv_length, time_step_value, step_len)` — [`L20`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/sliding_window_kv_cache_test.py#L20)

