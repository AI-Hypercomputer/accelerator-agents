---
title: 'Module: axlearn/common/kv_cache/kv_cache_test.py'
type: catalog
provenance: extracted
module: axlearn/common/kv_cache/kv_cache_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.kv_cache.kv_cache_test`/KVCacheTest#
symbols:
  KVCacheTest: ''
  KVCacheTest.test_kv_cache: test_kv_cache().
  KVCacheTest.test_segment_ids: test_segment_ids().
  KVCacheTest.test_kv_cache_onehot_vs_dynamic: test_kv_cache_onehot_vs_dynamic().
  KVCacheTest.test_init_states_kv_partition_spec: test_init_states_kv_partition_spec().
---
# Module: [`axlearn/common/kv_cache/kv_cache_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/kv_cache_test.py)

## Classes
### `KVCacheTest`
- def: [`axlearn/common/kv_cache/kv_cache_test.py:16`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/kv_cache_test.py#L16)
- doc: Tests KVCache.
- signature: `class KVCacheTest(TestCase):`
- members:
  - `test_init_states_kv_partition_spec(self)` — [`L163`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/kv_cache_test.py#L163) — Verify init_states applies kv_partition_spec sharding inside jit.
  - `test_kv_cache(self, cached_kv_length, time_step_value, cache_dtype, step_len)` — [`L25`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/kv_cache_test.py#L25)
  - `test_kv_cache_onehot_vs_dynamic(self, cache_dtype)` — [`L119`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/kv_cache_test.py#L119)
  - `test_segment_ids(self)` — [`L75`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/kv_cache_test.py#L75) — segment_ids=[0,1,1,1,0] must not evict a previously cached valid token.

