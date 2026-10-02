---
title: 'Module: python/sgl_jax/srt/disaggregation/debug_utils.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/debug_utils.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.debug_utils`/
symbols:
  build_kv_debug_snapshot: build_kv_debug_snapshot().
  kv_debug_enabled: kv_debug_enabled().
  KVDebugSnapshot.page_digests: KVDebugSnapshot#page_digests.
  count_kv_debug_mismatches: count_kv_debug_mismatches().
  find_first_kv_debug_mismatch: find_first_kv_debug_mismatch().
  KVDebugSnapshot: KVDebugSnapshot#
  _validate_snapshot_shapes: _validate_snapshot_shapes().
  KVDebugSnapshot.sample_page_digests: KVDebugSnapshot#sample_page_digests().
  KVDebugSnapshot.global_digest: KVDebugSnapshot#global_digest.
  KVDebugSnapshot.shape: KVDebugSnapshot#shape.
  KVDebugSnapshot.dtype: KVDebugSnapshot#dtype.
  KVDebugSnapshot.sharding: KVDebugSnapshot#sharding.
  _digest_bytes: _digest_bytes().
  _host_array: _host_array().
  _ENV_ENABLE: _ENV_ENABLE.
  _ENV_REQ_FILTER: _ENV_REQ_FILTER.
  _TRUTHY: _TRUTHY.
  safe_sharding_repr: safe_sharding_repr().
---
# Module: [`python/sgl_jax/srt/disaggregation/debug_utils.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py)

## Classes
### `KVDebugSnapshot`
- def: [`python/sgl_jax/srt/disaggregation/debug_utils.py:60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L60)
- signature: `class KVDebugSnapshot:`
- members:
  - `sample_page_digests(self, *, max_layers: int = 2, max_pages: int = 4)` — [`L67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L67)
  - `dtype` — [`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L62)
  - `global_digest` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L64)
  - `page_digests` — [`L65`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L65)
  - `shape` — [`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L61)
  - `sharding` — [`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L63)
- used by: [`build_kv_debug_snapshot`](debug_utils.md#build_kv_debug_snapshot), [`_maybe_log_decode_pull_debug`](decode.md#SchedulerDisaggregationDecodeMixin._maybe_log_decode_pull_debug), [`_maybe_log_prefill_extract_debug`](prefill.md#SchedulerDisaggregationPrefillMixin._maybe_log_prefill_extract_debug), [`_maybe_verify_decode_writeback_debug`](decode.md#SchedulerDisaggregationDecodeMixin._maybe_verify_decode_writeback_debug), [`count_kv_debug_mismatches`](debug_utils.md#count_kv_debug_mismatches), [`find_first_kv_debug_mismatch`](debug_utils.md#find_first_kv_debug_mismatch), [`_validate_snapshot_shapes`](debug_utils.md#_validate_snapshot_shapes)

## Functions
- `_digest_bytes(raw: bytes)` — [`L38`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L38)
- `_host_array(value)` — [`L42`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L42)
- `_validate_snapshot_shapes(left: KVDebugSnapshot, right: KVDebugSnapshot)` — [`L121`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L121)
- `build_kv_debug_snapshot(value)` — [`L73`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L73)
- `count_kv_debug_mismatches(left: KVDebugSnapshot, right: KVDebugSnapshot)` — [`L101`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L101)
- `find_first_kv_debug_mismatch(left: KVDebugSnapshot, right: KVDebugSnapshot)` — [`L110`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L110)
- `kv_debug_enabled(req_id: str | None = None)` — [`L17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L17)
- `safe_sharding_repr(value)` — [`L28`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L28)

## Module values
- `_ENV_ENABLE` — [`L12`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L12)
- `_ENV_REQ_FILTER` — [`L13`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L13)
- `_TRUTHY` — [`L14`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/debug_utils.py#L14)

