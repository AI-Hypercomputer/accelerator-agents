---
title: 'Module: axlearn/common/kv_cache/paged_kv_storage_test.py'
type: catalog
provenance: extracted
module: axlearn/common/kv_cache/paged_kv_storage_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.kv_cache.paged_kv_storage_test`/
symbols:
  _make_bf16_storage: _make_bf16_storage().
  PagedKVStorageProtocolTest: PagedKVStorageProtocolTest#
  PagedKVStorageProtocolTest.test_bf16_storage_is_paged_storage: PagedKVStorageProtocolTest#test_bf16_storage_is_paged_storage().
  PagedKVStorageProtocolTest.test_bf16_storage_field_order: PagedKVStorageProtocolTest#test_bf16_storage_field_order().
  Bf16PagedStorageMethodsTest: Bf16PagedStorageMethodsTest#
  Bf16PagedStorageMethodsTest.test_kernel_inputs_keys_and_tensors: Bf16PagedStorageMethodsTest#test_kernel_inputs_keys_and_tensors().
  Bf16PagedStorageMethodsTest.test_kernel_inputs_page_tables_spec_defaults_to_replicated: Bf16PagedStorageMethodsTest#test_kernel_inputs_page_tables_spec_defaults_to_replicated().
  Bf16PagedStorageMethodsTest.test_as_dense_matches_reconstruct_kv: Bf16PagedStorageMethodsTest#test_as_dense_matches_reconstruct_kv().
  Bf16PagedStorageMethodsTest.test_kernel_for_missing_backend_raises_with_hint: Bf16PagedStorageMethodsTest#test_kernel_for_missing_backend_raises_with_hint().
  RegisterBf16KernelTest: RegisterBf16KernelTest#
  RegisterBf16KernelTest.setUp: RegisterBf16KernelTest#setUp().
  RegisterBf16KernelTest.tearDown: RegisterBf16KernelTest#tearDown().
  RegisterBf16KernelTest.test_register_and_lookup: RegisterBf16KernelTest#test_register_and_lookup().
  AsDenseKvDispatchTest: AsDenseKvDispatchTest#
  AsDenseKvDispatchTest.test_dense_kv_state_returns_inputs_unchanged: AsDenseKvDispatchTest#test_dense_kv_state_returns_inputs_unchanged().
---
# Module: [`axlearn/common/kv_cache/paged_kv_storage_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py)

## Classes
### `AsDenseKvDispatchTest`
- def: [`axlearn/common/kv_cache/paged_kv_storage_test.py:135`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py#L135)
- doc: Freeze `BaseKVCache.as_dense_kv` default (dense `KVState`) behaviour.
- signature: `class AsDenseKvDispatchTest(TestCase):`
- members:
  - `test_dense_kv_state_returns_inputs_unchanged(self)` — [`L142`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py#L142)

### `Bf16PagedStorageMethodsTest`
- def: [`axlearn/common/kv_cache/paged_kv_storage_test.py:70`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py#L70)
- signature: `class Bf16PagedStorageMethodsTest(TestCase):`
- members:
  - `test_as_dense_matches_reconstruct_kv(self)` — [`L98`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py#L98)
  - `test_kernel_for_missing_backend_raises_with_hint(self)` — [`L108`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py#L108)
  - `test_kernel_inputs_keys_and_tensors(self)` — [`L71`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py#L71)
  - `test_kernel_inputs_page_tables_spec_defaults_to_replicated(self)` — [`L92`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py#L92)

### `PagedKVStorageProtocolTest`
- def: [`axlearn/common/kv_cache/paged_kv_storage_test.py:54`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py#L54)
- doc: Structural-typing tests for the protocol.
- signature: `class PagedKVStorageProtocolTest(TestCase):`
- members:
  - `test_bf16_storage_field_order(self)` — [`L61`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py#L61)
  - `test_bf16_storage_is_paged_storage(self)` — [`L57`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py#L57)

### `RegisterBf16KernelTest`
- def: [`axlearn/common/kv_cache/paged_kv_storage_test.py:117`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py#L117)
- signature: `class RegisterBf16KernelTest(TestCase):`
- members:
  - `setUp(self)` — [`L118`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py#L118)
  - `tearDown(self)` — [`L122`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py#L122)
  - `test_register_and_lookup(self)` — [`L127`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py#L127)

## Functions
- `_make_bf16_storage(*, batch: int=2, num_heads: int=3, num_pages: int=8, page_size: int=4, head_dim: int=16, pages_per_request: int=3, seed: int=0)` — [`L21`](../../../../../../../raw/code/axlearn/axlearn/common/kv_cache/paged_kv_storage_test.py#L21)

