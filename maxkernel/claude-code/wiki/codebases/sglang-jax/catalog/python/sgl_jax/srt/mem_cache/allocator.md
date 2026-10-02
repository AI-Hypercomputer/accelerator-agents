---
title: 'Module: python/sgl_jax/srt/mem_cache/allocator.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/mem_cache/allocator.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.mem_cache.allocator`/
symbols:
  BaseTokenToKVPoolAllocator.page_size: BaseTokenToKVPoolAllocator#page_size.
  SWATokenToKVPoolAllocator.free: SWATokenToKVPoolAllocator#free().
  SWATokenToKVPoolAllocator.swa_attn_allocator: SWATokenToKVPoolAllocator#swa_attn_allocator.
  BaseTokenToKVPoolAllocator: BaseTokenToKVPoolAllocator#
  BaseTokenToKVPoolAllocator.available_size: BaseTokenToKVPoolAllocator#available_size().
  BaseTokenToKVPoolAllocator.free: BaseTokenToKVPoolAllocator#free().
  SWATokenToKVPoolAllocator.full_attn_allocator: SWATokenToKVPoolAllocator#full_attn_allocator.
  PagedTokenToKVPoolAllocator.free_pages: PagedTokenToKVPoolAllocator#free_pages.
  SWATokenToKVPoolAllocator.full_to_swa_index_mapping: SWATokenToKVPoolAllocator#full_to_swa_index_mapping.
  SWATokenToKVPoolAllocator: SWATokenToKVPoolAllocator#
  SWATokenToKVPoolAllocator.alloc: SWATokenToKVPoolAllocator#alloc().
  TokenToKVPoolAllocator.clear: TokenToKVPoolAllocator#clear().
  BaseTokenToKVPoolAllocator.dp_size: BaseTokenToKVPoolAllocator#dp_size.
  BaseTokenToKVPoolAllocator.free_group_end: BaseTokenToKVPoolAllocator#free_group_end().
  PagedTokenToKVPoolAllocator.free: PagedTokenToKVPoolAllocator#free().
  PagedTokenToKVPoolAllocator.clear: PagedTokenToKVPoolAllocator#clear().
  SWATokenToKVPoolAllocator.alloc_extend: SWATokenToKVPoolAllocator#alloc_extend().
  SWATokenToKVPoolAllocator.alloc_decode: SWATokenToKVPoolAllocator#alloc_decode().
  BaseTokenToKVPoolAllocator.get_kvcache: BaseTokenToKVPoolAllocator#get_kvcache().
  SWATokenToKVPoolAllocator.free_group: SWATokenToKVPoolAllocator#free_group.
  BaseTokenToKVPoolAllocator.free_group_begin: BaseTokenToKVPoolAllocator#free_group_begin().
  TokenToKVPoolAllocator.free_slots: TokenToKVPoolAllocator#free_slots.
  SWATokenToKVPoolAllocator.clear: SWATokenToKVPoolAllocator#clear().
  SWATokenToKVPoolAllocator.free_swa: SWATokenToKVPoolAllocator#free_swa().
  TokenToKVPoolAllocator.free: TokenToKVPoolAllocator#free().
  TokenToKVPoolAllocator.available_size: TokenToKVPoolAllocator#available_size().
  PagedTokenToKVPoolAllocator.alloc_extend: PagedTokenToKVPoolAllocator#alloc_extend().
  PagedTokenToKVPoolAllocator.alloc_decode: PagedTokenToKVPoolAllocator#alloc_decode().
  BaseTokenToKVPoolAllocator.backup_state: BaseTokenToKVPoolAllocator#backup_state().
  BaseTokenToKVPoolAllocator.restore_state: BaseTokenToKVPoolAllocator#restore_state().
  TokenToKVPoolAllocator.free_group_end: TokenToKVPoolAllocator#free_group_end().
  PagedTokenToKVPoolAllocator.free_group_end: PagedTokenToKVPoolAllocator#free_group_end().
  SWATokenToKVPoolAllocator.free_group_end: SWATokenToKVPoolAllocator#free_group_end().
  BaseTokenToKVPoolAllocator.alloc_decode: BaseTokenToKVPoolAllocator#alloc_decode().
  BaseTokenToKVPoolAllocator.clear: BaseTokenToKVPoolAllocator#clear().
  BaseTokenToKVPoolAllocator.alloc: BaseTokenToKVPoolAllocator#alloc().
  BaseTokenToKVPoolAllocator.merge_and_sort_free: BaseTokenToKVPoolAllocator#merge_and_sort_free().
  PagedTokenToKVPoolAllocator.alloc: PagedTokenToKVPoolAllocator#alloc().
  SWATokenToKVPoolAllocator.available_size: SWATokenToKVPoolAllocator#available_size().
  SWATokenToKVPoolAllocator.debug_print: SWATokenToKVPoolAllocator#debug_print().
  BaseTokenToKVPoolAllocator._kvcache: BaseTokenToKVPoolAllocator#_kvcache.
  BaseTokenToKVPoolAllocator.alloc_extend: BaseTokenToKVPoolAllocator#alloc_extend().
  SWATokenToKVPoolAllocator.full_available_size: SWATokenToKVPoolAllocator#full_available_size().
  SWATokenToKVPoolAllocator.swa_available_size: SWATokenToKVPoolAllocator#swa_available_size().
  TokenToKVPoolAllocator: TokenToKVPoolAllocator#
  PagedTokenToKVPoolAllocator: PagedTokenToKVPoolAllocator#
  BaseTokenToKVPoolAllocator.size_per_rank: BaseTokenToKVPoolAllocator#size_per_rank.
  BaseTokenToKVPoolAllocator.__init__: BaseTokenToKVPoolAllocator#__init__().
  BaseTokenToKVPoolAllocator.free_pages: BaseTokenToKVPoolAllocator#free_pages.
  BaseTokenToKVPoolAllocator.release_pages: BaseTokenToKVPoolAllocator#release_pages.
  BaseTokenToKVPoolAllocator.get_cpu_copy: BaseTokenToKVPoolAllocator#get_cpu_copy().
  BaseTokenToKVPoolAllocator.load_cpu_copy: BaseTokenToKVPoolAllocator#load_cpu_copy().
  TokenToKVPoolAllocator.__init__: TokenToKVPoolAllocator#__init__().
  TokenToKVPoolAllocator.free_group: TokenToKVPoolAllocator#free_group.
  TokenToKVPoolAllocator.get_cpu_copy: TokenToKVPoolAllocator#get_cpu_copy().
  TokenToKVPoolAllocator.load_cpu_copy: TokenToKVPoolAllocator#load_cpu_copy().
  TokenToKVPoolAllocator.backup_state: TokenToKVPoolAllocator#backup_state().
  PagedTokenToKVPoolAllocator.__init__: PagedTokenToKVPoolAllocator#__init__().
  PagedTokenToKVPoolAllocator.free_group: PagedTokenToKVPoolAllocator#free_group.
  PagedTokenToKVPoolAllocator.get_cpu_copy: PagedTokenToKVPoolAllocator#get_cpu_copy().
  PagedTokenToKVPoolAllocator.load_cpu_copy: PagedTokenToKVPoolAllocator#load_cpu_copy().
  SWATokenToKVPoolAllocator.__init__: SWATokenToKVPoolAllocator#__init__().
  SWATokenToKVPoolAllocator.count_swa_mapped: SWATokenToKVPoolAllocator#count_swa_mapped().
  TokenToKVPoolAllocator.alloc: TokenToKVPoolAllocator#alloc().
  PagedTokenToKVPoolAllocator.pages_per_rank: PagedTokenToKVPoolAllocator#pages_per_rank.
  TokenToKVPoolAllocator.is_not_in_free_group: TokenToKVPoolAllocator#is_not_in_free_group.
  PagedTokenToKVPoolAllocator.is_not_in_free_group: PagedTokenToKVPoolAllocator#is_not_in_free_group.
  PagedTokenToKVPoolAllocator._alloc_extend_impl: PagedTokenToKVPoolAllocator#_alloc_extend_impl().
  PagedTokenToKVPoolAllocator._alloc_decode_impl: PagedTokenToKVPoolAllocator#_alloc_decode_impl().
  BaseTokenToKVPoolAllocator.free_group: BaseTokenToKVPoolAllocator#free_group.
  PagedTokenToKVPoolAllocator.debug_mode: PagedTokenToKVPoolAllocator#debug_mode.
  PagedTokenToKVPoolAllocator.release_pages: PagedTokenToKVPoolAllocator#release_pages.
  SWATokenToKVPoolAllocator.is_not_in_free_group: SWATokenToKVPoolAllocator#is_not_in_free_group.
  BaseTokenToKVPoolAllocator.debug_print: BaseTokenToKVPoolAllocator#debug_print().
  TokenToKVPoolAllocator.origin_size: TokenToKVPoolAllocator#origin_size.
  TokenToKVPoolAllocator.free_group_begin: TokenToKVPoolAllocator#free_group_begin().
  TokenToKVPoolAllocator.restore_state: TokenToKVPoolAllocator#restore_state().
  PagedTokenToKVPoolAllocator.free_group_begin: PagedTokenToKVPoolAllocator#free_group_begin().
  SWATokenToKVPoolAllocator.size_full: SWATokenToKVPoolAllocator#size_full().
  SWATokenToKVPoolAllocator.size_swa: SWATokenToKVPoolAllocator#size_swa().
  SWATokenToKVPoolAllocator.get_kvcache: SWATokenToKVPoolAllocator#get_kvcache().
  SWATokenToKVPoolAllocator.free_group_begin: SWATokenToKVPoolAllocator#free_group_begin().
  BaseTokenToKVPoolAllocator.size: BaseTokenToKVPoolAllocator#size.
  BaseTokenToKVPoolAllocator.is_not_in_free_group: BaseTokenToKVPoolAllocator#is_not_in_free_group.
  TokenToKVPoolAllocator.release_pages: TokenToKVPoolAllocator#release_pages.
  PagedTokenToKVPoolAllocator.num_pages: PagedTokenToKVPoolAllocator#num_pages.
  SWATokenToKVPoolAllocator._size_full: SWATokenToKVPoolAllocator#_size_full.
  SWATokenToKVPoolAllocator._size_swa: SWATokenToKVPoolAllocator#_size_swa.
  logger: logger.
  SWATokenToKVPoolAllocator.backup_state: SWATokenToKVPoolAllocator#backup_state().
  SWATokenToKVPoolAllocator.restore_state: SWATokenToKVPoolAllocator#restore_state().
---
# Module: [`python/sgl_jax/srt/mem_cache/allocator.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py)

## Classes
### `BaseTokenToKVPoolAllocator`  ·  implements/extends ABC
- def: [`python/sgl_jax/srt/mem_cache/allocator.py:11`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L11) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
- signature: `class BaseTokenToKVPoolAllocator(abc.ABC):`
- members:
  - `alloc(self, need_size: int, dp_rank: int = 0)` — [`L83`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L83)
  - `alloc_decode(self, *args, **kwargs)` — [`L75`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L75)
  - `alloc_extend(self, *args, **kwargs)` — [`L72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L72)
  - `available_size(self, dp_rank: int = 0)` — [`L35`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L35) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
  - `backup_state(self)` — [`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L44)
  - `clear(self, dp_rank: int | None = None)` — [`L79`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L79)
  - `debug_print(self)` — [`L32`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L32)
  - `free(self, free_index: np.ndarray, dp_rank: int = 0)` — [`L87`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L87) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
  - `free_group_begin(self)` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L47)
  - `free_group_end(self)` — [`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L51)
  - `get_cpu_copy(self, *args, **kwargs)` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L64)
  - `get_kvcache(self)` — [`L38`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L38)
  - `load_cpu_copy(self, *args, **kwargs)` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L68)
  - `merge_and_sort_free(self, dp_rank: int = 0)` — [`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L57)
  - `restore_state(self, state)` — [`L41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L41)
  - `dp_size` — [`L22`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L22) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
  - `free_group` — [`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L30)
  - `free_pages` — [`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L27)
  - `is_not_in_free_group` — [`L29`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L29)
  - `page_size` — [`L21`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L21) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
  - `release_pages` — [`L28`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L28)
  - `size` — [`L20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L20)
  - `size_per_rank` — [`L23`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L23)
- protocol/private: `__init__`[`L13`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L13), `_kvcache`[`L25`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L25)
- uses (calls/refs, reference-scoped): [`free`](allocator.md#SWATokenToKVPoolAllocator.free), [`KVCache`](memory_pool.md#KVCache), [`SWATokenToKVPoolAllocator`](allocator.md#SWATokenToKVPoolAllocator), [`alloc`](allocator.md#SWATokenToKVPoolAllocator.alloc), [`clear`](allocator.md#TokenToKVPoolAllocator.clear), [`free`](allocator.md#PagedTokenToKVPoolAllocator.free), [`alloc_decode`](allocator.md#SWATokenToKVPoolAllocator.alloc_decode), [`alloc_extend`](allocator.md#SWATokenToKVPoolAllocator.alloc_extend), [`clear`](allocator.md#PagedTokenToKVPoolAllocator.clear), [`clear`](allocator.md#SWATokenToKVPoolAllocator.clear), [`alloc_decode`](allocator.md#PagedTokenToKVPoolAllocator.alloc_decode), [`alloc_extend`](allocator.md#PagedTokenToKVPoolAllocator.alloc_extend), [`available_size`](allocator.md#TokenToKVPoolAllocator.available_size), [`free`](allocator.md#TokenToKVPoolAllocator.free), [`free_group_end`](allocator.md#PagedTokenToKVPoolAllocator.free_group_end), [`free_group_end`](allocator.md#SWATokenToKVPoolAllocator.free_group_end), [`free_group_end`](allocator.md#TokenToKVPoolAllocator.free_group_end), [`alloc`](allocator.md#PagedTokenToKVPoolAllocator.alloc), [`available_size`](allocator.md#SWATokenToKVPoolAllocator.available_size), [`debug_print`](allocator.md#SWATokenToKVPoolAllocator.debug_print), [`PagedTokenToKVPoolAllocator`](allocator.md#PagedTokenToKVPoolAllocator), [`TokenToKVPoolAllocator`](allocator.md#TokenToKVPoolAllocator), [`backup_state`](allocator.md#TokenToKVPoolAllocator.backup_state), [`get_cpu_copy`](allocator.md#PagedTokenToKVPoolAllocator.get_cpu_copy), [`get_cpu_copy`](allocator.md#TokenToKVPoolAllocator.get_cpu_copy), [`load_cpu_copy`](allocator.md#PagedTokenToKVPoolAllocator.load_cpu_copy), [`load_cpu_copy`](allocator.md#TokenToKVPoolAllocator.load_cpu_copy), [`alloc`](allocator.md#TokenToKVPoolAllocator.alloc), [`free_group_begin`](allocator.md#PagedTokenToKVPoolAllocator.free_group_begin), [`free_group_begin`](allocator.md#SWATokenToKVPoolAllocator.free_group_begin), [`free_group_begin`](allocator.md#TokenToKVPoolAllocator.free_group_begin), [`get_kvcache`](allocator.md#SWATokenToKVPoolAllocator.get_kvcache), [`restore_state`](allocator.md#TokenToKVPoolAllocator.restore_state), [`backup_state`](allocator.md#SWATokenToKVPoolAllocator.backup_state), [`restore_state`](allocator.md#SWATokenToKVPoolAllocator.restore_state)
- used by: [`process_batch_result_decode`](../managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_decode), [`install_disaggregation_wiring`](../disaggregation/runtime.md#install_disaggregation_wiring), `prepare_for_extend`, `prepare_for_decode`, [`_admit_decode_prealloc`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._admit_decode_prealloc), [`cache_unfinished_req`](unified_radix_cache.md#UnifiedRadixCache.cache_unfinished_req), [`get_internal_state`](../managers/scheduler.md#Scheduler.get_internal_state), [`cache_unfinished_req`](radix_cache.md#RadixCache.cache_unfinished_req), [`flush_cache`](../managers/scheduler.md#Scheduler.flush_cache), [`cache_finished_req`](unified_radix_cache.md#UnifiedRadixCache.cache_finished_req), [`build_kv_cache`](kv_cache_builder.md#build_kv_cache), [`process_input_requests_disagg_decode`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin.process_input_requests_disagg_decode), [`prepare_for_decode`](../speculative/eagle_util.md#EagleDraftInput.prepare_for_decode), [`cache_finished_req`](radix_cache.md#RadixCache.cache_finished_req), `init_new`, [`_insert_helper`](unified_radix_cache.md#UnifiedRadixCache._insert_helper), [`_write_kv_to_pool`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._write_kv_to_pool), `token_to_kv_pool_allocator`, [`_extract_req_kv`](../disaggregation/prefill.md#SchedulerDisaggregationPrefillMixin._extract_req_kv), [`check_memory`](../managers/scheduler.md#Scheduler.check_memory), [`free`](allocator.md#SWATokenToKVPoolAllocator.free), [`swa_attn_allocator`](allocator.md#SWATokenToKVPoolAllocator.swa_attn_allocator), `_evict_tree_cache_if_needed`, [`evict`](radix_cache.md#RadixCache.evict), `_available_and_evictable_str`, [`_build_kv_spec_for_req`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._build_kv_spec_for_req), [`rem_total_tokens_for_dp`](../managers/schedule_policy.md#PrefillAdder.rem_total_tokens_for_dp), [`_decode_backlog_snapshot`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._decode_backlog_snapshot), [`cur_rem_tokens_for_dp`](../managers/schedule_policy.md#PrefillAdder.cur_rem_tokens_for_dp), [`cache_finished_req`](chunk_cache.md#ChunkCache.cache_finished_req), `alloc_paged_token_slots_decode`, `has_sufficient_memory`, [`SWATokenToKVPoolAllocator`](allocator.md#SWATokenToKVPoolAllocator), [`alloc`](allocator.md#SWATokenToKVPoolAllocator.alloc), [`_get_token_info`](../managers/scheduler.md#Scheduler._get_token_info), [`clear`](allocator.md#TokenToKVPoolAllocator.clear), `new_tokens_required_next_decode`, [`free`](allocator.md#PagedTokenToKVPoolAllocator.free), [`_release_decode_kv_indices`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._release_decode_kv_indices), [`alloc_decode`](allocator.md#SWATokenToKVPoolAllocator.alloc_decode)  (+33 more)

### `PagedTokenToKVPoolAllocator`  ·  implements/extends BaseTokenToKVPoolAllocator
- def: [`python/sgl_jax/srt/mem_cache/allocator.py:161`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L161)
- signature: `class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):`
- members:
  - `_alloc_decode_impl(self, allocated_pages: np.ndarray, needs_new_page: np.ndarray, last_loc_np: np.ndarray, batch_size: int)` — [`L302`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L302) — Common implementation for alloc_decode.
  - `_alloc_extend_impl(self, allocated_pages: np.ndarray, prefix_lens_np: np.ndarray, extend_lens: np.ndarray, last_loc_np: np.ndarray, extend_num_tokens: int)` — [`L202`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L202) — Common implementation for alloc_extend.
  - `alloc(self, need_size: int, dp_rank: int = 0)` — [`L184`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L184) — Page-aligned allocation.
  - `alloc_decode(self, seq_lens: list[int], last_loc: list[int], dp_rank: int = 0)` — [`L325`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L325)
  - `alloc_extend(self, prefix_lens: list[int], seq_lens: list[int], last_loc: list[int], extend_num_tokens: int, dp_rank: int = 0)` — [`L268`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L268)
  - `clear(self, dp_rank: int | None = None)` — [`L387`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L387)
  - `free(self, free_index: np.ndarray, dp_rank: int = 0)` — [`L358`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L358) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
  - `free_group_begin(self)` — [`L376`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L376)
  - `free_group_end(self)` — [`L379`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L379)
  - `get_cpu_copy(self, indices)` — [`L395`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L395)
  - `load_cpu_copy(self, kv_cache_cpu, indices)` — [`L398`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L398)
  - `debug_mode` — [`L174`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L174)
  - `free_group` — [`L181`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L181)
  - `free_pages` — [`L177`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L177)
  - `is_not_in_free_group` — [`L182`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L182)
  - `num_pages` — [`L172`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L172)
  - `pages_per_rank` — [`L173`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L173)
  - `release_pages` — [`L180`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L180)
- protocol/private: `__init__`[`L162`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L162)
- uses (calls/refs, reference-scoped): [`page_size`](allocator.md#BaseTokenToKVPoolAllocator.page_size), [`KVCache`](memory_pool.md#KVCache), [`BaseTokenToKVPoolAllocator`](allocator.md#BaseTokenToKVPoolAllocator), [`dp_size`](allocator.md#BaseTokenToKVPoolAllocator.dp_size), [`get_cpu_copy`](memory_pool.md#KVCache.get_cpu_copy), [`load_cpu_copy`](memory_pool.md#KVCache.load_cpu_copy), [`merge_and_sort_free`](allocator.md#BaseTokenToKVPoolAllocator.merge_and_sort_free), [`_kvcache`](allocator.md#BaseTokenToKVPoolAllocator._kvcache), [`__init__`](allocator.md#BaseTokenToKVPoolAllocator.__init__)
- used by: [`_init_pools`](../model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._init_pools), [`swa_attn_allocator`](allocator.md#SWATokenToKVPoolAllocator.swa_attn_allocator), [`BaseTokenToKVPoolAllocator`](allocator.md#BaseTokenToKVPoolAllocator), [`free`](allocator.md#BaseTokenToKVPoolAllocator.free), [`free_group_end`](allocator.md#BaseTokenToKVPoolAllocator.free_group_end), [`free_group_begin`](allocator.md#BaseTokenToKVPoolAllocator.free_group_begin), [`alloc_decode`](allocator.md#BaseTokenToKVPoolAllocator.alloc_decode), [`alloc`](allocator.md#BaseTokenToKVPoolAllocator.alloc), [`clear`](allocator.md#BaseTokenToKVPoolAllocator.clear), [`alloc_extend`](allocator.md#BaseTokenToKVPoolAllocator.alloc_extend), [`get_cpu_copy`](allocator.md#BaseTokenToKVPoolAllocator.get_cpu_copy), [`load_cpu_copy`](allocator.md#BaseTokenToKVPoolAllocator.load_cpu_copy)

### `SWATokenToKVPoolAllocator`  ·  implements/extends BaseTokenToKVPoolAllocator
- def: [`python/sgl_jax/srt/mem_cache/allocator.py:402`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L402) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
- doc: Allocator for SWA hybrid KV cache.
- signature: `class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):`
- members:
  - `alloc(self, need_size: int, dp_rank: int = 0)` — [`L489`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L489) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
  - `alloc_decode(self, seq_lens: list[int], last_loc: list[int], dp_rank: int = 0)` — [`L541`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L541) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
  - `alloc_extend(self, prefix_lens: list[int], seq_lens: list[int], last_loc: list[int], extend_num_tokens: int, dp_rank: int = 0)` — [`L509`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L509) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
  - `available_size(self, dp_rank: int = 0)` — [`L460`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L460)
  - `backup_state(self)` — [`L608`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L608)
  - `clear(self, dp_rank: int | None = None)` — [`L614`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L614)
  - `count_swa_mapped(self, indices: np.array, dp_rank: int = 0)` — [`L599`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L599) — Count how many of the given full indices have an active SWA mapping.
  - `debug_print(self)` — [`L480`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L480)
  - `free(self, free_index: np.array, dp_rank: int = 0)` — [`L567`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L567) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
  - `free_group_begin(self)` — [`L633`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L633)
  - `free_group_end(self)` — [`L636`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L636)
  - `free_swa(self, free_index: np.array, dp_rank: int = 0)` — [`L588`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L588) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
  - `full_available_size(self, dp_rank: int = 0)` — [`L466`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L466)
  - `get_kvcache(self)` — [`L486`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L486)
  - `restore_state(self, state)` — [`L611`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L611)
  - `size_full(self)` — [`L473`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L473)
  - `size_swa(self)` — [`L477`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L477)
  - `swa_available_size(self, dp_rank: int = 0)` — [`L469`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L469)
  - `free_group` — [`L455`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L455) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
  - `full_attn_allocator` — [`L419`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L419) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
  - `full_to_swa_index_mapping` — [`L449`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L449) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
  - `is_not_in_free_group` — [`L454`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L454)
  - `swa_attn_allocator` — [`L424`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L424) — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
- protocol/private: `__init__`[`L405`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L405), `_size_full`[`L415`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L415), `_size_swa`[`L416`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L416)
- uses (calls/refs, reference-scoped): [`BaseTokenToKVPoolAllocator`](allocator.md#BaseTokenToKVPoolAllocator), [`clear`](allocator.md#TokenToKVPoolAllocator.clear), [`dp_size`](allocator.md#BaseTokenToKVPoolAllocator.dp_size), [`available_size`](allocator.md#TokenToKVPoolAllocator.available_size), [`free`](allocator.md#TokenToKVPoolAllocator.free), [`full_kv_pool`](memory_pool.md#SWAKVPool.full_kv_pool), [`swa_kv_pool`](memory_pool.md#SWAKVPool.swa_kv_pool), [`alloc_decode`](allocator.md#BaseTokenToKVPoolAllocator.alloc_decode), [`SWAKVPool`](memory_pool.md#SWAKVPool), [`_kvcache`](allocator.md#BaseTokenToKVPoolAllocator._kvcache), [`alloc_extend`](allocator.md#BaseTokenToKVPoolAllocator.alloc_extend), [`PagedTokenToKVPoolAllocator`](allocator.md#PagedTokenToKVPoolAllocator), [`TokenToKVPoolAllocator`](allocator.md#TokenToKVPoolAllocator), [`size_per_rank`](allocator.md#BaseTokenToKVPoolAllocator.size_per_rank), [`__init__`](allocator.md#BaseTokenToKVPoolAllocator.__init__), [`alloc`](allocator.md#TokenToKVPoolAllocator.alloc), [`size`](allocator.md#BaseTokenToKVPoolAllocator.size)
- used by: [`evict`](swa_radix_cache.md#SWARadixCache.evict), [`_init_pools`](../model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._init_pools), [`cache_finished_req`](swa_radix_cache.md#SWARadixCache.cache_finished_req), [`_insert_helper`](swa_radix_cache.md#SWARadixCache._insert_helper), `init_new`, [`_iteratively_delete_tombstone_leaf`](swa_radix_cache.md#SWARadixCache._iteratively_delete_tombstone_leaf), [`evict_from_tree_cache`](common.md#evict_from_tree_cache), [`BaseTokenToKVPoolAllocator`](allocator.md#BaseTokenToKVPoolAllocator), [`evict_req_swa`](swa_radix_cache.md#SWARadixCache.evict_req_swa), [`available_size`](allocator.md#BaseTokenToKVPoolAllocator.available_size), [`free`](allocator.md#BaseTokenToKVPoolAllocator.free), [`free_group_end`](allocator.md#BaseTokenToKVPoolAllocator.free_group_end), [`is_hybrid`](../managers/schedule_policy.md#PrefillAdder.is_hybrid), [`get_kvcache`](allocator.md#BaseTokenToKVPoolAllocator.get_kvcache), [`free_group_begin`](allocator.md#BaseTokenToKVPoolAllocator.free_group_begin), [`_swa_eff_len`](swa_radix_cache.md#SWARadixCache._swa_eff_len), [`backup_state`](allocator.md#BaseTokenToKVPoolAllocator.backup_state), [`restore_state`](allocator.md#BaseTokenToKVPoolAllocator.restore_state), [`available_and_evictable_str`](common.md#available_and_evictable_str), [`alloc_decode`](allocator.md#BaseTokenToKVPoolAllocator.alloc_decode), [`alloc`](allocator.md#BaseTokenToKVPoolAllocator.alloc), [`clear`](allocator.md#BaseTokenToKVPoolAllocator.clear), [`__init__`](chunk_cache.md#SWAChunkCache.__init__), [`alloc_extend`](allocator.md#BaseTokenToKVPoolAllocator.alloc_extend), [`__init__`](swa_radix_cache.md#SWARadixCache.__init__), [`debug_print`](allocator.md#BaseTokenToKVPoolAllocator.debug_print)

### `TokenToKVPoolAllocator`  ·  implements/extends BaseTokenToKVPoolAllocator
- def: [`python/sgl_jax/srt/mem_cache/allocator.py:91`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L91)
- signature: `class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):`
- members:
  - `alloc(self, need_size: int, dp_rank: int = 0)` — [`L109`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L109)
  - `available_size(self, dp_rank: int = 0)` — [`L125`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L125)
  - `backup_state(self)` — [`L153`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L153)
  - `clear(self, dp_rank: int | None = None)` — [`L128`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L128)
  - `free(self, free_index: np.ndarray, dp_rank: int = 0)` — [`L117`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L117)
  - `free_group_begin(self)` — [`L135`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L135)
  - `free_group_end(self)` — [`L138`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L138)
  - `get_cpu_copy(self, indices)` — [`L147`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L147)
  - `load_cpu_copy(self, kv_cache_cpu, indices)` — [`L150`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L150)
  - `restore_state(self, state)` — [`L156`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L156)
  - `free_group` — [`L106`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L106)
  - `free_slots` — [`L102`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L102)
  - `is_not_in_free_group` — [`L107`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L107)
  - `origin_size` — [`L105`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L105)
  - `release_pages` — [`L158`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L158)
- protocol/private: `__init__`[`L92`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L92)
- uses (calls/refs, reference-scoped): [`KVCache`](memory_pool.md#KVCache), [`BaseTokenToKVPoolAllocator`](allocator.md#BaseTokenToKVPoolAllocator), [`dp_size`](allocator.md#BaseTokenToKVPoolAllocator.dp_size), [`get_cpu_copy`](memory_pool.md#KVCache.get_cpu_copy), [`load_cpu_copy`](memory_pool.md#KVCache.load_cpu_copy), [`_kvcache`](allocator.md#BaseTokenToKVPoolAllocator._kvcache), [`size_per_rank`](allocator.md#BaseTokenToKVPoolAllocator.size_per_rank), [`__init__`](allocator.md#BaseTokenToKVPoolAllocator.__init__)
- used by: [`_init_pools`](../model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._init_pools), [`free`](allocator.md#SWATokenToKVPoolAllocator.free), [`swa_attn_allocator`](allocator.md#SWATokenToKVPoolAllocator.swa_attn_allocator), [`BaseTokenToKVPoolAllocator`](allocator.md#BaseTokenToKVPoolAllocator), [`available_size`](allocator.md#BaseTokenToKVPoolAllocator.available_size), [`free`](allocator.md#BaseTokenToKVPoolAllocator.free), [`full_attn_allocator`](allocator.md#SWATokenToKVPoolAllocator.full_attn_allocator), [`alloc`](allocator.md#SWATokenToKVPoolAllocator.alloc), [`free_group_end`](allocator.md#BaseTokenToKVPoolAllocator.free_group_end), [`alloc_decode`](allocator.md#SWATokenToKVPoolAllocator.alloc_decode), [`alloc_extend`](allocator.md#SWATokenToKVPoolAllocator.alloc_extend), [`clear`](allocator.md#SWATokenToKVPoolAllocator.clear), [`free_group_begin`](allocator.md#BaseTokenToKVPoolAllocator.free_group_begin), [`free_swa`](allocator.md#SWATokenToKVPoolAllocator.free_swa), [`backup_state`](allocator.md#BaseTokenToKVPoolAllocator.backup_state), [`restore_state`](allocator.md#BaseTokenToKVPoolAllocator.restore_state), [`alloc`](allocator.md#BaseTokenToKVPoolAllocator.alloc), [`clear`](allocator.md#BaseTokenToKVPoolAllocator.clear), [`available_size`](allocator.md#SWATokenToKVPoolAllocator.available_size), [`debug_print`](allocator.md#SWATokenToKVPoolAllocator.debug_print), [`full_available_size`](allocator.md#SWATokenToKVPoolAllocator.full_available_size), [`swa_available_size`](allocator.md#SWATokenToKVPoolAllocator.swa_available_size), [`get_cpu_copy`](allocator.md#BaseTokenToKVPoolAllocator.get_cpu_copy), [`load_cpu_copy`](allocator.md#BaseTokenToKVPoolAllocator.load_cpu_copy)

## Module values
- `logger` — [`L8`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/allocator.py#L8)

