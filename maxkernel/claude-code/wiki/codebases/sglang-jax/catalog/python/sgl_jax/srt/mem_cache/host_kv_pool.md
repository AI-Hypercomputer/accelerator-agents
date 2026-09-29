---
title: 'Module: python/sgl_jax/srt/mem_cache/host_kv_pool.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/mem_cache/host_kv_pool.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.mem_cache.host_kv_pool`/
symbols:
  LRUHostKVPool.stage_backup: LRUHostKVPool#stage_backup().
  LRUHostKVPool.flush_load: LRUHostKVPool#flush_load().
  LRUHostKVPool.precompile_transfers: LRUHostKVPool#precompile_transfers().
  LRUHostKVPool.alloc: LRUHostKVPool#alloc().
  QueueHostKVPool.copy_from_device: QueueHostKVPool#copy_from_device().
  LRUHostKVPool.reserve: LRUHostKVPool#reserve().
  QueueHostKVPool.reserve: QueueHostKVPool#reserve().
  LRUHostKVPool.stage_load: LRUHostKVPool#stage_load().
  LRUHostKVPool.free: LRUHostKVPool#free().
  LRUHostKVPool.flush_backup: LRUHostKVPool#flush_backup().
  LRUHostKVPool.release: LRUHostKVPool#release().
  LRUHostKVPool._pool_size: LRUHostKVPool#_pool_size.
  QueueHostKVPool._release: QueueHostKVPool#_release().
  LRUHostKVPool._lock: LRUHostKVPool#_lock.
  LRUHostKVPool._free_ids: LRUHostKVPool#_free_ids.
  LRUHostKVPool._drop_pending: LRUHostKVPool#_drop_pending().
  HostKVPool: HostKVPool#
  LRUHostKVPool._allocated: LRUHostKVPool#_allocated.
  LRUHostKVPool._require_allocated: LRUHostKVPool#_require_allocated().
  LRUHostKVPool._generation: LRUHostKVPool#_generation.
  LRUHostKVPool.inc_lock_ref: LRUHostKVPool#inc_lock_ref().
  LRUHostKVPool.dec_lock_ref: LRUHostKVPool#dec_lock_ref().
  logger: logger.
  QueueHostKVPool._pool_size: QueueHostKVPool#_pool_size.
  LRUHostKVPool._lock_ref: LRUHostKVPool#_lock_ref.
  HostKVPool.copy_from_device: HostKVPool#copy_from_device().
  QueueHostKVPool._free_ids: QueueHostKVPool#_free_ids.
  QueueHostKVPool._inc_alloc_metric: QueueHostKVPool#_inc_alloc_metric().
  LRUHostKVPool._host_sharding: LRUHostKVPool#_host_sharding.
  LRUHostKVPool._batched_layer_sharding: LRUHostKVPool#_batched_layer_sharding.
  LRUHostKVPool.available_size: LRUHostKVPool#available_size().
  HostKVPool.reserve: HostKVPool#reserve().
  HostKVPool.release: HostKVPool#release().
  HostKVPool.available_size: HostKVPool#available_size().
  HostKVPool.total_size: HostKVPool#total_size().
  QueueHostKVPool.available_size: QueueHostKVPool#available_size().
  LRUHostKVPool._peak_used: LRUHostKVPool#_peak_used.
  LRUHostKVPool.copy_into: LRUHostKVPool#copy_into().
  LRUHostKVPool.copy_to_device: LRUHostKVPool#copy_to_device().
  _make_host_sharding: _make_host_sharding().
  QueueHostKVPool: QueueHostKVPool#
  LRUHostKVPool._pad_to_page_bucket: LRUHostKVPool#_pad_to_page_bucket.
  LRUHostKVPool._require_device_pages: LRUHostKVPool#_require_device_pages().
  LRUHostKVPool._device_pool: LRUHostKVPool#_device_pool.
  LRUHostKVPool._slots: LRUHostKVPool#_slots.
  LRUHostKVPool._exhaust_count: LRUHostKVPool#_exhaust_count.
  LRUHostKVPool._last_exhaust_log: LRUHostKVPool#_last_exhaust_log.
  QueueHostKVPool._host_sharding: QueueHostKVPool#_host_sharding.
  QueueHostKVPool._record_d2h_bytes: QueueHostKVPool#_record_d2h_bytes().
  LRUHostKVPool._device_packed_sharding: LRUHostKVPool#_device_packed_sharding.
  LRUHostKVPool._jit_gather_one_layer: LRUHostKVPool#_jit_gather_one_layer.
  StagedData: StagedData#
  QueueHostKVPool._pool_name: QueueHostKVPool#_pool_name.
  QueueHostKVPool._lock: QueueHostKVPool#_lock.
  QueueHostKVPool._peak_used: QueueHostKVPool#_peak_used.
  LRUHostKVPool._pending_lock: LRUHostKVPool#_pending_lock.
  LRUHostKVPool._pending_gather: LRUHostKVPool#_pending_gather.
  LRUHostKVPool._pending_load_lock: LRUHostKVPool#_pending_load_lock.
  LRUHostKVPool._pending_load: LRUHostKVPool#_pending_load.
  HostKVPool.copy_into: HostKVPool#copy_into().
  HostKVPool.copy_to_device: HostKVPool#copy_to_device().
  QueueHostKVPool.release: QueueHostKVPool#release().
  QueueHostKVPool.total_size: QueueHostKVPool#total_size().
  LRUHostKVPool: LRUHostKVPool#
  LRUHostKVPool.total_size: LRUHostKVPool#total_size().
  StagedData.array_pytree: StagedData#array_pytree.
  QueueHostKVPool._max_padded_pages: QueueHostKVPool#_max_padded_pages.
  QueueHostKVPool._layer_num: QueueHostKVPool#_layer_num.
  QueueHostKVPool._exhaust_count: QueueHostKVPool#_exhaust_count.
  QueueHostKVPool._last_exhaust_log: QueueHostKVPool#_last_exhaust_log.
  LRUHostKVPool._page_size: LRUHostKVPool#_page_size.
  LRUHostKVPool._layer_num: LRUHostKVPool#_layer_num.
  LRUHostKVPool._pool_name: LRUHostKVPool#_pool_name.
  LRUHostKVPool._slot_spec: LRUHostKVPool#_slot_spec.
  StagedData.buffer_id: StagedData#buffer_id.
  QueueHostKVPool._per_layer_shape: QueueHostKVPool#_per_layer_shape.
  QueueHostKVPool._dtype: QueueHostKVPool#_dtype.
  LRUHostKVPool._mesh: LRUHostKVPool#_mesh.
  QueueHostKVPool.__init__: QueueHostKVPool#__init__().
  QueueHostKVPool._mesh: QueueHostKVPool#_mesh.
  QueueHostKVPool._partition_spec: QueueHostKVPool#_partition_spec.
  LRUHostKVPool.__init__: LRUHostKVPool#__init__().
  LRUHostKVPool._per_layer_shape: LRUHostKVPool#_per_layer_shape.
  LRUHostKVPool._dtype: LRUHostKVPool#_dtype.
  LRUHostKVPool._partition_spec: LRUHostKVPool#_partition_spec.
  make_unit_mesh: make_unit_mesh().
---
# Module: [`python/sgl_jax/srt/mem_cache/host_kv_pool.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py)

## Classes
### `HostKVPool`  ·  implements/extends ABC
- def: [`python/sgl_jax/srt/mem_cache/host_kv_pool.py:66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L66)
- doc: Backend-agnostic host-staging KV pool contract.
- signature: `class HostKVPool(abc.ABC):`
- members:
  - `available_size(self)` — [`L82`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L82) — Entries currently free.
  - `copy_from_device(self, layers: list[jax.Array], buffer_id: int)` — [`L93`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L93) — D2H staging for PD ``producer_handoff``: write ``layers`` into the
  - `copy_into(self, device_indices: list[int], host_buffer_ids: list[int])` — [`L99`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L99) — Retaining D2H: copy device page(s) into reserved slot(s), pairwise.
  - `copy_to_device(self, host_buffer_ids: list[int], device_indices: list[int])` — [`L105`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L105) — H2D: scatter slot(s) back into the device pool by index, pairwise.
  - `release(self, buffer_id: int)` — [`L78`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L78) — Return a reserved slot to the pool by id.
  - `reserve(self)` — [`L74`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L74) — Pop a free slot id, or ``None`` if exhausted.
  - `total_size(self)` — [`L86`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L86) — Total entries in the pool (free + in-use).
- uses (calls/refs, reference-scoped): [`copy_from_device`](host_kv_pool.md#QueueHostKVPool.copy_from_device), [`reserve`](host_kv_pool.md#LRUHostKVPool.reserve), [`reserve`](host_kv_pool.md#QueueHostKVPool.reserve), [`release`](host_kv_pool.md#LRUHostKVPool.release), [`available_size`](host_kv_pool.md#LRUHostKVPool.available_size), [`available_size`](host_kv_pool.md#QueueHostKVPool.available_size), [`copy_into`](host_kv_pool.md#LRUHostKVPool.copy_into), [`copy_to_device`](host_kv_pool.md#LRUHostKVPool.copy_to_device), [`QueueHostKVPool`](host_kv_pool.md#QueueHostKVPool), [`StagedData`](host_kv_pool.md#StagedData), [`LRUHostKVPool`](host_kv_pool.md#LRUHostKVPool), [`release`](host_kv_pool.md#QueueHostKVPool.release), [`total_size`](host_kv_pool.md#LRUHostKVPool.total_size), [`total_size`](host_kv_pool.md#QueueHostKVPool.total_size)
- used by: [`producer_handoff`](../disaggregation/jax_transfer/conn.md#JaxTransferKVManager.producer_handoff), [`__init__`](../disaggregation/jax_transfer/conn.md#JaxTransferKVManager.__init__), [`host_pool`](../disaggregation/jax_transfer/conn.md#JaxTransferKVManager.host_pool), [`QueueHostKVPool`](host_kv_pool.md#QueueHostKVPool), [`LRUHostKVPool`](host_kv_pool.md#LRUHostKVPool)

### `LRUHostKVPool`  ·  implements/extends HostKVPool
- def: [`python/sgl_jax/srt/mem_cache/host_kv_pool.py:266`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L266)
- doc: Retaining host pool for HiCache L2.
- signature: `class LRUHostKVPool(HostKVPool):`
- members:
  - `alloc(self, need_pages: int)` — [`L415`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L415) — Pop ``need_pages`` free page slots, returning their ids.
  - `available_size(self)` — [`L814`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L814) — Free page slots (HiCache control plane counts in pages).
  - `copy_into(self, device_indices: list[int], host_buffer_ids: list[int])` — [`L563`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L563)
  - `copy_to_device(self, host_buffer_ids: list[int], device_indices: list[int])` — [`L570`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L570)
  - `dec_lock_ref(self, buffer_id: int)` — [`L868`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L868)
  - `flush_backup(self, host_buffer_ids: list[int])` — [`L531`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L531) — D2H phase 2: ``device_put`` the staged pages into their host slots.
  - `flush_load(self, host_buffer_ids: list[int], device_indices: list[int])` — [`L620`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L620) — H2D phase 2: scatter the staged pages into the KV buffer via the
  - `free(self, host_page_ids)` — [`L449`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L449) — Release page slot(s) by id, deduplicated. Rejects locked
  - `inc_lock_ref(self, buffer_id: int)` — [`L863`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L863)
  - `precompile_transfers(self, max_pages: int | None = None)` — [`L751`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L751) — Warm the JIT/Pallas compile of the four transfer kernels for every
  - `release(self, buffer_id: int)` — [`L395`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L395)
  - `reserve(self)` — [`L368`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L368)
  - `stage_backup(self, device_indices: list[int], host_buffer_ids: list[int])` — [`L474`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L474) — D2H phase 1: gather live device pages into per-buffer staging arrays.
  - `stage_load(self, host_buffer_ids: list[int])` — [`L579`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L579) — H2D phase 1: ``device_put`` host slots onto the device into per-buffer
  - `total_size(self)` — [`L819`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L819) — Total page slots in the pool (free + in-use).
- protocol/private: `__init__`[`L287`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L287), `_allocated`[`L340`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L340), `_batched_layer_sharding`[`L320`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L320), `_device_packed_sharding`[`L317`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L317), `_device_pool`[`L302`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L302), `_drop_pending`[`L829`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L829), `_dtype`[`L307`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L307), `_exhaust_count`[`L349`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L349), `_free_ids`[`L335`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L335), `_generation`[`L347`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L347), `_host_sharding`[`L316`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L316), `_jit_gather_one_layer`[`L331`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L331), `_last_exhaust_log`[`L350`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L350), `_layer_num`[`L305`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L305), `_lock`[`L334`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L334), `_lock_ref`[`L337`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L337), `_mesh`[`L308`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L308), `_pad_to_page_bucket`[`L332`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L332), `_page_size`[`L304`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L304), `_partition_spec`[`L309`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L309), `_peak_used`[`L348`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L348), `_pending_gather`[`L356`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L356), `_pending_load`[`L362`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L362), `_pending_load_lock`[`L361`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L361), `_pending_lock`[`L355`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L355), `_per_layer_shape`[`L306`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L306), `_pool_name`[`L310`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L310), `_pool_size`[`L303`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L303), `_require_allocated`[`L838`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L838), `_require_device_pages`[`L847`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L847), `_slot_spec`[`L315`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L315), `_slots`[`L336`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L336)
- uses (calls/refs, reference-scoped): [`HostKVPool`](host_kv_pool.md#HostKVPool), [`_pad_to_page_bucket`](../disaggregation/prefill.md#_pad_to_page_bucket), [`logger`](host_kv_pool.md#logger), [`write_kv_layer`](memory_pool.md#write_kv_layer), [`_KV_GATHER_PAGE_BUCKETS`](../disaggregation/prefill.md#_KV_GATHER_PAGE_BUCKETS), [`_make_host_sharding`](host_kv_pool.md#_make_host_sharding), [`_jit_gather_one_layer`](../disaggregation/prefill.md#_jit_gather_one_layer)
- used by: [`HostKVPool`](host_kv_pool.md#HostKVPool), [`available_size`](host_kv_pool.md#HostKVPool.available_size), [`release`](host_kv_pool.md#HostKVPool.release), [`reserve`](host_kv_pool.md#HostKVPool.reserve), [`total_size`](host_kv_pool.md#HostKVPool.total_size), [`copy_into`](host_kv_pool.md#HostKVPool.copy_into), [`copy_to_device`](host_kv_pool.md#HostKVPool.copy_to_device)

### `QueueHostKVPool`  ·  implements/extends HostKVPool
- def: [`python/sgl_jax/srt/mem_cache/host_kv_pool.py:110`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L110)
- doc: FIFO one-shot pool for PD: reserve -&gt; copy_from_device -&gt; release.
- signature: `class QueueHostKVPool(HostKVPool):`
- members:
  - `available_size(self)` — [`L222`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L222)
  - `copy_from_device(self, layers: list[jax.Array], buffer_id: int)` — [`L186`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L186)
  - `release(self, buffer_id: int)` — [`L183`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L183)
  - `reserve(self)` — [`L160`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L160)
  - `total_size(self)` — [`L226`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L226)
- protocol/private: `__init__`[`L120`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L120), `_dtype`[`L140`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L140), `_exhaust_count`[`L153`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L153), `_free_ids`[`L147`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L147), `_host_sharding`[`L144`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L144), `_inc_alloc_metric`[`L231`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L231), `_last_exhaust_log`[`L154`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L154), `_layer_num`[`L138`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L138), `_lock`[`L146`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L146), `_max_padded_pages`[`L137`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L137), `_mesh`[`L141`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L141), `_partition_spec`[`L142`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L142), `_peak_used`[`L152`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L152), `_per_layer_shape`[`L139`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L139), `_pool_name`[`L143`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L143), `_pool_size`[`L136`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L136), `_record_d2h_bytes`[`L239`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L239), `_release`[`L249`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L249)
- uses (calls/refs, reference-scoped): [`HostKVPool`](host_kv_pool.md#HostKVPool), [`host_pool_alloc`](../disaggregation/common/metrics.md#host_pool_alloc), [`host_pool_free`](../disaggregation/common/metrics.md#host_pool_free), [`logger`](host_kv_pool.md#logger), [`PD_TRANSFER_BYTES_TOTAL`](../disaggregation/common/metrics.md#PD_TRANSFER_BYTES_TOTAL), [`_make_host_sharding`](host_kv_pool.md#_make_host_sharding), [`StagedData`](host_kv_pool.md#StagedData), [`array_pytree`](host_kv_pool.md#StagedData.array_pytree), [`buffer_id`](host_kv_pool.md#StagedData.buffer_id)
- used by: [`install_disaggregation_wiring`](../disaggregation/runtime.md#install_disaggregation_wiring), [`HostKVPool`](host_kv_pool.md#HostKVPool), [`copy_from_device`](host_kv_pool.md#HostKVPool.copy_from_device), [`available_size`](host_kv_pool.md#HostKVPool.available_size), [`release`](host_kv_pool.md#HostKVPool.release), [`reserve`](host_kv_pool.md#HostKVPool.reserve), [`total_size`](host_kv_pool.md#HostKVPool.total_size)

### `StagedData`
- def: [`python/sgl_jax/srt/mem_cache/host_kv_pool.py:57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L57)
- doc: Result of a D2H staging copy: a per-layer list of host arrays
- signature: `class StagedData:`
- members:
  - `array_pytree` — [`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L63)
  - `buffer_id` — [`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L62)
- used by: [`producer_handoff`](../disaggregation/jax_transfer/conn.md#JaxTransferKVManager.producer_handoff), [`copy_from_device`](host_kv_pool.md#QueueHostKVPool.copy_from_device), [`copy_from_device`](host_kv_pool.md#HostKVPool.copy_from_device)

## Functions
- `_make_host_sharding(mesh: Mesh, partition_spec: PartitionSpec)` — [`L37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L37) — Pinned-host sharding (falls back to default on platforms without it).
- `make_unit_mesh()` — [`L876`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L876) — Convenience for tests: a single-device mesh with axis ``x``.

## Module values
- `logger` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/mem_cache/host_kv_pool.py#L34)

