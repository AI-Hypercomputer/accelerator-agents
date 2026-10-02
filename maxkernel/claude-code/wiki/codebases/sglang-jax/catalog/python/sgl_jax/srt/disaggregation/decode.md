---
title: 'Module: python/sgl_jax/srt/disaggregation/decode.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/decode.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.decode`/
symbols:
  SchedulerDisaggregationDecodeMixin._admit_decode_prealloc: SchedulerDisaggregationDecodeMixin#_admit_decode_prealloc().
  SchedulerDisaggregationDecodeMixin.process_input_requests_disagg_decode: SchedulerDisaggregationDecodeMixin#process_input_requests_disagg_decode().
  SchedulerDisaggregationDecodeMixin.process_decode_queue: SchedulerDisaggregationDecodeMixin#process_decode_queue().
  SchedulerDisaggregationDecodeMixin._write_kv_to_pool: SchedulerDisaggregationDecodeMixin#_write_kv_to_pool().
  SchedulerDisaggregationDecodeMixin.event_loop_normal_disagg_decode: SchedulerDisaggregationDecodeMixin#event_loop_normal_disagg_decode().
  SchedulerDisaggregationDecodeMixin._drain_transfer_queue_synced: SchedulerDisaggregationDecodeMixin#_drain_transfer_queue_synced().
  SchedulerDisaggregationDecodeMixin._abort_decode_request: SchedulerDisaggregationDecodeMixin#_abort_decode_request().
  SchedulerDisaggregationDecodeMixin._build_kv_spec_for_req: SchedulerDisaggregationDecodeMixin#_build_kv_spec_for_req().
  SchedulerDisaggregationDecodeMixin._maybe_log_decode_pull_debug: SchedulerDisaggregationDecodeMixin#_maybe_log_decode_pull_debug().
  SchedulerDisaggregationDecodeMixin._maybe_verify_decode_writeback_debug: SchedulerDisaggregationDecodeMixin#_maybe_verify_decode_writeback_debug().
  DecodeBookkeeping.req: DecodeBookkeeping#req.
  DecodeTransferQueue.drain_terminal: DecodeTransferQueue#drain_terminal().
  SchedulerDisaggregationDecodeMixin._decode_backlog_snapshot: SchedulerDisaggregationDecodeMixin#_decode_backlog_snapshot().
  SchedulerDisaggregationDecodeMixin._release_decode_req_resources: SchedulerDisaggregationDecodeMixin#_release_decode_req_resources().
  SchedulerDisaggregationDecodeMixin.disagg_transfer_queue: SchedulerDisaggregationDecodeMixin#disagg_transfer_queue.
  DecodeBookkeeping.req_id: DecodeBookkeeping#req_id.
  SchedulerDisaggregationDecodeMixin._pd_mark_time: SchedulerDisaggregationDecodeMixin#_pd_mark_time().
  SchedulerDisaggregationDecodeMixin._release_decode_kv_indices: SchedulerDisaggregationDecodeMixin#_release_decode_kv_indices().
  DecodeBookkeeping: DecodeBookkeeping#
  DecodeBookkeeping.receiver: DecodeBookkeeping#receiver.
  DecodeTransferQueue._entries: DecodeTransferQueue#_entries.
  SchedulerDisaggregationDecodeMixin._extract_pd_reqs_from_waiting_queue: SchedulerDisaggregationDecodeMixin#_extract_pd_reqs_from_waiting_queue().
  SchedulerDisaggregationDecodeMixin.disagg_prealloc_queue: SchedulerDisaggregationDecodeMixin#disagg_prealloc_queue.
  logger: logger.
  DecodePreallocQueue.add: DecodePreallocQueue#add().
  DecodeTransferQueue.add: DecodeTransferQueue#add().
  DecodeBookkeeping.kv_indices: DecodeBookkeeping#kv_indices.
  DecodePreallocQueue._entries: DecodePreallocQueue#_entries.
  SchedulerDisaggregationDecodeMixin.disagg_bootstrap_client: SchedulerDisaggregationDecodeMixin#disagg_bootstrap_client.
  DecodePreallocQueue.items_fifo: DecodePreallocQueue#items_fifo().
  DecodePreallocQueue.abort_matching: DecodePreallocQueue#abort_matching().
  DecodeTransferQueue.abort_matching: DecodeTransferQueue#abort_matching().
  SchedulerDisaggregationDecodeMixin._pick_prefill_peer_for_this_host: SchedulerDisaggregationDecodeMixin#_pick_prefill_peer_for_this_host().
  SchedulerDisaggregationDecodeMixin._enqueue_for_decode: SchedulerDisaggregationDecodeMixin#_enqueue_for_decode().
  DecodePreallocQueue.remove: DecodePreallocQueue#remove().
  SchedulerDisaggregationDecodeMixin._record_decode_transfer_failure: SchedulerDisaggregationDecodeMixin#_record_decode_transfer_failure().
  DecodeTransferQueue._lock: DecodeTransferQueue#_lock.
  DecodeBookkeeping.synced_state: DecodeBookkeeping#synced_state.
  SchedulerDisaggregationDecodeMixin.disagg_prefill_info_cache: SchedulerDisaggregationDecodeMixin#disagg_prefill_info_cache.
  DecodePreallocQueue._lock: DecodePreallocQueue#_lock.
  DecodePreallocQueue.__len__: DecodePreallocQueue#__len__().
  DecodeTransferQueue.__len__: DecodeTransferQueue#__len__().
  SchedulerDisaggregationDecodeMixin: SchedulerDisaggregationDecodeMixin#
  SchedulerDisaggregationDecodeMixin._record_decode_transfer_bytes: SchedulerDisaggregationDecodeMixin#_record_decode_transfer_bytes().
  DecodePreallocQueue: DecodePreallocQueue#
  DecodeTransferQueue: DecodeTransferQueue#
  SchedulerDisaggregationDecodeMixin.disagg_kv_manager: SchedulerDisaggregationDecodeMixin#disagg_kv_manager.
  DecodeBookkeeping.p_info: DecodeBookkeeping#p_info.
  DecodeBookkeeping.started: DecodeBookkeeping#started.
  DecodePreallocQueue.__init__: DecodePreallocQueue#__init__().
  DecodeTransferQueue.__init__: DecodeTransferQueue#__init__().
---
# Module: [`python/sgl_jax/srt/disaggregation/decode.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py)

## Classes
### `DecodeBookkeeping`
- def: [`python/sgl_jax/srt/disaggregation/decode.py:33`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L33)
- doc: Per-request decode-side state.
- signature: `class DecodeBookkeeping:`
- members:
  - `kv_indices` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L40)
  - `p_info` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L48)
  - `receiver` — [`L38`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L38)
  - `req` — [`L37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L37)
  - `req_id` — [`L36`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L36)
  - `started` — [`L42`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L42)
  - `synced_state` — [`L45`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L45)
- uses (calls/refs, reference-scoped): `Req`, [`KVPoll`](base/kv_manager.md#KVPoll), [`JaxTransferKVReceiver`](jax_transfer/conn.md#JaxTransferKVReceiver)
- used by: [`_admit_decode_prealloc`](decode.md#SchedulerDisaggregationDecodeMixin._admit_decode_prealloc), [`abort_request`](../managers/scheduler.md#Scheduler.abort_request), [`process_decode_queue`](decode.md#SchedulerDisaggregationDecodeMixin.process_decode_queue), [`process_input_requests_disagg_decode`](decode.md#SchedulerDisaggregationDecodeMixin.process_input_requests_disagg_decode), [`_drain_transfer_queue_synced`](decode.md#SchedulerDisaggregationDecodeMixin._drain_transfer_queue_synced), [`drain_terminal`](decode.md#DecodeTransferQueue.drain_terminal), [`_entries`](decode.md#DecodeTransferQueue._entries), [`add`](decode.md#DecodePreallocQueue.add), [`add`](decode.md#DecodeTransferQueue.add), [`_entries`](decode.md#DecodePreallocQueue._entries), [`abort_matching`](decode.md#DecodePreallocQueue.abort_matching), [`abort_matching`](decode.md#DecodeTransferQueue.abort_matching), [`items_fifo`](decode.md#DecodePreallocQueue.items_fifo)

### `DecodePreallocQueue`
- def: [`python/sgl_jax/srt/disaggregation/decode.py:51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L51)
- doc: PD reqs awaiting capacity-gated KV alloc. FIFO, thread-safe.
- signature: `class DecodePreallocQueue:`
- members:
  - `abort_matching(self, rid_prefix: str, abort_all: bool)` — [`L86`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L86)
  - `add(self, entry: DecodeBookkeeping)` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L68)
  - `items_fifo(self)` — [`L74`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L74) — FIFO snapshot for the admission gate (does not remove).
  - `remove(self, req_id: str)` — [`L80`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L80) — Drop an admitted (or failed) entry by id.
- protocol/private: `__init__`[`L60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L60), `__len__`[`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L64), `_entries`[`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L62), `_lock`[`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L61)
- uses (calls/refs, reference-scoped): [`req_id`](decode.md#DecodeBookkeeping.req_id), [`DecodeBookkeeping`](decode.md#DecodeBookkeeping)
- used by: [`install_disaggregation_wiring`](runtime.md#install_disaggregation_wiring), [`_admit_decode_prealloc`](decode.md#SchedulerDisaggregationDecodeMixin._admit_decode_prealloc), [`abort_request`](../managers/scheduler.md#Scheduler.abort_request), [`process_input_requests_disagg_decode`](decode.md#SchedulerDisaggregationDecodeMixin.process_input_requests_disagg_decode), [`disagg_prealloc_queue`](decode.md#SchedulerDisaggregationDecodeMixin.disagg_prealloc_queue)

### `DecodeTransferQueue`
- def: [`python/sgl_jax/srt/disaggregation/decode.py:95`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L95)
- doc: Receivers in TRANSFERRING; polled each tick.
- signature: `class DecodeTransferQueue:`
- members:
  - `abort_matching(self, rid_prefix: str, abort_all: bool)` — [`L125`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L125)
  - `add(self, entry: DecodeBookkeeping)` — [`L106`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L106)
  - `drain_terminal(self)` — [`L112`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L112) — Return entries whose receiver reached SUCCESS or FAILED. — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)
- protocol/private: `__init__`[`L98`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L98), `__len__`[`L102`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L102), `_entries`[`L100`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L100), `_lock`[`L99`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L99)
- uses (calls/refs, reference-scoped): [`KVPoll`](base/kv_manager.md#KVPoll), [`poll`](jax_transfer/conn.md#JaxTransferKVReceiver.poll), [`FAILED`](base/kv_manager.md#KVPoll.FAILED), [`req_id`](decode.md#DecodeBookkeeping.req_id), [`DecodeBookkeeping`](decode.md#DecodeBookkeeping), [`SUCCESS`](base/kv_manager.md#KVPoll.SUCCESS), [`receiver`](decode.md#DecodeBookkeeping.receiver)
- used by: [`install_disaggregation_wiring`](runtime.md#install_disaggregation_wiring), [`_admit_decode_prealloc`](decode.md#SchedulerDisaggregationDecodeMixin._admit_decode_prealloc), [`abort_request`](../managers/scheduler.md#Scheduler.abort_request), [`_drain_transfer_queue_synced`](decode.md#SchedulerDisaggregationDecodeMixin._drain_transfer_queue_synced), [`disagg_transfer_queue`](decode.md#SchedulerDisaggregationDecodeMixin.disagg_transfer_queue)

### `SchedulerDisaggregationDecodeMixin`
- def: [`python/sgl_jax/srt/disaggregation/decode.py:134`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L134)
- doc: Mixin for PD decode mode on Scheduler.
- signature: `class SchedulerDisaggregationDecodeMixin:`
- members:
  - `_abort_decode_request(self: Scheduler, req: Req, reason: str)` — [`L665`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L665) — Release resources AND send AbortReq back to tokenizer.
  - `_admit_decode_prealloc(self: Scheduler)` — [`L430`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L430) — Capacity-gated FIFO admission of preallocated PD reqs. — documented in [python-sgl_jax-srt-mem_cache-allocator](../../../../../concepts/python-sgl_jax-srt-mem_cache-allocator.md)
  - `_build_kv_spec_for_req(self: Scheduler, req: Req)` — [`L537`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L537) — Build per-layer ShapeDtypeStructs matching P's KV layout.
  - `_decode_backlog_snapshot(self: Scheduler)` — [`L184`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L184) — One-line backlog snapshot for the watchdog stall report.
  - `_drain_transfer_queue_synced(self: Scheduler)` — [`L394`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L394) — On multi-host, only drain entries whose receiver has reached a
  - `_enqueue_for_decode(self: Scheduler, req: Req)` — [`L643`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L643) — Put ``req`` into the scheduler's decode-ready queue.
  - `_extract_pd_reqs_from_waiting_queue(self: Scheduler, rids: set)` — [`L298`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L298) — Extract PD reqs from waiting_queue by rid set.
  - `_pd_mark_time(self: Scheduler, req: Req, name: str)` — [`L511`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L511) — Record a PD lifecycle mark on ``req`` (no-op unless enabled).
  - `_pick_prefill_peer_for_this_host(self: Scheduler)` — [`L370`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L370) — Multi-host: find the P host whose jax_process_index matches ours.
  - `_release_decode_kv_indices(self: Scheduler, kv_indices)` — [`L525`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L525) — Release KV indices back to the allocator.
  - `_release_decode_req_resources(self: Scheduler, req: Req)` — [`L649`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L649) — Best-effort release of req_to_token_pool slot. Does NOT
  - `_write_kv_to_pool(self: Scheduler, req: Req, kv_indices, kv: jax.Array)` — [`L559`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L559) — Write pulled KV into the local paged pool (in place).
  - `event_loop_normal_disagg_decode(self: Scheduler)` — [`L143`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L143) — Decode event loop.
  - `process_decode_queue(self: Scheduler)` — [`L313`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L313) — Drive prealloc -> transfer -> ready transitions. — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)
  - `process_input_requests_disagg_decode(self: Scheduler, recv_reqs)` — [`L208`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L208) — Decode-mode request intake. PD reqs are extracted from
  - `disagg_bootstrap_client` — [`L138`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L138)
  - `disagg_kv_manager` — [`L137`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L137)
  - `disagg_prealloc_queue` — [`L140`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L140)
  - `disagg_prefill_info_cache` — [`L139`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L139)
  - `disagg_transfer_queue` — [`L141`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L141)
- protocol/private: `_maybe_log_decode_pull_debug`[`L704`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L704), `_maybe_verify_decode_writeback_debug`[`L725`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L725), `_record_decode_transfer_bytes`[`L692`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L692), `_record_decode_transfer_failure`[`L684`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L684)
- uses (calls/refs, reference-scoped): `Req`, [`run_batch`](../managers/scheduler.md#Scheduler.run_batch), [`KVPoll`](base/kv_manager.md#KVPoll), [`running_batch`](../managers/scheduler.md#Scheduler.running_batch), [`poll`](jax_transfer/conn.md#JaxTransferKVReceiver.poll), [`get_next_batch_to_run`](../managers/scheduler.md#Scheduler.get_next_batch_to_run), [`Scheduler`](../managers/scheduler.md#Scheduler), `output_ids`, `origin_input_ids`, `prefix_indices`, `rid`, [`fail`](jax_transfer/conn.md#JaxTransferKVReceiver.fail), [`page_size`](../mem_cache/allocator.md#BaseTokenToKVPoolAllocator.page_size), [`process_batch_result`](../managers/scheduler.md#Scheduler.process_batch_result), [`token_to_kv_pool_allocator`](../managers/scheduler.md#Scheduler.token_to_kv_pool_allocator), `fill_ids`, `req_pool_idx`, [`_comm_backend`](../managers/scheduler.md#Scheduler._comm_backend), [`build_kv_debug_snapshot`](debug_utils.md#build_kv_debug_snapshot), [`select_dp_for_request`](../managers/scheduler.md#Scheduler.select_dp_for_request), [`available_size`](../mem_cache/allocator.md#BaseTokenToKVPoolAllocator.available_size), [`pick_for_room`](bootstrap.md#PrefillInfoCache.pick_for_room), [`last_batch`](../managers/scheduler.md#Scheduler.last_batch), [`FAILED`](base/kv_manager.md#KVPoll.FAILED), [`free`](../mem_cache/allocator.md#BaseTokenToKVPoolAllocator.free), [`_pd_mark_time`](prefill.md#SchedulerDisaggregationPrefillMixin._pd_mark_time), [`layer_num`](../mem_cache/memory_pool.md#KVCache.layer_num), [`req`](decode.md#DecodeBookkeeping.req), [`drain_terminal`](decode.md#DecodeTransferQueue.drain_terminal), [`req_to_token_pool`](../managers/scheduler.md#Scheduler.req_to_token_pool), [`page_size`](../mem_cache/memory_pool.md#KVCache.page_size), [`waiting_queue`](../managers/scheduler.md#Scheduler.waiting_queue), [`dtype`](../mem_cache/memory_pool.md#KVCache.dtype), [`recv_requests`](../managers/scheduler.md#Scheduler.recv_requests), [`rid`](../managers/io_struct.md#BaseReq.rid), [`server_args`](../managers/scheduler.md#Scheduler.server_args), [`start_layer`](../mem_cache/memory_pool.md#KVCache.start_layer), `last_matched_prefix_len`, [`init_new_token_ratio`](../managers/scheduler.md#Scheduler.init_new_token_ratio), [`kv_debug_enabled`](debug_utils.md#kv_debug_enabled)  (+73 more)
- used by: [`install_disaggregation_wiring`](runtime.md#install_disaggregation_wiring), [`abort_request`](../managers/scheduler.md#Scheduler.abort_request), [`get_internal_state`](../managers/scheduler.md#Scheduler.get_internal_state), [`Scheduler`](../managers/scheduler.md#Scheduler), [`spec_algorithm`](../managers/scheduler.md#Scheduler.spec_algorithm), [`_can_flush_cache`](../managers/scheduler.md#Scheduler._can_flush_cache), [`_shutdown`](runtime.md#_make_disagg_shutdown._shutdown), [`dispatch_scheduler_event_loop`](../managers/scheduler.md#dispatch_scheduler_event_loop), [`disagg_shutdown`](../managers/scheduler.md#Scheduler.disagg_shutdown), [`disagg_decode_watchdog`](../managers/scheduler.md#Scheduler.disagg_decode_watchdog)

## Module values
- `logger` — [`L29`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode.py#L29)

