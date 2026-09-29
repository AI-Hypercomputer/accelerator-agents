---
title: 'Module: python/sgl_jax/srt/disaggregation/prefill.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/prefill.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.prefill`/
symbols:
  SchedulerDisaggregationPrefillMixin.process_prefill_chunk: SchedulerDisaggregationPrefillMixin#process_prefill_chunk().
  SchedulerDisaggregationPrefillMixin.event_loop_normal_disagg_prefill: SchedulerDisaggregationPrefillMixin#event_loop_normal_disagg_prefill().
  SchedulerDisaggregationPrefillMixin._extract_req_kv: SchedulerDisaggregationPrefillMixin#_extract_req_kv().
  SchedulerDisaggregationPrefillMixin._on_prefill_transfer_terminal: SchedulerDisaggregationPrefillMixin#_on_prefill_transfer_terminal().
  SchedulerDisaggregationPrefillMixin._finish_prefill_only_failure: SchedulerDisaggregationPrefillMixin#_finish_prefill_only_failure().
  SchedulerDisaggregationPrefillMixin._maybe_log_prefill_extract_debug: SchedulerDisaggregationPrefillMixin#_maybe_log_prefill_extract_debug().
  SchedulerDisaggregationPrefillMixin._pd_mark_time: SchedulerDisaggregationPrefillMixin#_pd_mark_time().
  SchedulerDisaggregationPrefillMixin._abort_prefill_req: SchedulerDisaggregationPrefillMixin#_abort_prefill_req().
  PrefillBootstrapQueue.drain_terminal: PrefillBootstrapQueue#drain_terminal().
  PrefillBootstrapQueue.add: PrefillBootstrapQueue#add().
  SchedulerDisaggregationPrefillMixin._finish_prefill_only_success: SchedulerDisaggregationPrefillMixin#_finish_prefill_only_success().
  SchedulerDisaggregationPrefillMixin.send_kv_chunk: SchedulerDisaggregationPrefillMixin#send_kv_chunk().
  SchedulerDisaggregationPrefillMixin._stream_prefill_req: SchedulerDisaggregationPrefillMixin#_stream_prefill_req().
  SchedulerDisaggregationPrefillMixin._release_prefill_req_resources: SchedulerDisaggregationPrefillMixin#_release_prefill_req_resources().
  SchedulerDisaggregationPrefillMixin._release_prefill_kv_pool: SchedulerDisaggregationPrefillMixin#_release_prefill_kv_pool().
  SchedulerDisaggregationPrefillMixin.disagg_kv_manager: SchedulerDisaggregationPrefillMixin#disagg_kv_manager.
  PrefillBootstrapQueue._entries: PrefillBootstrapQueue#_entries.
  SchedulerDisaggregationPrefillMixin.disagg_prefill_queue: SchedulerDisaggregationPrefillMixin#disagg_prefill_queue.
  _pad_to_page_bucket: _pad_to_page_bucket().
  PrefillBootstrapQueue.abort_matching: PrefillBootstrapQueue#abort_matching().
  SchedulerDisaggregationPrefillMixin._release_prefill_host_buffer: SchedulerDisaggregationPrefillMixin#_release_prefill_host_buffer().
  _KV_GATHER_PAGE_BUCKETS: _KV_GATHER_PAGE_BUCKETS.
  PrefillBookkeeping: PrefillBookkeeping#
  SchedulerDisaggregationPrefillMixin.disagg_use_d2h_staging: SchedulerDisaggregationPrefillMixin#disagg_use_d2h_staging.
  _jit_gather_all_layers: _jit_gather_all_layers().
  PrefillBookkeeping.sender: PrefillBookkeeping#sender.
  PrefillBootstrapQueue.__len__: PrefillBootstrapQueue#__len__().
  SchedulerDisaggregationPrefillMixin: SchedulerDisaggregationPrefillMixin#
  logger: logger.
  PrefillBookkeeping.req_id: PrefillBookkeeping#req_id.
  PrefillBookkeeping.on_terminal: PrefillBookkeeping#on_terminal.
  PrefillBootstrapQueue._lock: PrefillBootstrapQueue#_lock.
  SchedulerDisaggregationPrefillMixin._on_terminal: SchedulerDisaggregationPrefillMixin#_on_terminal().
  SchedulerDisaggregationPrefillMixin._record_prefill_transfer_failure: SchedulerDisaggregationPrefillMixin#_record_prefill_transfer_failure().
  _jit_gather_one_layer: _jit_gather_one_layer().
  PrefillBootstrapQueue: PrefillBootstrapQueue#
  local_kv_spec_for_pool: local_kv_spec_for_pool().
  _global_to_local_shard: _global_to_local_shard().
  PrefillBootstrapQueue.__init__: PrefillBootstrapQueue#__init__().
---
# Module: [`python/sgl_jax/srt/disaggregation/prefill.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py)

## Classes
### `PrefillBookkeeping`
- def: [`python/sgl_jax/srt/disaggregation/prefill.py:113`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L113)
- doc: Per-request prefill-side state tracked by the Mixin.
- signature: `class PrefillBookkeeping:`
- members:
  - `on_terminal` — [`L121`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L121)
  - `req_id` — [`L116`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L116)
  - `sender` — [`L117`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L117)
- uses (calls/refs, reference-scoped): [`JaxTransferKVSender`](jax_transfer/conn.md#JaxTransferKVSender)
- used by: [`abort_request`](../managers/scheduler.md#Scheduler.abort_request), [`drain_terminal`](prefill.md#PrefillBootstrapQueue.drain_terminal), [`add`](prefill.md#PrefillBootstrapQueue.add), [`send_kv_chunk`](prefill.md#SchedulerDisaggregationPrefillMixin.send_kv_chunk), [`_entries`](prefill.md#PrefillBootstrapQueue._entries), [`abort_matching`](prefill.md#PrefillBootstrapQueue.abort_matching)

### `PrefillBootstrapQueue`
- def: [`python/sgl_jax/srt/disaggregation/prefill.py:124`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L124)
- doc: Tracks senders pending decoder ack. Thread-safe.
- signature: `class PrefillBootstrapQueue:`
- members:
  - `abort_matching(self, rid_prefix: str, abort_all: bool)` — [`L160`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L160)
  - `add(self, req_id: str, sender: JaxTransferKVSender, on_terminal=None)` — [`L135`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L135)
  - `drain_terminal(self)` — [`L148`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L148) — Remove and return entries that reached SUCCESS or FAILED.
- protocol/private: `__init__`[`L127`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L127), `__len__`[`L131`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L131), `_entries`[`L129`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L129), `_lock`[`L128`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L128)
- uses (calls/refs, reference-scoped): [`KVPoll`](base/kv_manager.md#KVPoll), [`FAILED`](base/kv_manager.md#KVPoll.FAILED), [`SUCCESS`](base/kv_manager.md#KVPoll.SUCCESS), [`JaxTransferKVSender`](jax_transfer/conn.md#JaxTransferKVSender), [`poll`](jax_transfer/conn.md#JaxTransferKVSender.poll), [`PrefillBookkeeping`](prefill.md#PrefillBookkeeping), [`sender`](prefill.md#PrefillBookkeeping.sender), [`on_terminal`](prefill.md#PrefillBookkeeping.on_terminal), [`req_id`](prefill.md#PrefillBookkeeping.req_id)
- used by: [`install_disaggregation_wiring`](runtime.md#install_disaggregation_wiring), [`abort_request`](../managers/scheduler.md#Scheduler.abort_request), [`process_prefill_chunk`](prefill.md#SchedulerDisaggregationPrefillMixin.process_prefill_chunk), [`send_kv_chunk`](prefill.md#SchedulerDisaggregationPrefillMixin.send_kv_chunk), [`disagg_prefill_queue`](prefill.md#SchedulerDisaggregationPrefillMixin.disagg_prefill_queue)

### `SchedulerDisaggregationPrefillMixin`
- def: [`python/sgl_jax/srt/disaggregation/prefill.py:169`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L169)
- doc: Mixin for PD prefill mode on Scheduler.
- signature: `class SchedulerDisaggregationPrefillMixin:`
- members:
  - `_extract_req_kv(self: Scheduler, req: Req)` — [`L362`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L362) — Gather prefilled KV from the paged pool for ``req``.
  - `_pd_mark_time(self: Scheduler, req: Req, name: str)` — [`L348`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L348) — Record a PD lifecycle mark on ``req`` (no-op unless enabled).
  - `_release_prefill_kv_pool(self: Scheduler, req: Req)` — [`L416`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L416) — Release the prefill device KV cache + request-pool slot.
  - `_release_prefill_req_resources(self: Scheduler, req: Req)` — [`L428`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L428) — Release prefill-side KV and request-pool resources.
  - `event_loop_normal_disagg_prefill(self: Scheduler)` — [`L176`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L176) — Prefill-only event loop.
  - `process_prefill_chunk(self: Scheduler, batch, result)` — [`L213`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L213) — Extract KV for PD reqs and hand off to sender. — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)
  - `send_kv_chunk(self: Scheduler)` — [`L328`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L328) — Reap senders that reached SUCCESS / FAILED.
  - `disagg_kv_manager` — [`L172`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L172)
  - `disagg_prefill_queue` — [`L173`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L173)
  - `disagg_use_d2h_staging` — [`L174`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L174)
- protocol/private: `_abort_prefill_req`[`L460`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L460), `_finish_prefill_only_failure`[`L512`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L512), `_finish_prefill_only_success`[`L504`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L504), `_maybe_log_prefill_extract_debug`[`L533`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L533), `_on_prefill_transfer_terminal`[`L479`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L479), `_on_terminal`[`L323`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L323), `_record_prefill_transfer_failure`[`L445`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L445), `_release_prefill_host_buffer`[`L434`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L434), `_stream_prefill_req`[`L453`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L453)
- uses (calls/refs, reference-scoped): `Req`, [`run_batch`](../managers/scheduler.md#Scheduler.run_batch), [`KVPoll`](base/kv_manager.md#KVPoll), [`get_next_batch_to_run`](../managers/scheduler.md#Scheduler.get_next_batch_to_run), [`Scheduler`](../managers/scheduler.md#Scheduler), `output_ids`, [`tree_cache`](../managers/scheduler.md#Scheduler.tree_cache), [`send`](jax_transfer/conn.md#JaxTransferKVSender.send), `origin_input_ids`, `rid`, [`process_batch_result`](../managers/scheduler.md#Scheduler.process_batch_result), [`token_to_kv_pool_allocator`](../managers/scheduler.md#Scheduler.token_to_kv_pool_allocator), `req_pool_idx`, [`_comm_backend`](../managers/scheduler.md#Scheduler._comm_backend), [`build_kv_debug_snapshot`](debug_utils.md#build_kv_debug_snapshot), [`select_dp_for_request`](../managers/scheduler.md#Scheduler.select_dp_for_request), [`last_batch`](../managers/scheduler.md#Scheduler.last_batch), [`req_to_token`](../mem_cache/memory_pool.md#ReqToTokenPool.req_to_token), [`layer_num`](../mem_cache/memory_pool.md#KVCache.layer_num), [`drain_terminal`](prefill.md#PrefillBootstrapQueue.drain_terminal), [`req_to_token_pool`](../managers/scheduler.md#Scheduler.req_to_token_pool), [`page_size`](../mem_cache/memory_pool.md#KVCache.page_size), `finished_reason`, [`recv_requests`](../managers/scheduler.md#Scheduler.recv_requests), [`release_kv_cache`](../mem_cache/common.md#release_kv_cache), [`server_args`](../managers/scheduler.md#Scheduler.server_args), [`add`](prefill.md#PrefillBootstrapQueue.add), [`failure_exception`](jax_transfer/conn.md#JaxTransferKVSender.failure_exception), [`start_layer`](../mem_cache/memory_pool.md#KVCache.start_layer), [`set_next_batch_sampling_info_done`](../managers/scheduler.md#Scheduler.set_next_batch_sampling_info_done), [`init_new_token_ratio`](../managers/scheduler.md#Scheduler.init_new_token_ratio), [`init`](jax_transfer/conn.md#JaxTransferKVSender.init), [`kv_debug_enabled`](debug_utils.md#kv_debug_enabled), [`new_token_ratio`](../managers/scheduler.md#Scheduler.new_token_ratio), `return_logprob`, [`cur_batch`](../managers/scheduler.md#Scheduler.cur_batch), [`get_kvcache`](../mem_cache/allocator.md#BaseTokenToKVPoolAllocator.get_kvcache), [`SUCCESS`](base/kv_manager.md#KVPoll.SUCCESS), [`mesh`](../mem_cache/memory_pool.md#KVCache.mesh), [`process_input_requests`](../managers/scheduler.md#Scheduler.process_input_requests)  (+35 more)
- used by: [`install_disaggregation_wiring`](runtime.md#install_disaggregation_wiring), [`_admit_decode_prealloc`](decode.md#SchedulerDisaggregationDecodeMixin._admit_decode_prealloc), [`abort_request`](../managers/scheduler.md#Scheduler.abort_request), [`get_internal_state`](../managers/scheduler.md#Scheduler.get_internal_state), [`process_decode_queue`](decode.md#SchedulerDisaggregationDecodeMixin.process_decode_queue), [`process_input_requests_disagg_decode`](decode.md#SchedulerDisaggregationDecodeMixin.process_input_requests_disagg_decode), [`Scheduler`](../managers/scheduler.md#Scheduler), [`spec_algorithm`](../managers/scheduler.md#Scheduler.spec_algorithm), [`_can_flush_cache`](../managers/scheduler.md#Scheduler._can_flush_cache), [`_shutdown`](runtime.md#_make_disagg_shutdown._shutdown), [`dispatch_scheduler_event_loop`](../managers/scheduler.md#dispatch_scheduler_event_loop), [`_decode_backlog_snapshot`](decode.md#SchedulerDisaggregationDecodeMixin._decode_backlog_snapshot), [`disagg_shutdown`](../managers/scheduler.md#Scheduler.disagg_shutdown)

## Functions
- `_global_to_local_shard(arr: jax.Array)` — [`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L62) — View this host's addressable shards of a globally-sharded ``arr`` as a
- `_jit_gather_all_layers(buffers, page_indices, out_sharding)` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L54) — Gather ``page_indices`` from every per-layer KV buffer.
- `_jit_gather_one_layer(buf, page_indices, out_sharding)` — [`L45`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L45) — Gather ``page_indices`` from a single per-layer KV buffer.
- `_pad_to_page_bucket(num_pages: int)` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L34)
- `local_kv_spec_for_pool(kv_pool, layer_num: int, padded_pages: int)` — [`L87`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L87) — Build the ShapeDtypeStruct that D should pull on a multi-host process:

## Module values
- `_KV_GATHER_PAGE_BUCKETS` — [`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L31)
- `logger` — [`L26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/prefill.py#L26)

