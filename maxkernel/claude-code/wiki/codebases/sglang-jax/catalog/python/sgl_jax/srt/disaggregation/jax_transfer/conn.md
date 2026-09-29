---
title: 'Module: python/sgl_jax/srt/disaggregation/jax_transfer/conn.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/jax_transfer/conn.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.jax_transfer.conn`/
symbols:
  JaxTransferKVReceiver.poll: JaxTransferKVReceiver#poll().
  JaxTransferKVSender.fail: JaxTransferKVSender#fail().
  JaxTransferKVSender._on_ack: JaxTransferKVSender#_on_ack().
  JaxTransferKVSender.send: JaxTransferKVSender#send().
  JaxTransferKVReceiver._run_pull: JaxTransferKVReceiver#_run_pull().
  JaxTransferKVReceiver.fail: JaxTransferKVReceiver#fail().
  JaxTransferKVManager.producer_handoff: JaxTransferKVManager#producer_handoff().
  JaxTransferKVSender.failure_exception: JaxTransferKVSender#failure_exception().
  JaxTransferKVReceiver._metadata: JaxTransferKVReceiver#_metadata.
  JaxTransferKVReceiver.failure_exception: JaxTransferKVReceiver#failure_exception().
  JaxTransferKVSender._mgr: JaxTransferKVSender#_mgr.
  JaxTransferKVSender._req_id: JaxTransferKVSender#_req_id.
  JaxTransferKVSender.init: JaxTransferKVSender#init().
  JaxTransferKVReceiver._req_id: JaxTransferKVReceiver#_req_id.
  JaxTransferKVReceiver.init: JaxTransferKVReceiver#init().
  JaxTransferKVManager.zmq_notifier: JaxTransferKVManager#zmq_notifier().
  JaxTransferKVReceiver._mgr: JaxTransferKVReceiver#_mgr.
  JaxTransferKVSender.__init__: JaxTransferKVSender#__init__().
  JaxTransferKVReceiver.__init__: JaxTransferKVReceiver#__init__().
  JaxTransferKVSender.uuid: JaxTransferKVSender#uuid().
  JaxTransferKVSender: JaxTransferKVSender#
  JaxTransferKVSender.attach_payload: JaxTransferKVSender#attach_payload().
  JaxTransferKVReceiver: JaxTransferKVReceiver#
  JaxTransferKVManager: JaxTransferKVManager#
  JaxTransferKVManager.__init__: JaxTransferKVManager#__init__().
  JaxTransferKVSender._status: JaxTransferKVSender#_status.
  JaxTransferKVSender.poll: JaxTransferKVSender#poll().
  JaxTransferKVSender.clear: JaxTransferKVSender#clear().
  JaxTransferKVManager._pull_workers: JaxTransferKVManager#_pull_workers.
  JaxTransferKVManager._pull_worker_loop: JaxTransferKVManager#_pull_worker_loop().
  JaxTransferKVManager.wrapper: JaxTransferKVManager#wrapper().
  JaxTransferKVReceiver.clear: JaxTransferKVReceiver#clear().
  PMetadata.uuid: PMetadata#uuid.
  JaxTransferKVReceiver._close_pull_timer: JaxTransferKVReceiver#_close_pull_timer().
  JaxTransferKVManager.enqueue_pull: JaxTransferKVManager#enqueue_pull().
  JaxTransferKVManager.create_sender: JaxTransferKVManager#create_sender().
  JaxTransferKVManager.create_receiver: JaxTransferKVManager#create_receiver().
  JaxTransferKVReceiver._transfer_started_at: JaxTransferKVReceiver#_transfer_started_at.
  JaxTransferKVSender._close_ack_timer: JaxTransferKVSender#_close_ack_timer().
  PMetadata: PMetadata#
  JaxTransferKVManager._wrapper: JaxTransferKVManager#_wrapper.
  JaxTransferKVManager.host_pool: JaxTransferKVManager#host_pool().
  JaxTransferKVSender._payload: JaxTransferKVSender#_payload.
  JaxTransferKVSender._state_lock: JaxTransferKVSender#_state_lock.
  JaxTransferKVSender._transfer_started_at: JaxTransferKVSender#_transfer_started_at.
  JaxTransferKVReceiver._state_lock: JaxTransferKVReceiver#_state_lock.
  JaxTransferKVManager._pull_queue: JaxTransferKVManager#_pull_queue.
  JaxTransferKVSender.abort: JaxTransferKVSender#abort().
  JaxTransferKVReceiver.abort: JaxTransferKVReceiver#abort().
  TransferStatus: TransferStatus#
  TransferStatus.sub_uuids: TransferStatus#sub_uuids.
  TransferStatus.on_done: TransferStatus#on_done.
  JaxTransferKVSender._use_d2h_staging: JaxTransferKVSender#_use_d2h_staging.
  JaxTransferKVSender._ack_timer: JaxTransferKVSender#_ack_timer.
  JaxTransferKVReceiver._results: JaxTransferKVReceiver#_results.
  JaxTransferKVReceiver._pull_timer: JaxTransferKVReceiver#_pull_timer.
  JaxTransferKVReceiver.result: JaxTransferKVReceiver#result().
  JaxTransferKVManager._host_pool: JaxTransferKVManager#_host_pool.
  JaxTransferKVSender.req_id: JaxTransferKVSender#req_id().
  JaxTransferKVSender.transfer_started_at: JaxTransferKVSender#transfer_started_at().
  JaxTransferKVReceiver.req_id: JaxTransferKVReceiver#req_id().
  JaxTransferKVReceiver.transfer_started_at: JaxTransferKVReceiver#transfer_started_at().
  PMetadata.remote_addr: PMetadata#remote_addr.
  PMetadata.specs: PMetadata#specs.
  PMetadata.p_side_channel_host: PMetadata#p_side_channel_host.
  PMetadata.p_side_channel_port: PMetadata#p_side_channel_port.
  TransferStatus.uuid: TransferStatus#uuid.
  JaxTransferKVSender._transfer_id: JaxTransferKVSender#_transfer_id.
  JaxTransferKVSender._buffer_id: JaxTransferKVSender#_buffer_id.
  logger: logger.
  JaxTransferKVManager._zmq_notifier: JaxTransferKVManager#_zmq_notifier.
  JaxTransferKVManager._pull_worker_count: JaxTransferKVManager#_pull_worker_count.
  __all__: __all__.
---
# Module: [`python/sgl_jax/srt/disaggregation/jax_transfer/conn.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py)

## Classes
### `JaxTransferKVManager`  ·  implements/extends CommonKVManager
- def: [`python/sgl_jax/srt/disaggregation/jax_transfer/conn.py:92`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L92)
- doc: Concrete KV transfer manager for `jax.experimental.transfer`.
- signature: `class JaxTransferKVManager(CommonKVManager):`
- members:
  - `create_receiver(self, req_id: str)` — [`L250`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L250)
  - `create_sender(self, req_id: str)` — [`L245`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L245)
  - `enqueue_pull(self, receiver: JaxTransferKVReceiver)` — [`L138`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L138) — Hand a TRANSFERRING receiver to the background pull worker. — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)
  - `host_pool(self)` — [`L170`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L170)
  - `producer_handoff(self, uuid: str, payload: dict[str, jax.Array], *, use_d2h_staging: bool, buffer_id: int | None = None)` — [`L177`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L177) — Register ``payload`` entries for remote pull under sub-uuids. — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)
  - `wrapper(self)` — [`L162`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L162)
  - `zmq_notifier(self)` — [`L166`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L166)
- protocol/private: `__init__`[`L101`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L101), `_host_pool`[`L119`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L119), `_pull_queue`[`L127`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L127), `_pull_worker_count`[`L126`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L126), `_pull_worker_loop`[`L147`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L147), `_pull_workers`[`L128`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L128), `_wrapper`[`L117`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L117), `_zmq_notifier`[`L118`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L118)
- uses (calls/refs, reference-scoped): [`_run_pull`](conn.md#JaxTransferKVReceiver._run_pull), [`register_pull`](wrapper.md#JaxTransferWrapper.register_pull), [`JaxTransferKVSender`](conn.md#JaxTransferKVSender), [`JaxTransferKVReceiver`](conn.md#JaxTransferKVReceiver), [`register_receiver`](../common/core.md#CommonKVManager.register_receiver), [`register_sender`](../common/core.md#CommonKVManager.register_sender), [`HostKVPool`](../../mem_cache/host_kv_pool.md#HostKVPool), [`release`](wrapper.md#JaxTransferWrapper.release), [`CommonKVManager`](../common/core.md#CommonKVManager), [`JaxTransferWrapper`](wrapper.md#JaxTransferWrapper), [`copy_from_device`](../../mem_cache/host_kv_pool.md#HostKVPool.copy_from_device), [`ZmqPullNotifier`](../common/zmq_notifier.md#ZmqPullNotifier), [`TransferStatus`](conn.md#TransferStatus), [`on_done`](conn.md#TransferStatus.on_done), [`sub_uuids`](conn.md#TransferStatus.sub_uuids), [`array_pytree`](../../mem_cache/host_kv_pool.md#StagedData.array_pytree), [`uuid`](conn.md#TransferStatus.uuid), [`__init__`](../common/core.md#CommonKVManager.__init__), [`logger`](conn.md#logger)
- used by: [`install_disaggregation_wiring`](../runtime.md#install_disaggregation_wiring), [`_admit_decode_prealloc`](../decode.md#SchedulerDisaggregationDecodeMixin._admit_decode_prealloc), [`poll`](conn.md#JaxTransferKVReceiver.poll), [`_on_ack`](conn.md#JaxTransferKVSender._on_ack), [`fail`](conn.md#JaxTransferKVSender.fail), [`process_prefill_chunk`](../prefill.md#SchedulerDisaggregationPrefillMixin.process_prefill_chunk), [`_run_pull`](conn.md#JaxTransferKVReceiver._run_pull), [`send`](conn.md#JaxTransferKVSender.send), [`_shutdown`](../runtime.md#_make_disagg_shutdown._shutdown), [`__init__`](conn.md#JaxTransferKVReceiver.__init__), [`__init__`](conn.md#JaxTransferKVSender.__init__), [`disagg_kv_manager`](../prefill.md#SchedulerDisaggregationPrefillMixin.disagg_kv_manager), [`CommonKVManager`](../common/core.md#CommonKVManager), [`create_receiver`](../base/kv_manager.md#KVManager.create_receiver), [`create_sender`](../base/kv_manager.md#KVManager.create_sender), [`disagg_kv_manager`](../decode.md#SchedulerDisaggregationDecodeMixin.disagg_kv_manager)

### `JaxTransferKVReceiver`
- def: [`python/sgl_jax/srt/disaggregation/jax_transfer/conn.py:459`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L459)
- members:
  - `_run_pull(self)` — [`L613`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L613) — Run the pull on a background worker thread, off the decode
  - `abort(self)` — [`L493`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L493)
  - `clear(self)` — [`L490`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L490)
  - `fail(self, *, reason: str = "receiver_fail")` — [`L511`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L511)
  - `failure_exception(self)` — [`L496`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L496)
  - `init(self, p_metadata: PMetadata)` — [`L533`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L533)
  - `poll(self)` — [`L544`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L544) — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)
  - `req_id(self)` — [`L479`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L479)
  - `result(self)` — [`L483`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L483)
  - `transfer_started_at(self)` — [`L487`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L487)
- protocol/private: `__init__`[`L468`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L468), `_close_pull_timer`[`L657`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L657), `_metadata`[`L472`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L472), `_mgr`[`L470`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L470), `_pull_timer`[`L474`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L474), `_req_id`[`L471`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L471), `_results`[`L473`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L473), `_state_lock`[`L476`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L476), `_transfer_started_at`[`L475`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L475)
- uses (calls/refs, reference-scoped): [`KVPoll`](../base/kv_manager.md#KVPoll), [`record_terminal`](../common/core.md#CommonKVManager.record_terminal), [`_transition_to`](../base/kv_manager.md#StateHolder._transition_to), [`FAILED`](../base/kv_manager.md#KVPoll.FAILED), [`state`](../base/kv_manager.md#StateHolder.state), [`SUCCESS`](../base/kv_manager.md#KVPoll.SUCCESS), [`send_done`](../common/zmq_notifier.md#ZmqPullNotifier.send_done), [`zmq_notifier`](conn.md#JaxTransferKVManager.zmq_notifier), [`PD_TRANSFER_FAILURES_TOTAL`](../common/metrics.md#PD_TRANSFER_FAILURES_TOTAL), [`_prune_receiver`](../common/core.md#CommonKVManager._prune_receiver), [`TRANSFERRING`](../base/kv_manager.md#KVPoll.TRANSFERRING), [`JaxTransferKVManager`](conn.md#JaxTransferKVManager), [`time_phase`](../common/metrics.md#time_phase), [`get_terminal_record`](../common/core.md#CommonKVManager.get_terminal_record), [`_clear_terminal_record`](../common/core.md#CommonKVManager._clear_terminal_record), [`pull`](wrapper.md#JaxTransferWrapper.pull), [`wrapper`](conn.md#JaxTransferKVManager.wrapper), [`state`](../common/core.md#TerminalTransferRecord.state), [`WAITING_FOR_INPUT`](../base/kv_manager.md#KVPoll.WAITING_FOR_INPUT), [`__enter__`](../common/metrics.md#_DurationTimer.__enter__), [`__init__`](../base/kv_manager.md#StateHolder.__init__), [`uuid`](conn.md#PMetadata.uuid), [`enqueue_pull`](conn.md#JaxTransferKVManager.enqueue_pull), [`PMetadata`](conn.md#PMetadata), [`StateHolder`](../base/kv_manager.md#StateHolder), [`BOOTSTRAPPING`](../base/kv_manager.md#KVPoll.BOOTSTRAPPING), [`KVReceiver`](../base/kv_manager.md#KVReceiver), [`reason`](../common/core.md#TerminalTransferRecord.reason), [`p_side_channel_host`](conn.md#PMetadata.p_side_channel_host), [`p_side_channel_port`](conn.md#PMetadata.p_side_channel_port), [`remote_addr`](conn.md#PMetadata.remote_addr), [`specs`](conn.md#PMetadata.specs)
- used by: [`_admit_decode_prealloc`](../decode.md#SchedulerDisaggregationDecodeMixin._admit_decode_prealloc), [`abort_request`](../../managers/scheduler.md#Scheduler.abort_request), [`process_decode_queue`](../decode.md#SchedulerDisaggregationDecodeMixin.process_decode_queue), [`_drain_transfer_queue_synced`](../decode.md#SchedulerDisaggregationDecodeMixin._drain_transfer_queue_synced), [`drain_terminal`](../decode.md#DecodeTransferQueue.drain_terminal), [`receiver`](../decode.md#DecodeBookkeeping.receiver), [`_pull_worker_loop`](conn.md#JaxTransferKVManager._pull_worker_loop), [`create_receiver`](conn.md#JaxTransferKVManager.create_receiver), [`enqueue_pull`](conn.md#JaxTransferKVManager.enqueue_pull), [`poll`](../base/kv_manager.md#KVReceiver.poll), [`_pull_queue`](conn.md#JaxTransferKVManager._pull_queue), [`abort`](../base/kv_manager.md#KVReceiver.abort), [`clear`](../base/kv_manager.md#KVReceiver.clear), [`failure_exception`](../base/kv_manager.md#KVReceiver.failure_exception), [`init`](../base/kv_manager.md#KVReceiver.init)

### `JaxTransferKVSender`
- def: [`python/sgl_jax/srt/disaggregation/jax_transfer/conn.py:256`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L256)
- members:
  - `abort(self)` — [`L353`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L353)
  - `attach_payload(self, payload: dict[str, jax.Array], *, use_d2h_staging: bool, buffer_id: int | None = None)` — [`L299`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L299)
  - `clear(self)` — [`L350`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L350)
  - `fail(self, *, reason: str = "sender_fail")` — [`L371`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L371) — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)
  - `failure_exception(self)` — [`L356`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L356)
  - `init(self, kv_indices, transfer_id: str | None = None)` — [`L294`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L294)
  - `poll(self)` — [`L346`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L346)
  - `req_id(self)` — [`L283`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L283)
  - `send(self)` — [`L310`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L310) — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)
  - `transfer_started_at(self)` — [`L291`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L291)
  - `uuid(self)` — [`L287`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L287)
- protocol/private: `__init__`[`L269`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L269), `_ack_timer`[`L279`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L279), `_buffer_id`[`L276`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L276), `_close_ack_timer`[`L450`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L450), `_mgr`[`L271`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L271), `_on_ack`[`L400`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L400), `_payload`[`L274`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L274), `_req_id`[`L272`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L272), `_state_lock`[`L278`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L278), `_status`[`L277`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L277), `_transfer_id`[`L273`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L273), `_transfer_started_at`[`L280`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L280), `_use_d2h_staging`[`L275`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L275)
- uses (calls/refs, reference-scoped): [`KVPoll`](../base/kv_manager.md#KVPoll), [`record_terminal`](../common/core.md#CommonKVManager.record_terminal), [`_transition_to`](../base/kv_manager.md#StateHolder._transition_to), [`producer_handoff`](conn.md#JaxTransferKVManager.producer_handoff), [`FAILED`](../base/kv_manager.md#KVPoll.FAILED), [`mark_retired`](../common/zmq_notifier.md#ZmqPullNotifier.mark_retired), [`state`](../base/kv_manager.md#StateHolder.state), [`register_callback`](../common/zmq_notifier.md#ZmqPullNotifier.register_callback), [`SUCCESS`](../base/kv_manager.md#KVPoll.SUCCESS), [`zmq_notifier`](conn.md#JaxTransferKVManager.zmq_notifier), [`PD_TRANSFER_FAILURES_TOTAL`](../common/metrics.md#PD_TRANSFER_FAILURES_TOTAL), [`TRANSFERRING`](../base/kv_manager.md#KVPoll.TRANSFERRING), [`JaxTransferKVManager`](conn.md#JaxTransferKVManager), [`time_phase`](../common/metrics.md#time_phase), [`_prune_sender`](../common/core.md#CommonKVManager._prune_sender), [`get_terminal_record`](../common/core.md#CommonKVManager.get_terminal_record), [`unregister_callback`](../common/zmq_notifier.md#ZmqPullNotifier.unregister_callback), [`_clear_terminal_record`](../common/core.md#CommonKVManager._clear_terminal_record), [`release`](wrapper.md#JaxTransferWrapper.release), [`wrapper`](conn.md#JaxTransferKVManager.wrapper), [`state`](../common/core.md#TerminalTransferRecord.state), [`WAITING_FOR_INPUT`](../base/kv_manager.md#KVPoll.WAITING_FOR_INPUT), [`__enter__`](../common/metrics.md#_DurationTimer.__enter__), [`__init__`](../base/kv_manager.md#StateHolder.__init__), [`StateHolder`](../base/kv_manager.md#StateHolder), [`BOOTSTRAPPING`](../base/kv_manager.md#KVPoll.BOOTSTRAPPING), [`TransferStatus`](conn.md#TransferStatus), [`on_done`](conn.md#TransferStatus.on_done), [`sub_uuids`](conn.md#TransferStatus.sub_uuids), [`KVSender`](../base/kv_manager.md#KVSender), [`reason`](../common/core.md#TerminalTransferRecord.reason)
- used by: [`abort_request`](../../managers/scheduler.md#Scheduler.abort_request), [`process_prefill_chunk`](../prefill.md#SchedulerDisaggregationPrefillMixin.process_prefill_chunk), [`_on_prefill_transfer_terminal`](../prefill.md#SchedulerDisaggregationPrefillMixin._on_prefill_transfer_terminal), [`_finish_prefill_only_failure`](../prefill.md#SchedulerDisaggregationPrefillMixin._finish_prefill_only_failure), [`drain_terminal`](../prefill.md#PrefillBootstrapQueue.drain_terminal), [`add`](../prefill.md#PrefillBootstrapQueue.add), [`create_sender`](conn.md#JaxTransferKVManager.create_sender), [`sender`](../prefill.md#PrefillBookkeeping.sender), [`poll`](../base/kv_manager.md#KVSender.poll), [`abort`](../base/kv_manager.md#KVSender.abort), [`clear`](../base/kv_manager.md#KVSender.clear), [`failure_exception`](../base/kv_manager.md#KVSender.failure_exception), [`init`](../base/kv_manager.md#KVSender.init), [`send`](../base/kv_manager.md#KVSender.send)

### `PMetadata`
- def: [`python/sgl_jax/srt/disaggregation/jax_transfer/conn.py:58`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L58)
- doc: Out-of-band metadata D needs to pull from P.
- signature: `class PMetadata:`
- members:
  - `p_side_channel_host` — [`L71`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L71)
  - `p_side_channel_port` — [`L72`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L72)
  - `remote_addr` — [`L68`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L68)
  - `specs` — [`L70`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L70)
  - `uuid` — [`L69`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L69)
- used by: [`_admit_decode_prealloc`](../decode.md#SchedulerDisaggregationDecodeMixin._admit_decode_prealloc), [`poll`](conn.md#JaxTransferKVReceiver.poll), [`_run_pull`](conn.md#JaxTransferKVReceiver._run_pull), [`fail`](conn.md#JaxTransferKVReceiver.fail), [`_metadata`](conn.md#JaxTransferKVReceiver._metadata), [`init`](conn.md#JaxTransferKVReceiver.init)

### `TransferStatus`
- def: [`python/sgl_jax/srt/disaggregation/jax_transfer/conn.py:76`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L76)
- doc: Result of `JaxTransferKVManager.producer_handoff`.
- signature: `class TransferStatus:`
- members:
  - `on_done` — [`L89`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L89)
  - `sub_uuids` — [`L88`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L88)
  - `uuid` — [`L87`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L87)
- used by: [`_on_ack`](conn.md#JaxTransferKVSender._on_ack), [`fail`](conn.md#JaxTransferKVSender.fail), [`producer_handoff`](conn.md#JaxTransferKVManager.producer_handoff), [`_status`](conn.md#JaxTransferKVSender._status)

## Module values
- `__all__` — [`L44`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L44)
- `logger` — [`L54`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/conn.py#L54)

