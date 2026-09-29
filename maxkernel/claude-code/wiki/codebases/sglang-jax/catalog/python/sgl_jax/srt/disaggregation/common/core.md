---
title: 'Module: python/sgl_jax/srt/disaggregation/common/core.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/common/core.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.common.core`/
symbols:
  CommonKVManager.record_terminal: CommonKVManager#record_terminal().
  CommonKVManager.reap_once: CommonKVManager#reap_once().
  CommonKVManager.graceful_shutdown: CommonKVManager#graceful_shutdown().
  CommonKVManager.start_reaper: CommonKVManager#start_reaper().
  CommonKVManager.inflight_count: CommonKVManager#inflight_count().
  CommonKVManager._prune_receiver: CommonKVManager#_prune_receiver().
  CommonKVManager.register_sender: CommonKVManager#register_sender().
  CommonKVManager.register_receiver: CommonKVManager#register_receiver().
  CommonKVManager._terminal_records: CommonKVManager#_terminal_records.
  CommonKVManager._prune_sender: CommonKVManager#_prune_sender().
  CommonKVManager.get_terminal_record: CommonKVManager#get_terminal_record().
  CommonKVManager.stop_reaper: CommonKVManager#stop_reaper().
  CommonKVManager._clear_terminal_record: CommonKVManager#_clear_terminal_record().
  CommonKVManager._reaper_loop: CommonKVManager#_reaper_loop().
  TerminalTransferRecord.state: TerminalTransferRecord#state.
  CommonKVManager: CommonKVManager#
  CommonKVManager._reaper_thread: CommonKVManager#_reaper_thread.
  CommonKVManager._senders: CommonKVManager#_senders.
  CommonKVManager._receivers: CommonKVManager#_receivers.
  CommonKVManager._senders_lock: CommonKVManager#_senders_lock.
  CommonKVManager._receivers_lock: CommonKVManager#_receivers_lock.
  TerminalTransferRecord: TerminalTransferRecord#
  CommonKVManager._reaper_stop: CommonKVManager#_reaper_stop.
  TerminalTransferRecord.reason: TerminalTransferRecord#reason.
  CommonKVManager._terminal_records_lock: CommonKVManager#_terminal_records_lock.
  CommonKVManager._ack_timeout_s: CommonKVManager#_ack_timeout_s.
  CommonKVManager._pull_timeout_s: CommonKVManager#_pull_timeout_s.
  CommonKVManager._reaper_interval_s: CommonKVManager#_reaper_interval_s.
  TerminalTransferRecord.req_id: TerminalTransferRecord#req_id.
  TerminalTransferRecord.role: TerminalTransferRecord#role.
  TerminalTransferRecord.transfer_id: TerminalTransferRecord#transfer_id.
  TerminalTransferRecord.terminal_at: TerminalTransferRecord#terminal_at.
  CommonKVManager.__init__: CommonKVManager#__init__().
  CommonKVManager._max_terminal_records: CommonKVManager#_max_terminal_records.
---
# Module: [`python/sgl_jax/srt/disaggregation/common/core.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py)

## Classes
### `CommonKVManager`  ·  implements/extends KVManager
- def: [`python/sgl_jax/srt/disaggregation/common/core.py:25`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L25)
- doc: Shared lifecycle manager for request-scoped transfers.
- signature: `class CommonKVManager(KVManager):`
- members:
  - `get_terminal_record(self, req_id: str, *, role: str)` — [`L128`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L128)
  - `graceful_shutdown(self, drain_timeout_seconds: float = 30)` — [`L211`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L211) — Drain in-flight transfers, then abort stragglers.
  - `inflight_count(self)` — [`L204`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L204)
  - `reap_once(self, now: float)` — [`L166`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L166) — Single reaper pass. Returns ``(timed_out_senders, timed_out_receivers)``.
  - `record_terminal(self, req_id: str, *, role: str, transfer_id: str, state: KVPoll, reason: str)` — [`L104`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L104) — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)
  - `register_receiver(self, req_id: str, receiver: object)` — [`L72`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L72)
  - `register_sender(self, req_id: str, sender: object)` — [`L63`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L63)
  - `start_reaper(self)` — [`L140`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L140)
  - `stop_reaper(self)` — [`L153`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L153)
- protocol/private: `__init__`[`L37`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L37), `_ack_timeout_s`[`L53`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L53), `_clear_terminal_record`[`L99`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L99), `_max_terminal_records`[`L51`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L51), `_prune_receiver`[`L88`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L88), `_prune_sender`[`L81`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L81), `_pull_timeout_s`[`L54`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L54), `_reaper_interval_s`[`L55`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L55), `_reaper_loop`[`L160`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L160), `_reaper_stop`[`L56`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L56), `_reaper_thread`[`L57`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L57), `_receivers`[`L47`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L47), `_receivers_lock`[`L45`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L45), `_senders`[`L46`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L46), `_senders_lock`[`L44`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L44), `_terminal_records`[`L50`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L50), `_terminal_records_lock`[`L49`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L49)
- uses (calls/refs, reference-scoped): [`KVPoll`](../base/kv_manager.md#KVPoll), [`JaxTransferKVManager`](../jax_transfer/conn.md#JaxTransferKVManager), [`state`](core.md#TerminalTransferRecord.state), [`PD_TRANSFER_INFLIGHT`](metrics.md#PD_TRANSFER_INFLIGHT), [`KVManager`](../base/kv_manager.md#KVManager), [`TerminalTransferRecord`](core.md#TerminalTransferRecord), [`reason`](core.md#TerminalTransferRecord.reason), [`req_id`](core.md#TerminalTransferRecord.req_id), [`role`](core.md#TerminalTransferRecord.role), [`terminal_at`](core.md#TerminalTransferRecord.terminal_at), [`transfer_id`](core.md#TerminalTransferRecord.transfer_id)
- used by: [`install_disaggregation_wiring`](../runtime.md#install_disaggregation_wiring), [`poll`](../jax_transfer/conn.md#JaxTransferKVReceiver.poll), [`_on_ack`](../jax_transfer/conn.md#JaxTransferKVSender._on_ack), [`fail`](../jax_transfer/conn.md#JaxTransferKVSender.fail), [`_run_pull`](../jax_transfer/conn.md#JaxTransferKVReceiver._run_pull), [`fail`](../jax_transfer/conn.md#JaxTransferKVReceiver.fail), [`_shutdown`](../runtime.md#_make_disagg_shutdown._shutdown), [`_decode_backlog_snapshot`](../decode.md#SchedulerDisaggregationDecodeMixin._decode_backlog_snapshot), [`failure_exception`](../jax_transfer/conn.md#JaxTransferKVSender.failure_exception), [`failure_exception`](../jax_transfer/conn.md#JaxTransferKVReceiver.failure_exception), [`JaxTransferKVManager`](../jax_transfer/conn.md#JaxTransferKVManager), [`__init__`](../jax_transfer/conn.md#JaxTransferKVManager.__init__), [`clear`](../jax_transfer/conn.md#JaxTransferKVSender.clear), [`clear`](../jax_transfer/conn.md#JaxTransferKVReceiver.clear), [`create_receiver`](../jax_transfer/conn.md#JaxTransferKVManager.create_receiver), [`create_sender`](../jax_transfer/conn.md#JaxTransferKVManager.create_sender), [`KVManager`](../base/kv_manager.md#KVManager)

### `TerminalTransferRecord`
- def: [`python/sgl_jax/srt/disaggregation/common/core.py:16`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L16)
- signature: `class TerminalTransferRecord:`
- members:
  - `reason` — [`L21`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L21)
  - `req_id` — [`L17`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L17)
  - `role` — [`L18`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L18)
  - `state` — [`L20`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L20)
  - `terminal_at` — [`L22`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L22)
  - `transfer_id` — [`L19`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/core.py#L19)
- uses (calls/refs, reference-scoped): [`KVPoll`](../base/kv_manager.md#KVPoll)
- used by: [`record_terminal`](core.md#CommonKVManager.record_terminal), [`failure_exception`](../jax_transfer/conn.md#JaxTransferKVSender.failure_exception), [`failure_exception`](../jax_transfer/conn.md#JaxTransferKVReceiver.failure_exception), [`_terminal_records`](core.md#CommonKVManager._terminal_records), [`get_terminal_record`](core.md#CommonKVManager.get_terminal_record)

