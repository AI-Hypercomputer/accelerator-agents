---
title: 'Module: python/sgl_jax/srt/disaggregation/base/kv_manager.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/base/kv_manager.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.base.kv_manager`/
symbols:
  KVPoll: KVPoll#
  StateHolder._transition_to: StateHolder#_transition_to().
  KVPoll.FAILED: KVPoll#FAILED.
  StateHolder.state: StateHolder#state().
  LEGAL_TRANSITIONS.LEGAL_TRANSITIONS: LEGAL_TRANSITIONS.LEGAL_TRANSITIONS.
  KVPoll.SUCCESS: KVPoll#SUCCESS.
  KVPoll.TRANSFERRING: KVPoll#TRANSFERRING.
  StateHolder._state: StateHolder#_state.
  KVPoll.WAITING_FOR_INPUT: KVPoll#WAITING_FOR_INPUT.
  StateHolder.__init__: StateHolder#__init__().
  is_legal_transition: is_legal_transition().
  KVPoll.BOOTSTRAPPING: KVPoll#BOOTSTRAPPING.
  StateHolder: StateHolder#
  KVManager.create_sender: KVManager#create_sender().
  KVManager.create_receiver: KVManager#create_receiver().
  KVSender.poll: KVSender#poll().
  KVReceiver.poll: KVReceiver#poll().
  KVManager: KVManager#
  StateHolder._role: StateHolder#_role.
  KVSender: KVSender#
  KVReceiver: KVReceiver#
  StateHolder.role: StateHolder#role().
  KVSender.init: KVSender#init().
  KVSender.send: KVSender#send().
  KVSender.clear: KVSender#clear().
  KVSender.abort: KVSender#abort().
  KVSender.failure_exception: KVSender#failure_exception().
  KVReceiver.init: KVReceiver#init().
  KVReceiver.clear: KVReceiver#clear().
  KVReceiver.abort: KVReceiver#abort().
  KVReceiver.failure_exception: KVReceiver#failure_exception().
  logger: logger.
---
# Module: [`python/sgl_jax/srt/disaggregation/base/kv_manager.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py)

## Classes
### `KVManager`  ·  implements/extends ABC
- def: [`python/sgl_jax/srt/disaggregation/base/kv_manager.py:99`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L99)
- doc: Per-process factory that produces sender/receiver handles.
- signature: `class KVManager(abc.ABC):`
- members:
  - `create_receiver(self, req_id: str)` — [`L106`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L106)
  - `create_sender(self, req_id: str)` — [`L103`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L103)
- uses (calls/refs, reference-scoped): [`CommonKVManager`](../common/core.md#CommonKVManager), [`create_receiver`](../jax_transfer/conn.md#JaxTransferKVManager.create_receiver), [`create_sender`](../jax_transfer/conn.md#JaxTransferKVManager.create_sender), [`KVReceiver`](kv_manager.md#KVReceiver), [`KVSender`](kv_manager.md#KVSender)
- used by: [`CommonKVManager`](../common/core.md#CommonKVManager)

### `KVPoll`  ·  implements/extends Enum
- def: [`python/sgl_jax/srt/disaggregation/base/kv_manager.py:20`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L20) — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)
- doc: Lifecycle of a single KV transfer request.
- signature: `class KVPoll(enum.Enum):`
- members:
  - `BOOTSTRAPPING` — [`L23`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L23)
  - `FAILED` — [`L27`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L27)
  - `SUCCESS` — [`L26`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L26)
  - `TRANSFERRING` — [`L25`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L25)
  - `WAITING_FOR_INPUT` — [`L24`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L24)
- used by: [`poll`](../jax_transfer/conn.md#JaxTransferKVReceiver.poll), [`process_decode_queue`](../decode.md#SchedulerDisaggregationDecodeMixin.process_decode_queue), [`_on_ack`](../jax_transfer/conn.md#JaxTransferKVSender._on_ack), [`fail`](../jax_transfer/conn.md#JaxTransferKVSender.fail), [`_run_pull`](../jax_transfer/conn.md#JaxTransferKVReceiver._run_pull), [`send`](../jax_transfer/conn.md#JaxTransferKVSender.send), [`fail`](../jax_transfer/conn.md#JaxTransferKVReceiver.fail), [`_on_prefill_transfer_terminal`](../prefill.md#SchedulerDisaggregationPrefillMixin._on_prefill_transfer_terminal), [`_drain_transfer_queue_synced`](../decode.md#SchedulerDisaggregationDecodeMixin._drain_transfer_queue_synced), [`record_terminal`](../common/core.md#CommonKVManager.record_terminal), [`_transition_to`](kv_manager.md#StateHolder._transition_to), [`drain_terminal`](../decode.md#DecodeTransferQueue.drain_terminal), [`drain_terminal`](../prefill.md#PrefillBootstrapQueue.drain_terminal), [`state`](kv_manager.md#StateHolder.state), [`failure_exception`](../jax_transfer/conn.md#JaxTransferKVSender.failure_exception), [`failure_exception`](../jax_transfer/conn.md#JaxTransferKVReceiver.failure_exception), [`LEGAL_TRANSITIONS`](kv_manager.md#LEGAL_TRANSITIONS.LEGAL_TRANSITIONS), [`init`](../jax_transfer/conn.md#JaxTransferKVSender.init), [`init`](../jax_transfer/conn.md#JaxTransferKVReceiver.init), [`__init__`](../jax_transfer/conn.md#JaxTransferKVReceiver.__init__), [`__init__`](../jax_transfer/conn.md#JaxTransferKVSender.__init__), [`synced_terminal_rooms`](../common/multihost_sync.md#synced_terminal_rooms), [`poll`](../jax_transfer/conn.md#JaxTransferKVSender.poll), [`_state`](kv_manager.md#StateHolder._state), [`state`](../common/core.md#TerminalTransferRecord.state), [`__init__`](kv_manager.md#StateHolder.__init__), [`is_legal_transition`](kv_manager.md#is_legal_transition), [`synced_state`](../decode.md#DecodeBookkeeping.synced_state), [`poll`](kv_manager.md#KVReceiver.poll), [`poll`](kv_manager.md#KVSender.poll)

### `KVReceiver`
- def: [`python/sgl_jax/srt/disaggregation/base/kv_manager.py:135`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L135)
- members:
  - `abort(self)` — [`L149`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L149) — Abort the current transfer.
  - `clear(self)` — [`L145`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L145) — Drop backend-local retained terminal state.
  - `failure_exception(self)` — [`L153`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L153) — Raise the terminal transfer failure as an exception.
  - `init(self, p_metadata)` — [`L139`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L139)
  - `poll(self)` — [`L142`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L142)
- uses (calls/refs, reference-scoped): [`KVPoll`](kv_manager.md#KVPoll), [`poll`](../jax_transfer/conn.md#JaxTransferKVReceiver.poll), [`failure_exception`](../jax_transfer/conn.md#JaxTransferKVReceiver.failure_exception), [`init`](../jax_transfer/conn.md#JaxTransferKVReceiver.init), [`clear`](../jax_transfer/conn.md#JaxTransferKVReceiver.clear), [`abort`](../jax_transfer/conn.md#JaxTransferKVReceiver.abort)
- used by: [`JaxTransferKVReceiver`](../jax_transfer/conn.md#JaxTransferKVReceiver), [`create_receiver`](kv_manager.md#KVManager.create_receiver)

### `KVSender`
- def: [`python/sgl_jax/srt/disaggregation/base/kv_manager.py:109`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L109)
- members:
  - `abort(self)` — [`L126`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L126) — Abort the current transfer.
  - `clear(self)` — [`L122`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L122) — Drop backend-local retained terminal state.
  - `failure_exception(self)` — [`L130`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L130) — Raise the terminal transfer failure as an exception.
  - `init(self, kv_indices, transfer_id: str | None = None)` — [`L113`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L113)
  - `poll(self)` — [`L119`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L119)
  - `send(self)` — [`L116`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L116)
- uses (calls/refs, reference-scoped): [`KVPoll`](kv_manager.md#KVPoll), [`send`](../jax_transfer/conn.md#JaxTransferKVSender.send), [`failure_exception`](../jax_transfer/conn.md#JaxTransferKVSender.failure_exception), [`init`](../jax_transfer/conn.md#JaxTransferKVSender.init), [`clear`](../jax_transfer/conn.md#JaxTransferKVSender.clear), [`poll`](../jax_transfer/conn.md#JaxTransferKVSender.poll), [`abort`](../jax_transfer/conn.md#JaxTransferKVSender.abort)
- used by: [`JaxTransferKVSender`](../jax_transfer/conn.md#JaxTransferKVSender), [`create_sender`](kv_manager.md#KVManager.create_sender)

### `StateHolder`
- def: [`python/sgl_jax/srt/disaggregation/base/kv_manager.py:46`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L46)
- doc: Shared lifecycle bookkeeping for `KVSender` / `KVReceiver`.
- signature: `class StateHolder:`
- members:
  - `role(self)` — [`L74`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L74)
  - `state(self)` — [`L70`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L70)
- protocol/private: `__init__`[`L60`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L60), `_role`[`L67`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L67), `_state`[`L66`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L66), `_transition_to`[`L77`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L77)
- uses (calls/refs, reference-scoped): [`KVPoll`](kv_manager.md#KVPoll), [`is_legal_transition`](kv_manager.md#is_legal_transition), [`BOOTSTRAPPING`](kv_manager.md#KVPoll.BOOTSTRAPPING), [`PD_STATE_TRANSITION_TOTAL`](../common/metrics.md#PD_STATE_TRANSITION_TOTAL), [`logger`](kv_manager.md#logger)
- used by: [`poll`](../jax_transfer/conn.md#JaxTransferKVReceiver.poll), [`_on_ack`](../jax_transfer/conn.md#JaxTransferKVSender._on_ack), [`fail`](../jax_transfer/conn.md#JaxTransferKVSender.fail), [`_run_pull`](../jax_transfer/conn.md#JaxTransferKVReceiver._run_pull), [`send`](../jax_transfer/conn.md#JaxTransferKVSender.send), [`fail`](../jax_transfer/conn.md#JaxTransferKVReceiver.fail), [`init`](../jax_transfer/conn.md#JaxTransferKVSender.init), [`init`](../jax_transfer/conn.md#JaxTransferKVReceiver.init), [`__init__`](../jax_transfer/conn.md#JaxTransferKVReceiver.__init__), [`__init__`](../jax_transfer/conn.md#JaxTransferKVSender.__init__), [`JaxTransferKVSender`](../jax_transfer/conn.md#JaxTransferKVSender), [`JaxTransferKVReceiver`](../jax_transfer/conn.md#JaxTransferKVReceiver), [`poll`](../jax_transfer/conn.md#JaxTransferKVSender.poll)

## Functions
- `is_legal_transition(current: KVPoll, next_state: KVPoll)` — [`L42`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L42) — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)

## Module values
- `LEGAL_TRANSITIONS` — [`L30`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L30) — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)
- `logger` — [`L17`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/base/kv_manager.py#L17)

