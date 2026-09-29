---
title: 'Module: python/sgl_jax/srt/disaggregation/common/zmq_notifier.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/common/zmq_notifier.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.common.zmq_notifier`/
symbols:
  ZmqPullNotifier._listen_loop: ZmqPullNotifier#_listen_loop().
  ZmqPullNotifier.start: ZmqPullNotifier#start().
  ZmqPullNotifier.mark_retired: ZmqPullNotifier#mark_retired().
  ZmqPullNotifier.register_callback: ZmqPullNotifier#register_callback().
  ZmqPullNotifier.stop: ZmqPullNotifier#stop().
  ZmqPullNotifier.send_done: ZmqPullNotifier#send_done().
  ZmqPullNotifier._router: ZmqPullNotifier#_router.
  ZmqPullNotifier._retired: ZmqPullNotifier#_retired.
  ZmqPullNotifier.unregister_callback: ZmqPullNotifier#unregister_callback().
  logger: logger.
  ZmqPullNotifier._started: ZmqPullNotifier#_started.
  ZmqPullNotifier._callbacks: ZmqPullNotifier#_callbacks.
  ZmqPullNotifier._listener_thread: ZmqPullNotifier#_listener_thread.
  ZmqPullNotifier: ZmqPullNotifier#
  ZmqPullNotifier._role: ZmqPullNotifier#_role.
  ZmqPullNotifier._port: ZmqPullNotifier#_port.
  ZmqPullNotifier._callbacks_lock: ZmqPullNotifier#_callbacks_lock.
  ZmqPullNotifier._dispatching: ZmqPullNotifier#_dispatching.
  ZmqPullNotifier._shared_secret: ZmqPullNotifier#_shared_secret.
  PullDoneCallback: PullDoneCallback.
  RetiredTransferInfo: RetiredTransferInfo#
  ZmqPullNotifier._host: ZmqPullNotifier#_host.
  ZmqPullNotifier._stop_event: ZmqPullNotifier#_stop_event.
  ZmqPullNotifier.role: ZmqPullNotifier#role().
  ZmqPullNotifier.host: ZmqPullNotifier#host().
  ZmqPullNotifier.port: ZmqPullNotifier#port().
  ZmqPullNotifier.is_started: ZmqPullNotifier#is_started().
  RetiredTransferInfo.state: RetiredTransferInfo#state.
  RetiredTransferInfo.reason: RetiredTransferInfo#reason.
  ZmqPullNotifier._ctx: ZmqPullNotifier#_ctx.
  RetiredTransferInfo.retired_at: RetiredTransferInfo#retired_at.
  ZmqPullNotifier._max_retired: ZmqPullNotifier#_max_retired.
  ZmqPullNotifier.__init__: ZmqPullNotifier#__init__().
---
# Module: [`python/sgl_jax/srt/disaggregation/common/zmq_notifier.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py)

## Classes
### `RetiredTransferInfo`
- def: [`python/sgl_jax/srt/disaggregation/common/zmq_notifier.py:22`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L22)
- signature: `class RetiredTransferInfo:`
- members:
  - `reason` — [`L24`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L24)
  - `retired_at` — [`L25`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L25)
  - `state` — [`L23`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L23)
- used by: [`_listen_loop`](zmq_notifier.md#ZmqPullNotifier._listen_loop), [`mark_retired`](zmq_notifier.md#ZmqPullNotifier.mark_retired), [`_retired`](zmq_notifier.md#ZmqPullNotifier._retired)

### `ZmqPullNotifier`
- def: [`python/sgl_jax/srt/disaggregation/common/zmq_notifier.py:28`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L28)
- doc: Backend-local ZMQ notifier for `pull-done` events.
- signature: `class ZmqPullNotifier:`
- members:
  - `host(self)` — [`L68`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L68)
  - `is_started(self)` — [`L76`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L76)
  - `mark_retired(self, uuid: bytes, *, state: str, reason: str)` — [`L162`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L162) — Remember that ``uuid`` reached a terminal state. — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)
  - `port(self)` — [`L72`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L72)
  - `register_callback(self, uuid: bytes, cb: PullDoneCallback)` — [`L139`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L139) — Register ``cb`` to fire when an ack for ``uuid`` arrives. — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)
  - `role(self)` — [`L64`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L64)
  - `send_done(self, uuid: bytes, target_host: str, target_port: int)` — [`L186`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L186) — D→P: tell the prefill side ``uuid`` has been pulled. — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)
  - `start(self)` — [`L79`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L79) — Idempotent. On P, binds the ROUTER and spawns the listener
  - `stop(self)` — [`L109`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L109) — Stop the listener and tear down the ROUTER. Idempotent.
  - `unregister_callback(self, uuid: bytes)` — [`L158`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L158)
- protocol/private: `__init__`[`L38`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L38), `_callbacks`[`L57`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L57), `_callbacks_lock`[`L56`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L56), `_ctx`[`L52`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L52), `_dispatching`[`L58`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L58), `_host`[`L49`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L49), `_listen_loop`[`L217`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L217), `_listener_thread`[`L55`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L55), `_max_retired`[`L60`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L60), `_port`[`L50`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L50), `_retired`[`L59`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L59), `_role`[`L48`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L48), `_router`[`L53`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L53), `_shared_secret`[`L51`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L51), `_started`[`L61`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L61), `_stop_event`[`L54`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L54)
- uses (calls/refs, reference-scoped): [`PD_TRANSFER_FAILURES_TOTAL`](metrics.md#PD_TRANSFER_FAILURES_TOTAL), [`logger`](zmq_notifier.md#logger), [`verify_tag`](../pd_auth.md#verify_tag), [`RetiredTransferInfo`](zmq_notifier.md#RetiredTransferInfo), [`PullDoneCallback`](zmq_notifier.md#PullDoneCallback), [`compute_tag`](../pd_auth.md#compute_tag), [`reason`](zmq_notifier.md#RetiredTransferInfo.reason), [`state`](zmq_notifier.md#RetiredTransferInfo.state), [`retired_at`](zmq_notifier.md#RetiredTransferInfo.retired_at)
- used by: [`install_disaggregation_wiring`](../runtime.md#install_disaggregation_wiring), [`poll`](../jax_transfer/conn.md#JaxTransferKVReceiver.poll), [`_on_ack`](../jax_transfer/conn.md#JaxTransferKVSender._on_ack), [`fail`](../jax_transfer/conn.md#JaxTransferKVSender.fail), [`send`](../jax_transfer/conn.md#JaxTransferKVSender.send), [`_shutdown`](../runtime.md#_make_disagg_shutdown._shutdown), [`zmq_notifier`](../jax_transfer/conn.md#JaxTransferKVManager.zmq_notifier), [`__init__`](../jax_transfer/conn.md#JaxTransferKVManager.__init__)

## Module values
- `PullDoneCallback` — [`L18`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L18)
- `logger` — [`L15`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/zmq_notifier.py#L15)

