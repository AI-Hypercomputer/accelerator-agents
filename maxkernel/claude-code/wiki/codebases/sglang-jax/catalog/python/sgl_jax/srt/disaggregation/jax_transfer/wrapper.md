---
title: 'Module: python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.jax_transfer.wrapper`/
symbols:
  JaxTransferWrapper.start: JaxTransferWrapper#start().
  JaxTransferWrapper.register_pull: JaxTransferWrapper#register_pull().
  get_or_create_wrapper: get_or_create_wrapper().
  _GLOBAL_WRAPPER._GLOBAL_WRAPPER: _GLOBAL_WRAPPER._GLOBAL_WRAPPER.
  JaxTransferWrapper.release: JaxTransferWrapper#release().
  JaxTransferWrapper.pull: JaxTransferWrapper#pull().
  JaxTransferWrapper._connect: JaxTransferWrapper#_connect().
  JaxTransferWrapper._server: JaxTransferWrapper#_server.
  JaxTransferWrapper._started: JaxTransferWrapper#_started.
  JaxTransferWrapper: JaxTransferWrapper#
  _reset_singleton_for_test: _reset_singleton_for_test().
  JaxTransferWrapper.host_ip: JaxTransferWrapper#host_ip().
  JaxTransferWrapper.port: JaxTransferWrapper#port().
  JaxTransferWrapper.channel_number: JaxTransferWrapper#channel_number().
  JaxTransferWrapper._host_ip: JaxTransferWrapper#_host_ip.
  JaxTransferWrapper._channel_number: JaxTransferWrapper#_channel_number.
  JaxTransferWrapper._pending: JaxTransferWrapper#_pending.
  JaxTransferWrapper._links: JaxTransferWrapper#_links.
  JaxTransferWrapper.is_started: JaxTransferWrapper#is_started().
  JaxTransferWrapper.server: JaxTransferWrapper#server().
  _GLOBAL_LOCK: _GLOBAL_LOCK.
  _uuid_to_int: _uuid_to_int().
  JaxTransferWrapper._port: JaxTransferWrapper#_port.
  JaxTransferWrapper._pending_lock: JaxTransferWrapper#_pending_lock.
  logger: logger.
  JaxTransferWrapper._init_lock: JaxTransferWrapper#_init_lock.
  JaxTransferWrapper._links_lock: JaxTransferWrapper#_links_lock.
  JaxTransferWrapper.__init__: JaxTransferWrapper#__init__().
---
# Module: [`python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py)

## Classes
### `JaxTransferWrapper`
- def: [`python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py:50`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L50)
- members:
  - `channel_number(self)` — [`L87`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L87)
  - `host_ip(self)` — [`L79`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L79)
  - `is_started(self)` — [`L91`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L91)
  - `port(self)` — [`L83`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L83)
  - `pull(self, uuid: str, spec: Any, remote_addr: str | None = None)` — [`L189`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L189) — Pull a previously registered buffer from ``remote_addr``.
  - `register_pull(self, uuid: str, data: Any)` — [`L132`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L132) — Register ``data`` for a future remote pull keyed by ``uuid``.
  - `release(self, uuid: str)` — [`L221`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L221) — Drop the wrapper's reference to a previously registered buffer.
  - `server(self)` — [`L95`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L95)
  - `start(self)` — [`L98`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L98) — Idempotent. Returns the underlying transfer server.
- protocol/private: `__init__`[`L56`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L56), `_channel_number`[`L64`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L64), `_connect`[`L235`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L235), `_host_ip`[`L62`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L62), `_init_lock`[`L65`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L65), `_links`[`L76`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L76), `_links_lock`[`L75`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L75), `_pending`[`L71`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L71), `_pending_lock`[`L70`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L70), `_port`[`L63`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L63), `_server`[`L66`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L66), `_started`[`L67`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L67)
- uses (calls/refs, reference-scoped): [`PD_TRANSFER_BYTES_TOTAL`](../common/metrics.md#PD_TRANSFER_BYTES_TOTAL), [`_uuid_to_int`](wrapper.md#_uuid_to_int), [`logger`](wrapper.md#logger)
- used by: [`install_disaggregation_wiring`](../runtime.md#install_disaggregation_wiring), [`_on_ack`](conn.md#JaxTransferKVSender._on_ack), [`fail`](conn.md#JaxTransferKVSender.fail), [`_run_pull`](conn.md#JaxTransferKVReceiver._run_pull), [`producer_handoff`](conn.md#JaxTransferKVManager.producer_handoff), [`get_or_create_wrapper`](wrapper.md#get_or_create_wrapper), [`__init__`](conn.md#JaxTransferKVManager.__init__), [`_GLOBAL_WRAPPER`](wrapper.md#_GLOBAL_WRAPPER._GLOBAL_WRAPPER), [`wrapper`](conn.md#JaxTransferKVManager.wrapper)

## Functions
- `_reset_singleton_for_test()` — [`L277`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L277) — Test-only: clear the module-level singleton between cases.
- `_uuid_to_int(uuid: str)` — [`L39`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L39) — Stable mapping from a public ``str`` uuid to the 32-bit int that the
- `get_or_create_wrapper(host_ip: str, port: int, channel_number: int = 1)` — [`L244`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L244) — Return the process-level wrapper, creating it on first call.

## Module values
- `_GLOBAL_LOCK` — [`L35`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L35)
- `_GLOBAL_WRAPPER` — [`L36`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L36)
- `logger` — [`L32`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/jax_transfer/wrapper.py#L32)

