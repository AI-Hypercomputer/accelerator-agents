---
title: 'Module: python/sgl_jax/srt/disaggregation/common/metrics.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/common/metrics.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.common.metrics`/
symbols:
  PD_TRANSFER_FAILURES_TOTAL: PD_TRANSFER_FAILURES_TOTAL.
  time_phase: time_phase().
  host_pool_alloc: host_pool_alloc().
  host_pool_free: host_pool_free().
  _DurationTimer.__exit__: _DurationTimer#__exit__().
  _DurationTimer.__enter__: _DurationTimer#__enter__().
  PD_TRANSFER_BYTES_TOTAL: PD_TRANSFER_BYTES_TOTAL.
  PD_TRANSFER_INFLIGHT: PD_TRANSFER_INFLIGHT.
  _DurationTimer: _DurationTimer#
  _Noop: _Noop#
  _pool_in_use._pool_in_use: _pool_in_use._pool_in_use.
  _PROM_AVAILABLE: _PROM_AVAILABLE.
  _DurationTimer._start: _DurationTimer#_start.
  _Noop.labels: _Noop#labels().
  _DurationTimer.__init__: _DurationTimer#__init__().
  PD_STATE_TRANSITION_TOTAL: PD_STATE_TRANSITION_TOTAL.
  PD_HOST_POOL_USED_BUFFERS: PD_HOST_POOL_USED_BUFFERS.
  PD_BOOTSTRAP_REGISTRY_SIZE: PD_BOOTSTRAP_REGISTRY_SIZE.
  _pool_in_use_lock: _pool_in_use_lock.
  logger: logger.
  PD_TRANSFER_DURATION_SECONDS: PD_TRANSFER_DURATION_SECONDS.
  _Noop.__init__: _Noop#__init__().
  _Noop.inc: _Noop#inc().
  _Noop.dec: _Noop#dec().
  _Noop.set: _Noop#set().
  _Noop.observe: _Noop#observe().
  _DurationTimer.__slots__: _DurationTimer#__slots__.
---
# Module: [`python/sgl_jax/srt/disaggregation/common/metrics.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py)

## Classes
### `_DurationTimer`
- def: [`python/sgl_jax/srt/disaggregation/common/metrics.py:92`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L92)
- doc: Context manager / explicit timer that observes wall-clock
- signature: `class _DurationTimer:`
- protocol/private: `__enter__`[`L103`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L103), `__exit__`[`L109`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L109), `__init__`[`L99`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L99), `__slots__`[`L97`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L97), `_start`[`L101`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L101)
- uses (calls/refs, reference-scoped): [`logger`](metrics.md#logger)
- used by: [`poll`](../jax_transfer/conn.md#JaxTransferKVReceiver.poll), [`send`](../jax_transfer/conn.md#JaxTransferKVSender.send), [`time_phase`](metrics.md#time_phase)

### `_Noop`
- def: [`python/sgl_jax/srt/disaggregation/common/metrics.py:22`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L22)
- doc: Stand-in for missing prometheus_client primitives.
- signature: `class _Noop:`
- members:
  - `dec(self, amount: float = 1)` — [`L34`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L34)
  - `inc(self, amount: float = 1)` — [`L31`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L31)
  - `labels(self, *args, **kwargs)` — [`L28`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L28)
  - `observe(self, value: float)` — [`L40`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L40)
  - `set(self, value: float)` — [`L37`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L37)
- protocol/private: `__init__`[`L25`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L25)

## Functions
- `host_pool_alloc(pool_name: str, count: int = 1)` — [`L137`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L137)
- `host_pool_free(pool_name: str, count: int = 1)` — [`L145`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L145)
- `time_phase(phase: str, role: str)` — [`L121`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L121) — Return a context manager that records a phase duration into — documented in [python-sgl_jax-srt-disaggregation-jax_transfer-conn](../../../../../../concepts/python-sgl_jax-srt-disaggregation-jax_transfer-conn.md)

## Module values
- `PD_BOOTSTRAP_REGISTRY_SIZE` — [`L82`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L82)
- `PD_HOST_POOL_USED_BUFFERS` — [`L71`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L71)
- `PD_STATE_TRANSITION_TOTAL` — [`L50`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L50)
- `PD_TRANSFER_BYTES_TOTAL` — [`L55`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L55)
- `PD_TRANSFER_DURATION_SECONDS` — [`L60`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L60)
- `PD_TRANSFER_FAILURES_TOTAL` — [`L77`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L77)
- `PD_TRANSFER_INFLIGHT` — [`L65`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L65)
- `_PROM_AVAILABLE` — [`L18`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L18)
- `_pool_in_use` — [`L134`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L134)
- `_pool_in_use_lock` — [`L133`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L133)
- `logger` — [`L13`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/common/metrics.py#L13)

