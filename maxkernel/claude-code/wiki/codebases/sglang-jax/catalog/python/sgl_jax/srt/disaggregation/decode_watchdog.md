---
title: 'Module: python/sgl_jax/srt/disaggregation/decode_watchdog.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/decode_watchdog.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.decode_watchdog`/
symbols:
  EventLoopWatchdog.check_once: EventLoopWatchdog#check_once().
  EventLoopWatchdog.start: EventLoopWatchdog#start().
  EventLoopWatchdog.beat: EventLoopWatchdog#beat().
  EventLoopWatchdog._report: EventLoopWatchdog#_report().
  EventLoopWatchdog._loop: EventLoopWatchdog#_loop().
  EventLoopWatchdog.stop: EventLoopWatchdog#stop().
  EventLoopWatchdog._thread: EventLoopWatchdog#_thread.
  EventLoopWatchdog._tick: EventLoopWatchdog#_tick.
  EventLoopWatchdog._stop: EventLoopWatchdog#_stop.
  EventLoopWatchdog._traceback_dumper: EventLoopWatchdog#_traceback_dumper.
  EventLoopWatchdog.enabled: EventLoopWatchdog#enabled().
  EventLoopWatchdog._clock: EventLoopWatchdog#_clock.
  EventLoopWatchdog._beat_ts: EventLoopWatchdog#_beat_ts.
  EventLoopWatchdog: EventLoopWatchdog#
  EventLoopWatchdog._stall_threshold_s: EventLoopWatchdog#_stall_threshold_s.
  EventLoopWatchdog._check_interval_s: EventLoopWatchdog#_check_interval_s.
  EventLoopWatchdog._snapshot_provider: EventLoopWatchdog#_snapshot_provider.
  EventLoopWatchdog._phase: EventLoopWatchdog#_phase.
  EventLoopWatchdog._last_reported_tick: EventLoopWatchdog#_last_reported_tick.
  logger: logger.
  EventLoopWatchdog._default_traceback_dumper: EventLoopWatchdog#_default_traceback_dumper().
  EventLoopWatchdog.__init__: EventLoopWatchdog#__init__().
---
# Module: [`python/sgl_jax/srt/disaggregation/decode_watchdog.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py)

## Classes
### `EventLoopWatchdog`
- def: [`python/sgl_jax/srt/disaggregation/decode_watchdog.py:34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L34)
- doc: Detects a stalled event loop and dumps diagnostics once per stall.
- signature: `class EventLoopWatchdog:`
- members:
  - `beat(self, phase: str)` — [`L65`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L65) — Mark the loop as alive and entering ``phase``. Hot-path cheap.
  - `check_once(self, now: float | None = None)` — [`L72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L72) — Single stall check. Returns True iff a stall was reported.
  - `enabled(self)` — [`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L62)
  - `start(self)` — [`L105`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L105)
  - `stop(self)` — [`L119`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L119)
- protocol/private: `__init__`[`L37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L37), `_beat_ts`[`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L52), `_check_interval_s`[`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L47), `_clock`[`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L49), `_default_traceback_dumper`[`L102`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L102), `_last_reported_tick`[`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L57), `_loop`[`L126`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L126), `_phase`[`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L51), `_report`[`L85`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L85), `_snapshot_provider`[`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L48), `_stall_threshold_s`[`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L46), `_stop`[`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L58), `_thread`[`L59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L59), `_tick`[`L53`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L53), `_traceback_dumper`[`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L50)
- uses (calls/refs, reference-scoped): [`logger`](decode_watchdog.md#logger)
- used by: [`install_disaggregation_wiring`](runtime.md#install_disaggregation_wiring)

## Module values
- `logger` — [`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/decode_watchdog.py#L31)

