---
title: 'Module: python/sgl_jax/srt/disaggregation/req_time_stats.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/disaggregation/req_time_stats.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.disaggregation.req_time_stats`/
symbols:
  maybe_log_time_stats: maybe_log_time_stats().
  TimeStats: TimeStats#
  TimeStats.phases: TimeStats#phases().
  format_time_stats: format_time_stats().
  TimeStats.marks: TimeStats#marks.
  TimeStats.mark: TimeStats#mark().
  TimeStats.duration: TimeStats#duration().
  TimeStats.__init__: TimeStats#__init__().
  logger: logger.
  _PHASE_SPECS._PHASE_SPECS: _PHASE_SPECS._PHASE_SPECS.
  TimeStats.__slots__: TimeStats#__slots__.
---
# Module: [`python/sgl_jax/srt/disaggregation/req_time_stats.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/req_time_stats.py)

## Classes
### `TimeStats`
- def: [`python/sgl_jax/srt/disaggregation/req_time_stats.py:41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/req_time_stats.py#L41)
- doc: Lifecycle marks + derived phase durations for one request.
- signature: `class TimeStats:`
- members:
  - `duration(self, start: str, end: str)` — [`L56`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/req_time_stats.py#L56)
  - `mark(self, name: str)` — [`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/req_time_stats.py#L51) — Record the current time for ``name`` (first write wins).
  - `phases(self)` — [`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/req_time_stats.py#L63) — Role-specific phase durations, skipping any with unset endpoints.
  - `marks` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/req_time_stats.py#L48)
- protocol/private: `__init__`[`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/req_time_stats.py#L46), `__slots__`[`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/req_time_stats.py#L44)
- uses (calls/refs, reference-scoped): [`_PHASE_SPECS`](req_time_stats.md#_PHASE_SPECS._PHASE_SPECS)
- used by: [`_pd_mark_time`](prefill.md#SchedulerDisaggregationPrefillMixin._pd_mark_time), [`_pd_mark_time`](decode.md#SchedulerDisaggregationDecodeMixin._pd_mark_time), [`maybe_log_time_stats`](req_time_stats.md#maybe_log_time_stats), [`format_time_stats`](req_time_stats.md#format_time_stats)

## Functions
- `format_time_stats(ts: TimeStats, *, req_id: str)` — [`L73`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/req_time_stats.py#L73)
- `maybe_log_time_stats(ts: TimeStats | None, *, req_id: str, enabled: bool)` — [`L84`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/req_time_stats.py#L84) — Log the phase breakdown when ``enabled`` and ``ts`` is present.

## Module values
- `_PHASE_SPECS` — [`L24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/req_time_stats.py#L24)
- `logger` — [`L18`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/disaggregation/req_time_stats.py#L18)

