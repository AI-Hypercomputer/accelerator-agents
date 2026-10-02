---
title: 'Module: ejkernel/ops/execution/profiler.py'
type: catalog
provenance: extracted
module: ejkernel/ops/execution/profiler.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.ops.execution.profiler`/Profil
symbols:
  Profiler.get_events_from_plane: er#get_events_from_plane().
  Profiler.profile_time_by_function_id: er#profile_time_by_function_id().
  ProfilingError: ingError#
  Profiler._tf_avail_cache: er#_tf_avail_cache.
  Profiler._tf_python_profiler_available: er#_tf_python_profiler_available().
  Profiler.max_events_per_profile: er#max_events_per_profile.
  Profiler.parse_profile_from_bytes: er#parse_profile_from_bytes().
  Profiler.find_device_plane_ids: er#find_device_plane_ids().
  Profiler._parse_stats: er#_parse_stats().
  Profiler._parse_event: er#_parse_event().
  Profiler._pattern: er#_pattern.
  Profiler: er#
  Profiler.prefix_filter: er#prefix_filter.
  Profiler._get_stat_value: er#_get_stat_value().
  Profiler.event_filter_regex: er#event_filter_regex.
  Profiler.min_duration_ns: er#min_duration_ns.
  Profiler.require_tf: er#require_tf.
  Profiler._find_children: er#_find_children().
  Profiler._sum_events: er#_sum_events().
  Profiler.__init__: er#__init__().
  Profiler.verbose: er#verbose.
---
# Module: [`ejkernel/ops/execution/profiler.py`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py)

## Classes
### `Profiler`
- def: [`ejkernel/ops/execution/profiler.py:70`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L70)
- doc: JAX profile capture and parsing with nested-event accounting.
- signature: `class Profiler:`
- members:
  - `__init__(self, *, prefix_filter: str = "jit_", event_filter_regex: str | None = None, min_duration_ns: float = 1000, max_events_per_profile: int | None = 10000, verbose: bool = False, require_tf: bool = False, silence_tf_cpp_logs: bool = True)` — [`L96`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L96) — Initialize the JAX profiler with filtering and processing options.
  - `_find_children(own_name: str, start_ps: int, end_ps: int, events_sorted: list[dict[str, Any]], starts_sorted: np.ndarray)` — [`L300`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L300) — Find all child events fully contained within a parent event's timespan.
  - `_get_stat_value(stat, metadata)` — [`L202`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L202) — Extract the actual value from a profile statistic object.
  - `_parse_event(cls, event, event_metadata, stat_metadata, line_name: str = "")` — [`L251`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L251) — Parse a single profile event into a structured dictionary.
  - `_parse_stats(cls, stats, stat_metadata)` — [`L232`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L232) — Parse all statistics from a profile event into a dictionary.
  - `_sum_events(events)` — [`L337`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L337) — Calculate total time covered by a collection of events using interval merging.
  - `_tf_python_profiler_available(self)` — [`L132`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L132) — Check once whether tensorflow.python.profiler.trace can be imported.
  - `find_device_plane_ids(p: Any, device_str: str)` — [`L173`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L173) — Find plane IDs corresponding to a specific device in profile data.
  - `get_events_from_plane(self, p: Any, plane_idx: int)` — [`L367`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L367) — Extract and process events from a specific execution plane.
  - `parse_profile_from_bytes(profile_bytes: bytes)` — [`L152`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L152) — Parse JAX profile data from serialized bytes.
  - `profile_time_by_function_id(self, timing_closure: Callable[[], None], platform: str, total_calls_number: int)` — [`L439`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L439) — Profile function execution times across multiple iterations with statistical analysis.
  - `event_filter_regex` — [`L120`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L120)
  - `max_events_per_profile` — [`L122`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L122)
  - `min_duration_ns` — [`L121`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L121)
  - `prefix_filter` — [`L119`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L119)
  - `require_tf` — [`L127`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L127)
  - `verbose` — [`L123`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L123)
- protocol/private: `_pattern`[`L125`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L125), `_tf_avail_cache`[`L128`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L128)
- uses (calls/refs, reference-scoped): [`ProfilingError`](profiler.md#ProfilingError)
- used by: [`tune`](tuning.md#FNAutotuner.tune), [`profiler`](tuning.md#FNAutotuner.profiler)

### `ProfilingError`  ·  implements/extends Exception
- def: [`ejkernel/ops/execution/profiler.py:64`](../../../../../../../raw/code/ejkernel/ejkernel/ops/execution/profiler.py#L64)
- doc: Exception raised when profiling operations fail.
- signature: `class ProfilingError(Exception):`
- used by: [`get_events_from_plane`](profiler.md#Profiler.get_events_from_plane), [`profile_time_by_function_id`](profiler.md#Profiler.profile_time_by_function_id), [`find_device_plane_ids`](profiler.md#Profiler.find_device_plane_ids), [`parse_profile_from_bytes`](profiler.md#Profiler.parse_profile_from_bytes)

