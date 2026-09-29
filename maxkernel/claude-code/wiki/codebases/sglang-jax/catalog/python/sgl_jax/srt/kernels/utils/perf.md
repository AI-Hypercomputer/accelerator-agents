---
title: 'Module: python/sgl_jax/srt/kernels/utils/perf.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/utils/perf.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.utils.perf`/
symbols:
  multiple_iteration_timeit_from_trace: multiple_iteration_timeit_from_trace().
  _extract_marker_durations_ms: _extract_marker_durations_ms().
  MARKER: MARKER.
  _load_trace: _load_trace().
---
# Module: [`python/sgl_jax/srt/kernels/utils/perf.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/utils/perf.py)

## Functions
- `_extract_marker_durations_ms(trace: dict[str, Any], task: str | None = None)` — [`L18`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/utils/perf.py#L18)
- `_load_trace(trace_root: str)` — [`L66`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/utils/perf.py#L66)
- `multiple_iteration_timeit_from_trace(compute_func, data_generator, task: str, tries: int = 5, trace_root: str = "/tmp/sglang_jax_moe_trace")` — [`L78`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/utils/perf.py#L78) — Profile multiple iterations and pull per-iteration kernel time from trace.

## Module values
- `MARKER` — [`L15`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/utils/perf.py#L15)

