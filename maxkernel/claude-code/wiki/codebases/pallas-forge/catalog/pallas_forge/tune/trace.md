---
title: 'Module: pallas_forge/tune/trace.py'
type: catalog
provenance: extracted
module: pallas_forge/tune/trace.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `pallas_forge.tune.trace`/capture_
symbols:
  capture_top_n_traces: top_n_traces().
  capture_xprof_trace: xprof_trace().
  capture_top_n_traces.run_kernel: top_n_traces().run_kernel().
---
# Module: [`pallas_forge/tune/trace.py`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/trace.py)

## Functions
- `capture_top_n_traces(results: list, kernel_fn: Callable, input_fn: Callable, *, n: int = 3, output_dir: str = "/tmp/xprof_traces")` — [`L56`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/trace.py#L56) — Capture XProf traces for the top-n fastest configurations.
- `capture_xprof_trace(fn: Callable, *args, output_dir: str = "/tmp/xprof_traces", trace_name: str | None = None, n_calls: int = 5)` — [`L16`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/trace.py#L16) — Capture an XProf trace of a function execution.
- `run_kernel()` — [`L83`](../../../../../../raw/code/pallas-forge/pallas_forge/tune/trace.py#L83)

