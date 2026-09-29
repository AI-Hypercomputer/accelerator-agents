---
title: "XProf docs: Profiling JAX computations"
type: source
tags: [docs, profiler, jax, tpu, gpu, profile-capture]
created: 2026-04-22
updated: 2026-04-22
---

The JAX-specific capture guide for XProf. Covers programmatic tracing with `jax.profiler.start_trace` / `stop_trace` / `trace` context manager, manual capture via the XProf UI talking to `jax.profiler.start_server(port)`, continuous profiling snapshots via the `xprof.api.continuous_profiling_snapshot` module, `jax.profiler.TraceAnnotation` for custom trace events, and the full `jax.profiler.ProfileOptions` surface (host/device/python tracer levels, TPU sparse-core tracing knobs, GPU CUPTI knobs). This is the primary reference for any JAX experiment in this wiki.

## Overview

Three things a JAX experiment needs to get right:

1. **Start a profile** — either programmatically around a known code region, or on-demand by starting an in-process gRPC profiler server and triggering from the XProf UI.
2. **Block for device completion** before `stop_trace` returns — JAX is asynchronously dispatched, so without `x.block_until_ready()` the trace will be missing on-device execution.
3. **Configure `ProfileOptions`** to enable/disable host, device, and python tracers, and (on TPU) pick the right `tpu_trace_mode` and sparse-core knobs.

XProf is the successor to `tensorboard_plugin_profile`; the "Profile" tab appears in TensorBoard whenever `xprof` is installed and TensorBoard is pointed at the same logdir. Standalone UI: `xprof --port=8791 <logdir>` → `http://localhost:8791/`.

## Key claims

- `jax.profiler.start_trace(<dir>)` + `jax.profiler.stop_trace()` is the canonical programmatic API; a `with jax.profiler.trace(<dir>):` context manager is equivalent.
- **Always call `block_until_ready()` before `stop_trace()`** — JAX async dispatch means computation may not be done at `stop_trace` time, causing missing on-device events.
- The manual capture flow requires `jax.profiler.start_server(<port>)` inside the program; the example port is **9999**. The capture is initiated from the XProf UI (`localhost:8791` → "CAPTURE PROFILE" → enter `localhost:9999`).
- Continuous profiling snapshots use `xprof.api.continuous_profiling_snapshot.{start_continuous_profiling, get_snapshot, stop_continuous_profiling}` — the profiler server must already be running.
- Custom trace labels use `jax.profiler.TraceAnnotation` / `jax.profiler.annotate_function`. Without them, the trace shows mostly low-level JAX internals.
- `ProfileOptions.host_tracer_level` defaults to `2`; `device_tracer_level` defaults to `1`; `python_tracer_level` defaults to `0`. Passing unrecognized keys or values to `advanced_configuration` raises `InvalidArgumentError`.
- On TPU, `tpu_trace_mode` defaults to `TRACE_ONLY_XLA` if unset.

## Key data points

### Tracer levels (`ProfileOptions` — general)

| Option | Value | Meaning |
|---|---|---|
| `host_tracer_level` | 0 | Disable host (CPU) tracing |
| | 1 | Only user-instrumented `TraceMe` events |
| | 2 *(default)* | + high-level exec details, expensive XLA ops |
| | 3 | + low-level / cheap XLA ops (verbose) |
| `device_tracer_level` | 0 | Disable device tracing |
| | 1 *(default)* | Enable device tracing |
| `python_tracer_level` | 0 *(default)* | Disable Python call tracing |
| | 1 | Enable Python tracing |

### TPU-specific options

| Option | Purpose |
|---|---|
| `tpu_trace_mode` | `TRACE_ONLY_HOST`, `TRACE_ONLY_XLA` *(default)*, `TRACE_COMPUTE`, `TRACE_COMPUTE_AND_SYNC` |
| `tpu_num_sparse_cores_to_trace` | # sparse cores to trace |
| `tpu_num_sparse_core_tiles_to_trace` | # tiles per sparse core to trace |
| `tpu_num_chips_to_profile_per_task` | Cap chips profiled per task (reduce trace size) |
| `tpu_cpu_perf_counter_profile_events` | Comma-separated PMU event names (e.g. `"context-switches,page-faults"`) |
| `tpu_cpu_perf_counter_configs` | Advanced: raw `config:type:name` triples mapping to Linux `perf_event_attr` fields |

### GPU-specific options (for completeness)

| Option | Default | Notes |
|---|---|---|
| `gpu_max_callback_api_events` | `2*1024*1024` | CUPTI callback event cap |
| `gpu_max_activity_api_events` | `2*1024*1024` | CUPTI activity event cap |
| `gpu_max_annotation_strings` | `1024*1024` | Annotation string cap |
| `gpu_enable_nvtx_tracking` | `False` | NVTX in CUPTI |
| `gpu_enable_cupti_activity_graph_trace` | `False` | CUDA graph tracing |
| `gpu_pm_sample_counters` | — | Comma-separated CUPTI PM metrics (disabled by default) |
| `gpu_pm_sample_interval_us` | `500` | PM sampling period (µs) |
| `gpu_pm_sample_buffer_size_per_gpu_mb` | `64` | Max 4096 (4 GB) |
| `gpu_num_chips_to_profile_per_task` | *(all)* | Cap GPUs profiled per task |
| `gpu_dump_graph_node_mapping` | `False` | Dump CUDA graph node map |

### Minimal capture snippets

Programmatic (API):
```python
import jax
jax.profiler.start_trace("/tmp/profile-data")
# ... work ...
y.block_until_ready()
jax.profiler.stop_trace()
```

Programmatic (context manager):
```python
with jax.profiler.trace("/tmp/profile-data"):
    # work; ends with block_until_ready()
    y.block_until_ready()
```

Manual / on-demand:
```python
import jax.profiler
jax.profiler.start_server(9999)   # keep alive; stop_server() when done
# In the XProf UI (localhost:8791) -> "CAPTURE PROFILE" -> "localhost:9999"
```

Continuous snapshot:
```python
from xprof.api import continuous_profiling_snapshot as cps
cps.start_continuous_profiling('localhost:9999', {})
cps.get_snapshot('localhost:9999', '/tmp/profile-data/')
cps.stop_continuous_profiling('localhost:9999')
```

Disable host/python tracing (minimize trace size):
```python
options = jax.profiler.ProfileOptions()
options.python_tracer_level = 0
options.host_tracer_level = 0
jax.profiler.start_trace("/tmp/profile-data", profiler_options=options)
```

Advanced (TPU example):
```python
options = jax.profiler.ProfileOptions()
options.advanced_configuration = {
    "tpu_trace_mode": "TRACE_ONLY_HOST",
    "tpu_num_sparse_cores_to_trace": 2,
}
```

## Techniques referenced

- **Programmatic capture** via `start_trace` / `stop_trace` or `trace` context manager.
- **In-process gRPC profiler server** (`jax.profiler.start_server(port)`) for on-demand capture.
- **Continuous profiling snapshots** via `xprof.api.continuous_profiling_snapshot`.
- **Custom trace annotations** (`jax.profiler.TraceAnnotation`, `jax.profiler.annotate_function`).
- **TPU tracing modes** — host-only / XLA-only / compute / compute-and-sync trade-offs.
- **Sparse-core and sparse-core tile tracing** for TPU SparseCore workloads.
- **CPU perf counter profiling** on TPU hosts (Linux `perf_event_attr`).
- **Trace size control** via chip-count caps and selective tracer disabling.

## Gaps & caveats

- **`block_until_ready` is mandatory** but easy to forget — a trace without it will show empty or misleading device regions. This is the single most common pitfall on JAX.
- No explicit guidance on **multi-host / multi-controller JAX** coordination of captures (e.g. ensuring all hosts capture the same window). The multi-host assumption is that each host writes its own `.xplane.pb` under the same logdir.
- `start_server(port)` must stay alive for the duration of any on-demand capture; abrupt process exit drops the trace. `stop_server()` exists but is not shown in examples.
- Specifying unknown keys/values in `advanced_configuration` returns `InvalidArgumentError` — fail-fast but the error site is late (inside `start_trace`).
- Doc lists example port `9999` for the JAX profiler server. This must match whatever is entered in the UI — no auto-discovery.
- **No TPU-specific environment-variable caveats** (e.g. `LIBTPU_INIT_ARGS`, `XLA_FLAGS`) are discussed here; they are orthogonal but commonly needed for reproducible perf runs.
- "Not all XProf profiling features are hooked up with JAX" — post-capture the UI may look empty initially; switch to Trace Viewer explicitly.
- TPU `tpu_trace_mode` default is `TRACE_ONLY_XLA`; `TRACE_COMPUTE_AND_SYNC` adds synchronization events but increases trace volume. The doc does not quantify the overhead.
- The continuous profiling API lives under `xprof.api`, not `jax.profiler` — easy to miss when searching JAX-first.

## Connections

- `profile-capture` — framework-agnostic umbrella.
- `jax-trace` — `jax.profiler.start_trace`/`stop_trace`/`trace`/`TraceAnnotation`.
- `jax-profiler-server` — `jax.profiler.start_server(port)` (in-process gRPC endpoint).
- `profile-options` — `jax.profiler.ProfileOptions` tracer levels and advanced config.
- `tpu-trace-mode` — host / XLA / compute / compute-and-sync.
- `sparse-core-tracing` — TPU SparseCore profiling knobs.
- `continuous-profiling-snapshot` — retroactive capture API.
- `block-until-ready` — JAX async-dispatch pitfall.
- `xprof-sessions` — multiple captures per run as separate sessions.

## See also

- [xprof](../codebases/xprof.md)
- [XProf docs: Capturing profiles](2026-xprof-capturing-profiles.md)
- [XProf docs: PyTorch/XLA profiling](2026-xprof-pytorch-xla-profiling.md)
- [XProf docs: TensorFlow profiling](2026-xprof-tensorflow-profiling.md)
- [XProf docs: Docker deployment](2026-xprof-docker-deployment.md)
- [XProf docs: Kubernetes deployment](2026-xprof-kubernetes-deployment.md)

## Sources

- `raw/code/xprof/docs/jax_profiling.md`
