---
title: "XProf (OpenXLA profiler + TensorBoard plugin)"
type: codebase
tags: [profiler, xplane, hlo, tpu, openxla, tensorboard, trace-viewer, roofline]
commit: 2e33c0119d9ccaba7630e193bc7bfa014ce64550
repo: https://github.com/openxla/xprof
created: 2026-04-22
updated: 2026-04-22
---

> **Grounded front door → [xprof overview](xprof/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

XProf is OpenXLA's performance profiler for JAX, PyTorch/XLA, and TensorFlow workloads — the primary profiler whose outputs (`.xplane.pb` traces, HLO dumps, and derived tool views) every experiment in this wiki consumes. This page describes the *tool's structure*: where profile formats live, how metrics are computed, and what the CLI/plugin surface looks like. Deep concept docs under `raw/code/xprof/docs/` will be ingested separately as `source` pages.

## Overview

XProf is a two-layer system:

1. A **C++ analysis core** (under `xprof/`) that reads the low-level `XSpace`/`XPlane` trace protobuf emitted by the XLA profiler and converts it into higher-level protobufs (`OpStats`, `InputPipeline`, `RooflineModelDatabase`, `MemoryProfile`, etc.) consumed by each tool view.
2. A **Python plugin + Angular frontend** (under `plugin/` and `frontend/`) that exposes the analysis core as (a) a TensorBoard plugin and (b) a standalone web server (`xprof` CLI), with one "tool" per view (trace viewer, overview, op profile, roofline, memory, HLO stats, megascale, smart suggestion, etc.).

The plugin also supports **remote profile capture** via a gRPC client to a running JAX/TF/XLA process, and a **distributed aggregator/worker** mode for large profiles.

## Architecture

```
+---------------------------------------------------------------+
|  Frontend (Angular, frontend/app)                              |
|   one component per tool: overview, trace_viewer,              |
|   roofline_model, memory_viewer, op_profile, hlo_stats, ...    |
+-------------------------------^-------------------------------+
                                | HTTP (JSON / gzipped)
+-------------------------------v-------------------------------+
|  Python plugin (plugin/xprof/profile_plugin.py)                |
|   routes: /data, /runs, /hosts, /capture_profile, /module_list,|
|           /generate_cache, /config, /version ...               |
|   wraps C++ analyzers via pybind (xprof/pywrap)                |
+-------------------------------^-------------------------------+
                                | pywrap (xspace_to_tools_data)
+-------------------------------v-------------------------------+
|  C++ analysis core (xprof/convert, xprof/utils)                |
|   XSpace/XPlane  ->  OpStats  ->  per-tool DataTables / protos |
|   + HLO proto -> graph / memory / roofline records             |
+---------------------------------------------------------------+
```

Boundaries:
- `XSpace` / `XPlane` input is defined upstream in XLA (`tsl::profiler`); XProf treats it as immutable input.
- `plugin/xprof/protobuf/` holds the **output schemas** every tool renders from.
- The pybind boundary lives at `xprof/pywrap/profiler_plugin_c_api.{cc,h}` and is surfaced in Python as `xprof.convert._pywrap_profiler_plugin`.

## Key abstractions

- **XSpace / XPlane**: the raw profile proto (upstream XLA type). Files land as `<host>.xplane.pb` under `<logdir>/plugins/profile/<session>/`.
- **OpStats** (`plugin/xprof/protobuf/op_stats.proto`): the central per-run aggregate — op metrics DB, step DB, run environment, perf env, kernel stats. Produced by `ConvertXSpaceToOpStats` in `xprof/convert/xplane_to_op_stats.{h,cc}`.
- **Tool**: a named view. The canonical list is `XPLANE_TOOLS` in `plugin/xprof/profile_plugin.py:129-148` — `trace_viewer`, `trace_viewer@` (streaming), `overview_page`, `input_pipeline_analyzer`, `framework_op_stats`, `kernel_stats`, `memory_profile`, `pod_viewer`, `op_profile`, `hlo_stats`, `roofline_model`, `inference_profile`, `memory_viewer`, `graph_viewer`, `megascale_stats`, `perf_counters`, `utilization_viewer`, `smart_suggestion`.
- **Processor**: per-tool C++ class (`*_processor.{h,cc}` in `xprof/convert/`) implementing `ProfileProcessor` / `UnifiedProfileProcessor` from `profile_processor.h` and `unified_profile_processor.h`. Registered via `profile_processor_factory.cc` / `unified_profile_processor_factory.cc`.
- **DataTable**: gviz-style table (`xprof/convert/data_table_utils.{h,cc}`) that most tools emit as their JSON payload.
- **Repository** (`xprof/convert/repository.{h,cc}`): abstracts where `.xplane.pb` files live (local FS or GCS via `file_utils_gcs.cc`).
- **Smart Suggestion rules** (`xprof/convert/smart_suggestion/`): named bottleneck classifiers (compute/memory/collective/input/SparseCore/TensorCore-idle/host-processing/data-transfer/data-shuffle bound) that fire recommendations.

## Entry points

| Surface | Entry point | Notes |
|---|---|---|
| CLI | `xprof` console script → `xprof.server:main` (`plugin/xprof/server.py`) | flags: `--logdir/-l`, `--port/-p`, `--grpc_port/-gp`, `--worker_service_address/-wsa`, `--hide_capture_profile_button/-hcpb`. |
| TensorBoard plugin | `tensorboard_plugins` entry point `profile = xprof.profile_plugin_loader:ProfilePluginLoader` | declared in `plugin/setup.py:88-95`. |
| HTTP routes | `ProfilePlugin.get_plugin_apps()` in `plugin/xprof/profile_plugin.py:820-870` | routes constants at `profile_plugin.py:71-93`. |
| Remote capture | `POST /capture_profile` → `capture_route_impl` at `profile_plugin.py:1431-1492` | calls `_pywrap_profiler_plugin.trace(service_addr, logdir, worker_list, True, duration_ms, retries, options)`. |
| Tool data fetch | `GET /data?run=&host=&tag=<tool>&...` → `ProfilePlugin.data_impl` → `xprof.convert.raw_to_tool_data.xspace_to_tool_data` (`plugin/xprof/convert/raw_to_tool_data.py`). |
| Distributed worker | `xprof --grpc_port=... --port=... --hide_capture_profile_button` | gRPC server in `plugin/xprof/worker/grpc_server.{h,cc}`, service at `worker_service.{h,cc}`. |
| Frontend bootstrap | `frontend/main.ts` → `AppModule` (`frontend/app/app_module.ts`). |

## Dependencies

- **Python runtime** (`plugin/setup.py`): `werkzeug`, `cheroot` (standalone WSGI), `etils[epath]` (GCS-aware paths), `fsspec`, `gcsfs`, `gviz_api`, `protobuf`. TensorFlow is **optional** (guarded import in `profile_plugin.py:57-65`) and is only needed for TPU-name resolution and some remote-capture paths.
- **C++ core**: XLA/TSL (`tsl::profiler`), Abseil, protobuf; `hlo_proto_to_graph_view` uses GraphViz (`graphviz_helper.h`).
- **Frontend**: Angular + Material, Plotly, Perfetto (trace viewer v2 is a Perfetto iframe; CSP allows `https://ui.perfetto.dev` in `profile_plugin.py:354-357`).
- **Build**: Bazel (`BUILD`, `WORKSPACE`, `config.bzl`, `defs/`), pip wheel built via `plugin/build_pip_package.sh`.

## Notable files

- `README.md` — CLI args, logdir layout (`<log_dir>/plugins/profile/<session>/*.xplane.pb`), distributed profiling setup.
- `docs/` — 18 markdown concept guides (capturing, trace viewer, roofline, memory, HLO op stats, megascale, JAX/PyTorch-XLA profiling, perf counters, utilization viewer, etc.). **Ingested separately as `source` pages** — do not duplicate here; back-link from those pages.
- `plugin/xprof/profile_plugin.py` (1900 lines) — the Python HTTP surface; route table, tool dispatch, cache, capture, CSV export.
- `plugin/xprof/server.py` — standalone webserver (cheroot WSGI) and CLI flag parsing.
- `plugin/xprof/profile_io.py` — `.xplane.pb` discovery and GCS/local filesystem abstraction.
- `plugin/xprof/convert/raw_to_tool_data.py` — Python-side tool dispatch; formats the `params` dict passed to each C++ processor.
- `plugin/xprof/protobuf/*.proto` — **all tool output schemas** (see below).
- `xprof/convert/` — the C++ analysis core (~150 files). One `<tool>_processor.{h,cc}` per tool.
- `xprof/utils/` — shared analyzers: HLO cost analysis, roofline, op metrics DB, kernel stats, step intersection, host-offload utils, TPU counter id tables (`tpu_counter_ids_v7.h`, `tpu_counter_ids_v7x.h`).
- `frontend/app/components/<tool>/` — one Angular component per tool view.
- `plugin/xprof/demo/profile_demo.py` — synthetic profile for UI development.
- `install_and_run.py` — legacy pip+tensorboard bootstrap script (pre-`xprof` console entry point).

## Performance-relevant surfaces

These are the files a hypothesis-writer will most often touch or quote when reasoning about TPU performance profiles. Each one defines either a metric, an I/O format, or a configurable knob.

1. **Tool output schemas — `plugin/xprof/protobuf/`.** The *canonical vocabulary* for every metric a hypothesis can cite. Hot files:
   - `op_stats.proto` — `OpStats`, `RunEnvironment`, `PerfEnv`, `OpMetricsDb`, `PrecisionStats`.
   - `op_metrics.proto` — per-op time, flops, bytes accessed, self-time, occurrences.
   - `input_pipeline.proto` + `tpu_input_pipeline.proto` — step-time breakdown (compute vs. infeed vs. outfeed vs. all-reduce etc.).
   - `steps_db.proto` — per-step event spans and HLO metrics DB.
   - `roofline_model.proto` — roofline records with `operational_intensity`, `measured_flop_rate`, `hbm_bw`, `cmem_bw`, etc.
   - `overview_page.proto` — summary metrics + recommendation text shown on the first page users see.
   - `memory_profile.proto`, `memory_viewer_preprocess.proto` — allocation timeline and peak usage breakdown.
   - `hlo_stats.proto`, `kernel_stats.proto`, `pod_stats.proto`, `pod_viewer.proto`, `inference_stats.proto`.
   - `dcn_slack_analysis.proto`, `dcn_collective_info.proto` — megascale / cross-slice DCN collective slack.
   - `power_metrics.proto`, `source_info.proto`, `smart_suggestion.proto`.

2. **`xplane -> OpStats` pipeline — `xprof/convert/xplane_to_op_stats.{h,cc}` (708 lines in `.cc`) and `op_stats_combiner.{h,cc}`.** `ConvertXSpaceToOpStats(space, OpStatsOptions)` is the fan-in every downstream TPU metric depends on. Knobs on `OpStatsOptions` (`xplane_to_op_stats.h:30-35`): `maybe_drop_incomplete_steps`, `generate_op_metrics_db`, `generate_step_db`, `generate_kernel_stats_db`. `MakePerfEnv` / `GetPerfEnvFromXPlane` (`xplane_to_op_stats.h:53-57`) populate the peak-FLOPS and peak-BW numbers the roofline uses — any mis-attribution of MFU ultimately traces back here.

3. **Overview metrics + recommendation thresholds — `xprof/convert/op_stats_to_overview_page.{h,cc}`.** Defines the headline numbers the Overview page reports and the thresholds (`xprof/convert/op_stats_to_overview_page.h:38-48`): `kTfFunctionReportThresholdInPercent = 20`, `kEagerReportThresholdInPercent = 10`, `kOutsideCompilationThresholdInPercent = 5`. `TpuPerformanceLimits` / `ComputeTpuAnalysisResult` compute the TPU MFU / HBM BW utilization that ranks most experiments. Any hypothesis citing "MFU" should reference this.

4. **Roofline model — `xprof/convert/op_stats_to_roofline_model.{h,cc}` + `xprof/utils/roofline_model_utils.{h,cc}` + `xprof/utils/hlo_cost_analysis_wrapper.{h,cc}`.** Computes per-HLO operational intensity, achieved FLOP rate, and the memory-bound vs. compute-bound classification. `include_infeed_outfeed` and `apply_time_scale_multiplier` flags on `ConvertOpStatsToRooflineModel` are directly configurable from the UI/URL params.

5. **Step time breakdown — `xprof/convert/op_stats_to_input_pipeline_analysis.{h,cc}` + `xprof/utils/tpu_step_breakdown_utils.h` + `xprof/utils/tpu_step_details_utils.h`.** Classifies each slice of a step (TensorCore busy, SparseCore busy, HBM waits, collectives, host-to-device, infeed, etc.). This is the file behind every "step time is dominated by X" claim.

6. **Kernel / op-profile / HLO-stats — `xprof/convert/op_stats_to_op_profile.{h,cc}`, `op_stats_to_hlo_stats.{h,cc}`, `xplane_to_kernel_stats_db.{h,cc}` + `xprof/utils/kernel_stats_utils.{h,cc}` + `xprof/utils/xla_op_utils.{h,cc}`.** Source of the per-HLO tables the MCP `get_top_hlo_ops` / `get_op_profile` tools surface.

7. **Memory — `xprof/convert/xplane_to_memory_profile.{h,cc}`, `memory_viewer_processor.{h,cc}`, `hlo_proto_to_memory_visualization_utils.{h,cc}` + `xprof/utils/host_offload_utils.{h,cc}`.** Peak HBM, allocation timeline, host-offload classification.

8. **Megascale / DCN — `xprof/convert/xspace_to_dcn_slack_analysis.{h,cc}`, `xplane_to_dcn_collective_stats.{h,cc}`, `dcn_analysis.{h,cc}`, `process_megascale_dcn.{h,cc}`, and `xprof/convert/megascale_perfetto/`.** Cross-slice collective slack — required for any multi-slice optimization hypothesis.

9. **Smart-suggestion rules — `xprof/convert/smart_suggestion/`.** Each `*_rule.h` is a named bottleneck classifier (`compute_bound_rule`, `memory_bound_rule`, `collective_bound_rule`, `input_bound_rule`, `host_processing_bound_rule`, `data_transfer_bound_rule`, `data_shuffle_bound_rule`, `sparse_core_bound_rule`, `sparse_core_offload_rule`, `tensor_core_idle_bound_rule`, `barrier_cores_rule`). Driven by `smart_suggestion_engine.{h,cc}`. These categories are a good prior for taxonomy of hypotheses.

10. **Trace viewer — `xprof/convert/trace_viewer/` + `xprof/convert/streaming_trace_viewer_processor.{h,cc}` + `xplane_to_trace_container.{h,cc}`.** Event filtering (`trace_events_filter_interface.h`), visibility decisions (`trace_viewer_visibility.{h,cc}`), and JSON rendering. Knobs are passed as `trace_viewer_options` through `raw_to_tool_data.py:116-120`.

11. **Capture-profile client — `profile_plugin.py:1431-1492`.** `POST /capture_profile?service_addr=&duration=&host_tracer_level=&device_tracer_level=&python_tracer_level=&delay=&num_retry=` is how an agent triggers an on-demand profile of a running JAX/TF/PyTorch-XLA job. Defaults: `host_tracer_level=2`, `device_tracer_level=1`, `python_tracer_level=0`, `duration_ms=1000`.

12. **Log-directory layout** (`README.md` + `plugin/xprof/profile_io.py`). `<logdir>/plugins/profile/<session>/<host>.xplane.pb` is **the** contract every experiment in this wiki must honor when writing profiles to `raw/profiles/`.

13. **TPU device caps — `xprof/utils/device_caps_utils.{h,cc}`, `hardware_type_utils.{h,cc}`, `tpu_counter_ids_v7.h`, `tpu_counter_ids_v7x.h`.** Per-generation peak FLOPs / HBM BW tables that drive MFU denominators.

## Connections

- Every `experiment` page in this wiki will link to a profile directory under `raw/profiles/<date>-<slug>/` that follows the XProf layout above, and will cite metrics from the protobufs listed in (1).
- Concept stubs to be created later: `xspace`, `op-stats`, `roofline-model`, `step-time-breakdown`, `hlo-op-stats`, `megascale-dcn-slack`, `smart-suggestion`, `trace-viewer`, `memory-profile`, `mfu`.
- The `docs/` markdown files will each become a `source` page and back-link to this codebase page.
- MCP tool layer: the agent has access to `mcp__xprof__*` tools (`get_overview`, `get_op_profile`, `get_memory_profile`, `get_top_hlo_ops`, `list_hlo_modules`, `get_hlo_dump`, `list_xplane_events`, `aggregate_xplane_events`, etc.) — these consume exactly the protos listed in the "Performance-relevant surfaces" section.

## See also

- [XProf docs: capturing profiles](../sources/2026-xprof-capturing-profiles.md)
- [XProf docs: trace viewer](../sources/2026-xprof-trace-viewer.md)
- [XProf docs: roofline model](../sources/2026-xprof-roofline-model.md)
- [XProf docs: JAX profiling](../sources/2026-xprof-jax-profiling.md)
- [XProf docs: PyTorch/XLA profiling](../sources/2026-xprof-pytorch-xla-profiling.md)

## Sources

- `raw/code/xprof/` at commit `2e33c0119d9ccaba7630e193bc7bfa014ce64550`
- `raw/code/xprof/README.md`
- `raw/code/xprof/plugin/setup.py`
- `raw/code/xprof/plugin/xprof/profile_plugin.py`
- `raw/code/xprof/plugin/xprof/server.py`
- `raw/code/xprof/plugin/xprof/convert/raw_to_tool_data.py`
- `raw/code/xprof/plugin/xprof/protobuf/`
- `raw/code/xprof/xprof/convert/xplane_to_op_stats.{h,cc}`
- `raw/code/xprof/xprof/convert/op_stats_to_overview_page.{h,cc}`
- `raw/code/xprof/xprof/convert/op_stats_to_roofline_model.{h,cc}`
- `raw/code/xprof/xprof/convert/smart_suggestion/`
- `raw/code/xprof/xprof/utils/`
- `raw/code/xprof/docs/` (to be ingested separately as `source` pages)
