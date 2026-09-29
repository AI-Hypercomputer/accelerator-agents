---
title: "XProf — Custom Call Profiling"
type: source
tags: [docs, profiler, xprof, custom-call, pallas, mosaic, llo, xla-flags]
created: 2026-04-22
updated: 2026-04-22
---

XProf documentation for **Custom Call Profiling**: how to make XLA custom calls (including Pallas and Mosaic kernels) visible in the Trace Viewer, with LLO (Low-Level Optimizer) utilization information. Off by default; enabled via two XLA flags. Critical for anyone writing or optimizing custom kernels on TPU.

## Overview

XLA custom calls invoke kernels not natively supported by XLA. Without additional flags, these show up as opaque boxes in the Trace Viewer. Two XLA flags enable (a) region tracing for custom calls and (b) LLO debug-info registration that lets XProf render a new **LLO utilization** line per TPU core or device for each custom call.

## Key claims

- Default behavior: custom calls are not richly traced in Trace Viewer.
- `--xla_enable_custom_call_region_trace=true` enables tracing for regions containing custom calls.
- `--xla_xprof_register_llo_debug_info=true` registers LLO debug info so XProf can display detailed utilization statistics for the custom call.
- Both flags are typically passed through `LIBTPU_INIT_ARGS` for TPU workloads.
- When flags are enabled, a new **LLO utilization** line appears in the Trace Viewer per TPU core / device executing the custom call.
- The LLO utilization line is the visualization primitive for **hardware-resource usage inside custom kernels** — the main affordance for bottleneck identification in Pallas/Mosaic kernels.
- Flags increase profile size and can slightly impact collection-time performance — use primarily for debugging and optimizing custom calls.
- If the LLO line does not appear, the compiler backend may not support registering LLO debug info for that particular custom call implementation.

## Key data points

### Flags

| Flag | Purpose |
|---|---|
| `--xla_enable_custom_call_region_trace=true` | Trace regions containing custom calls |
| `--xla_xprof_register_llo_debug_info=true` | Register LLO debug info → LLO utilization line in Trace Viewer |

### Example invocation

```shell
LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true" python your_jax_workload.py
```

### Trace Viewer artifact produced

| Artifact | Scope | Contents |
|---|---|---|
| LLO utilization line | Per TPU core / per device, per custom call | Visualization of HW resource usage during custom-call execution |

### Operational trade-offs

| Aspect | Effect |
|---|---|
| Profile size | Increased when flags enabled |
| Collection-time perf | Slight impact |
| Recommended usage | Debugging and optimizing custom calls only |
| Backend dependency | LLO info only if compiler backend supports registration for that custom call |

## Techniques referenced

- XLA custom calls (generic mechanism for non-XLA-native kernels).
- Pallas and Mosaic custom kernels (explicitly called out as typical users).
- LLO (Low-Level Optimizer) debug info registration.
- Trace Viewer as the surface for LLO utilization inspection.
- `LIBTPU_INIT_ARGS` as the flag-delivery path.

## Gaps & caveats

- The doc does not enumerate what "hardware resources" the LLO utilization line visualizes (e.g., MXU vs VPU vs scalar, VMEM read/write, DMA) — the exact axes must be read off the Trace Viewer or a separate LLO reference.
- Support is backend-dependent: absence of an LLO line is not necessarily a misconfiguration — it may be a backend limitation.
- The flags affect profile size and slightly perturb measurement; comparing custom-call timings with and without these flags is not apples-to-apples.
- No guidance on sampling frequency or granularity of LLO utilization vs normal Trace Viewer events.
- LLO utilization is Trace-Viewer-only here; there is no cross-reference to a table view.

## Connections

- `custom-call` — the mechanism being profiled.
- `pallas-kernel` / `mosaic-kernel` — explicit targets.
- `llo-utilization` — the new Trace Viewer line type.
- `trace-viewer` — surface for the visualization.
- `xla-flags` — delivery mechanism (`LIBTPU_INIT_ARGS`).
- `profile-overhead` — flag-induced size and timing impact.

## See also

- [xprof](../codebases/xprof.md)
- [XProf HLO Op Stats](2026-xprof-hlo-op-stats.md)
- [XProf HLO Op Profile](2026-xprof-hlo-op-profile.md)
- [XProf Framework Op Stats](2026-xprof-framework-op-stats.md)
- [XProf Perf Counters](2026-xprof-perf-counters.md)

## Sources

- `raw/code/xprof/docs/custom_call_profiling.md`
