---
title: "XProf — Framework Op Stats Tool"
type: source
tags: [docs, profiler, xprof, framework-op, jax, tensorflow, host-device]
created: 2026-04-22
updated: 2026-04-22
---

XProf documentation for the **Framework Op Stats** tool: per-op statistics at the framework level (JAX, TensorFlow, etc.), split across host and accelerator. Where HLO Op Stats operates below the compiler, this tool answers "which framework-visible op is expensive?" — the right level when mapping cost back to user code, or diagnosing host-side bottlenecks.

## Overview

The primary component is a table with one row per distinct framework op, defaulting to a sort by total self time (rank). A summarized-charts section shows per-category and per-op pie charts, separately for accelerator and host. A summarized-tables section gives per-category occurrence counts and total time. A drop-down toggles whether idle time is included in summary charts/tables. The whole table is exportable as CSV (with an optional pretty-print toggle).

Op-type is derived from the op name string (last part of the call stack) for easy sorting. Op-name is derived from framework-level metadata passed to XProf by XLA. Search boxes filter by host/device, op type, or op name.

## Key claims

- Framework Op Stats splits **host** and **device** cleanly — separate pie charts, separate summary tables, and an "Op execution location" column.
- Default rank is total self time, same convention as HLO Op Stats.
- Total time includes children; Total self time excludes children — the same fusion-style distinction applies at the framework level.
- Idle time is a first-class toggle for summary views.
- CSV export is supported, enabling offline/derived analysis.
- Framework-level metadata flow: XLA passes it along to XProf, and op-type is parsed from the call-stack tail.

## Key data points

### Columns in the framework operation statistics table

| Column | Definition |
|---|---|
| Op execution location | host or device |
| Framework op type | Derived from the last part of the call stack |
| Framework op name | From framework metadata relayed by XLA |
| Occurrences | Executions during the profile |
| Total time (μs) | Cumulative incl. children |
| Average time (μs) | Total / occurrences, incl. children |
| Total self time (μs) | Cumulative, excl. children |
| Average self time (μs) | Self / occurrences |
| Total self time on Device (%) | Self-time share of total device time |
| Cumulative total self time on device (%) | Running sum in sort order |
| Total self time on host (%) | Self-time share of total host time |
| Cumulative total self time on host (%) | Running sum in sort order |

### Summary sections

| Component | What it shows |
|---|---|
| Accelerator pie: time by category | Framework-op-category share on device |
| Host pie: time by category | Framework-op-category share on host |
| Top-N ops pie | Individual framework ops; rest as "Other" |
| Accelerator summary table | Category × occurrences × total time (abs + %) |
| Host summary table | Category × occurrences × total time (abs + %) |
| Idle-time toggle | Include/exclude idle in pies and summary |
| Export as CSV | Dumps the operation table |

## Techniques referenced

- Host/device time-attribution split.
- Framework-to-HLO cross-referencing (same op name thread runs through HLO Op Stats' "Framework op name" column).
- Idle-time accounting.

## Gaps & caveats

- "Framework op name" depends on metadata being relayed correctly by XLA; some generated or inlined ops may report terse or generic names.
- Op-type parsing from the call-stack tail is heuristic — ops with unusual naming may sort into unexpected categories.
- Summary pies top-N ops and bucket the rest as "Other"; a long tail of small but collectively large ops can be masked.
- Host time here is framework-level host work, not all host time: dispatch, input pipeline, and background Python can be present or absent depending on instrumentation.
- CSV export is useful, but there is no indication in the doc of a stable schema version — external scripts should guard against column-order changes.

## Connections

- `framework-op` — the unit measured.
- `host-vs-device-time` — the split the tool exposes.
- `jax-dispatch` / `tf-op` — framework provenance.
- `idle-time` — toggleable in summaries.
- `call-stack-attribution` — op-type derivation.

## See also

- [xprof](../codebases/xprof.md)
- [XProf HLO Op Stats](2026-xprof-hlo-op-stats.md)
- [XProf HLO Op Profile](2026-xprof-hlo-op-profile.md)
- [XProf Perf Counters](2026-xprof-perf-counters.md)
- [XProf Custom Call Profiling](2026-xprof-custom-call-profiling.md)

## Sources

- `raw/code/xprof/docs/framework_op_stats.md`
