---
title: "XProf — Perf Counters Tool"
type: source
tags: [docs, profiler, xprof, perf-counters, hardware-counters, nightly]
created: 2026-04-22
updated: 2026-04-22
---

XProf documentation for the **Perf Counters** tool: a tabular display of hardware performance counters collected during a profiling session, with filtering by host, chip, counter name, sample, and counter set. Currently available only in **nightly builds**. This is the lowest-level quantitative lens XProf exposes — the place to read raw HW counter values (for example, issue-slot counters) rather than compiler-derived rates.

## Overview

All accelerators (TPU and GPU) expose HW performance counters. The Perf Counters tool shows them in a filterable table. A typical profile contains a single counter sample taken at the end of the profiling period (after counters were cleared at the start); "Continuous Performance Counter Profile" mode produces multiple samples. By default only non-zero counters are shown.

## Key claims

- Tool availability: **nightly builds only** at the time of this doc.
- The table has four columns: row number (untitled), **Counter** (name), **Value (Dec)**, **Value (Hex)**.
- Default view filters out zero-valued counters.
- Profile sample semantics: counters are cleared at the start of the profiling period and sampled at the end (single sample typical).
- Continuous Performance Counter Profile produces multiple samples in one profile — selectable via the Sample filter.
- TPU filters: Host, Chip, Sample, Set (e.g., "issue" for issue counters), Counter text-substring.
- GPU filters: Host, Kernel (by computation fingerprint), Device, Counter text-substring.
- The Counter text filter is a substring match; typing any fragment narrows rows.

## Key data points

### Table columns

| # | Column | Meaning |
|---|---|---|
| 1 | (unnamed) | Row number |
| 2 | Counter | HW counter / metric name |
| 3 | Value (Dec) | Decimal numeric value |
| 4 | Value (Hex) | Hexadecimal numeric value |

### Filters

| Platform | Filter | Purpose |
|---|---|---|
| Both | Host | Limit to a host machine |
| Both | Counter (text) | Substring match on counter name |
| TPU | Chip | Limit to a specific chip |
| TPU | Sample | Pick a sample (multi-sample profiles only) |
| TPU | Set | E.g., "issue" — only the issue counter set |
| GPU | Kernel | GPU kernel computation fingerprint |
| GPU | Device | Device attached to the selected host |

### Sampling modes

| Mode | Samples per profile |
|---|---|
| Default | 1 (cleared at start, read at end) |
| Continuous Performance Counter Profile | Multiple |

## Techniques referenced

- Hardware performance counters as a ground-truth complement to compiler-derived rates.
- Counter sets (e.g., "issue") as a grouping mechanism for TPU.
- GPU kernel fingerprinting for per-kernel counter attribution.
- Continuous counter sampling within a single profile.

## Gaps & caveats

- Nightly-only means production profiles from stable releases may not include this data.
- The doc lists columns and filters but does **not** enumerate counter names, units, or meanings — counter semantics must come from the TPU/GPU reference or device-specific docs.
- Default non-zero filter hides counters that may be meaningful as "zero observed in this window" signals.
- Single-sample default is coarse: counters reflect the whole profiling window, not specific ops or phases; use Continuous mode to attribute to phases.
- TPU Chip filter provides per-chip breakdown but not per-core/per-tensor-core/per-SparseCore granularity — at least not documented here.
- Counter values are raw numbers; delta/rate interpretation is up to the analyst.

## Connections

- `perf-counters` — the core concept surfaced by this tool.
- `issue-counters` — the named TPU counter set example.
- `continuous-counter-profile` — multi-sample mode.
- `kernel-fingerprint` — GPU kernel identification.
- `hw-counters-vs-compiler-rates` — this tool's complement to static-analysis-derived rates elsewhere in XProf.

## See also

- [xprof](../codebases/xprof.md)
- [XProf HLO Op Stats](2026-xprof-hlo-op-stats.md)
- [XProf HLO Op Profile](2026-xprof-hlo-op-profile.md)
- [XProf Framework Op Stats](2026-xprof-framework-op-stats.md)
- [XProf Custom Call Profiling](2026-xprof-custom-call-profiling.md)

## Sources

- `raw/code/xprof/docs/perf_counters.md`
