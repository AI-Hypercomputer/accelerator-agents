---
title: "XProf — HLO Op Stats Tool"
type: source
tags: [docs, profiler, hlo, xprof, op-stats, roofline]
created: 2026-04-22
updated: 2026-04-22
---

XProf documentation for the **HLO Op Stats** tool: a per-op tabular breakdown of every HLO operation executed during a profiling session, plus summary charts. This is the primary lens for finding the most time-consuming HLO ops and judging whether they are compute- or memory-bound on TPU (and GPU).

## Overview

HLO Op Stats surfaces every distinct HLO op that ran during the profile, ranked by **total self time**. Alongside time columns it carries bandwidth consumption (FLOPs, HBM, internal memories), operational intensity, and a roofline-based compute/memory-bound classification. Summary pie charts break time down by HLO category, individual op, rematerialization status, outside compilation, and (TPU only) by replica group for collectives.

Rates are derived by dividing static compiler cost-analysis numerators (FLOPs, bytes) by measured profile durations. Utilizations are those absolute rates divided by per-device peaks — and the peak is **precision-dependent** (e.g., TPU v6e bf16 vs int8 differ by 2x), so "Normalized GFLOPS/s" defaults to a bf16 baseline.

## Key claims

- The table is the workhorse; default sort is by total self-time rank.
- "Total time" includes children of fusions; "Total self time" excludes them — use self time to attribute cost correctly.
- Compute/Memory bound is a roofline classification using `FLOPs / bytes` (operational intensity) vs the device roofline.
- GPU HLO ops have an N:M relationship with executed kernels; kernel-level stats live in the separate GPU Kernel Stats tool.
- Rematerialization time comes from XLA compiler metadata attached to the profile, not a heuristic on op names.
- Outside compilation (TF feature) flags ops that transparently ran on host CPU instead of the accelerator.
- For TPU collectives, a replica-group drop-down breaks down an op (e.g., AllGather) across its instances — useful for spotting an imbalanced replica group dominating time.

## Key data points

### Columns in the HLO Operation Statistics table

| Column | Definition | Notes for hypothesis writers |
|---|---|---|
| Program ID | HLO module identifier | Filter to a specific module when multi-module |
| HLO Op category | Compiler-assigned; XProf adds heuristics (e.g., convolution fusions) | Use for filtering |
| HLO op name | Unique name from XLA | Stable handle for cross-referencing profiles |
| HLO op text | Shapes/types of inputs | Read this to see precisions and layouts |
| Framework op name | JAX/TF-level provenance | Maps HLO back to user code |
| Occurrences | Count of executions in the profile | Per-step cost = total / occurrences / steps |
| Total time (μs) | Cumulative incl. children | Includes fusion bodies |
| Avg. time (μs) | Total / occurrences, incl. children | |
| Total self time (μs) | Cumulative, excl. children | **Primary ranking metric** |
| Avg. self time (μs) | Self / occurrences | |
| Total self time (%) | Self time / device total | |
| Cumulative total self time (%) | Running sum in rank order | Identify the 80/20 point |
| DMA stall (%) | Fraction of op time stalled on DMA | High values signal memory-bound pathologies |
| Bandwidth consumption | Per-op usage/sec for FLOPs, HBM, and internal memories (e.g., CMEM on v4) | Static FLOPs/bytes ÷ measured time |
| Model GFLOPS/s | Compiler-computed FLOPs ÷ measured time | |
| Normalized GFLOPS/s | Adjusted for numerical precision peak; bf16 default | Compare across int8/bf16 mixes |
| Memory BW | Bytes/sec from any memory (VMEM, HBM, …) | |
| HBM BW | Bytes/sec specifically from HBM | |
| Operational intensity | FLOPs/byte | Roofline x-axis |
| Compute/Memory bound | Roofline classification | Primary optimization signal |
| Rematerialization | Op is part of remat | From compiler metadata |
| Outside compilation | Ran on host CPU | |
| Autotuned | Op was autotuned by XLA | See XLA autotuning / persisted autotuning |

### Summary charts

| Chart | What it shows |
|---|---|
| Time per HLO category | Fraction of time by category |
| Time per HLO operation | Top-N ops; rest bucketed as "Other" |
| Time spent on rematerialization | Remat share of total time |
| Time on rematerialization per HLO category | Where remat cost lands |
| Time spent on outside compilation | Host-fallback share |
| GFLOPS/s vs self time | Plot ordered by self time — flat low-FLOPs tail = optimization target |
| Time per HLO by replica group (TPU) | Distribution across collective instances |

## Techniques referenced

- Roofline analysis (operational intensity, compute- vs memory-bound).
- Rematerialization (marked via compiler metadata).
- Outside compilation (TF: transparently run ops on host).
- XLA autotuning and persisted autotuning.
- Collective ops and replica groups (AllGather referenced).

## Gaps & caveats

- All rates rest on **compiler static cost analysis** for FLOPs/bytes. If the compiler mis-models a custom op, rates are wrong — the profile's duration is still real.
- "Total time" including children can mislead when a fusion has an expensive inner op; always cross-check with self time.
- Normalized GFLOPS/s uses bf16 as the default peak — for int8-heavy workloads the raw number can understate utilization unless re-normalized.
- DMA stall (%) is a symptom, not a diagnosis — it doesn't distinguish HBM pressure from bad layout or prefetch.
- GPU: N:M HLO-to-kernel relationship means an HLO op's "self time" aggregates across possibly many kernels; drill down via GPU Kernel Stats.
- Per-replica-group breakdown is TPU-only.

## Connections

- `hlo-op` — the unit measured by this tool.
- `fusion` — children-vs-self time distinction hinges on fusion structure.
- `roofline` / `arithmetic-intensity` — the operational-intensity and compute-vs-memory-bound columns.
- `rematerialization` — remat-tagged rows and pie charts.
- `mxu-utilization` — Normalized GFLOPS/s feeds MXU/FLOPs utilization reasoning.
- `hbm-bandwidth` — HBM BW column.
- `cmem` — v4-only internal memory reported when present.
- `collective-ops` / `all-gather` — replica-group breakdown.
- `outside-compilation` — host-fallback ops.
- `autotuning` — Autotuned column.

## See also

- [xprof](../codebases/xprof.md)
- [XProf HLO Op Profile](2026-xprof-hlo-op-profile.md)
- [XProf Framework Op Stats](2026-xprof-framework-op-stats.md)
- [XProf Perf Counters](2026-xprof-perf-counters.md)
- [XProf Custom Call Profiling](2026-xprof-custom-call-profiling.md)

## Sources

- `raw/code/xprof/docs/hlo_op_stats.md`
