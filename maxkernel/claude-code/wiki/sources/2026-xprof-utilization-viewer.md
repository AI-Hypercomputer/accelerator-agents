---
title: "XProf Utilization Viewer (docs)"
type: source
tags: [docs, profiler, xprof, utilization-viewer, tensorcore, mxu, dma, tpu]
created: 2026-04-22
updated: 2026-04-22
---

Utilization Viewer is a **TPU-only**, nightly-build XProf tool that surfaces per-chip, per-Tensor-Node utilization of execution units and DMA paths as four bar charts — effectively a hardware-counter-backed bird's-eye view of where the TPU's silicon was busy.

## Overview

The tool renders two types of bar charts per chip: **execution-unit utilization** (top two charts, one per Tensor Node) and **DMA-path bandwidth utilization** (bottom two charts, one per Tensor Node). Utilization = `achieved / peak`: achieved is derived from hardware performance counters for instruction or byte counts, and peak is the theoretical throughput for each unit or path. Currently only in nightly builds.

## Key claims

- Utilization Viewer is **TPU-only** and currently **nightly-only**.
- Utilization for execution units is defined as the **fraction of cycles the unit was busy** during the profiling period.
- Utilization for DMA paths is the **fraction of bandwidth (bytes/cycle) used** during the profiling period, derived from `NF_CTRL` counters.
- Per-unit peak throughput varies and is encoded in the formula's cycle divisor: scalar/vector ALUs = 2 inst/cycle, vector load/store = 1 inst/cycle, MXU/XU/RPU = 1 inst per 8 cycles.
- A TPU chip contains **2 Tensor Nodes** — this is why there are two charts per metric type.
- The tensor core is composed of a Core Sequencer (CS), Vector Programmable Unit (VPU), MXU, XU, and RPU; the last three are driven through the VPU.
- There are **14 DMA paths** among 7 source/destination nodes (HIB, HBM, IMem, SMem, BMem, VMem, ICI) per Tensor Node, with `BMem↔ICI` and `BMem↔VMem` sharing a counter reported as "BMem to ICI/VMem".
- A DMA to/from ICI is a DMA to/from **remote HBM or VMEM** on another chip; a DMA to/from HIB is to/from **host memory**.

## Key data points

### Execution unit formulas

| Unit | Numerator | Divisor (peak) | Throughput |
|---|---|---|---|
| Scalar Unit | `count_s0_instruction + count_s1_instruction` | `2 * cycles` | 2 inst/cycle |
| Vector ALUs | `count_v0_instruction + count_v1_instruction` | `2 * cycles` | 2 inst/cycle |
| Vector Stores | `count_vector_store` | `cycles` | 1 inst/cycle |
| Vector Loads | `count_vector_load` | `cycles` | 1 inst/cycle |
| Matrix Unit (MXU) | `count_matmul` | `cycles / 8` | 1 inst / 8 cycles |
| Transpose Unit (XU) | `count_transpose` | `cycles / 8` | 1 inst / 8 cycles |
| Reduction/Permutation (RPU) | `count_rpu_instruction` | `cycles / 8` | 1 inst / 8 cycles |

### Tensor core block diagram (from the doc)

- Core Sequencer (CS) → VPU.
- VPU ↔ MXU, XU, RPU (bidirectional).
- MXU/XU/RPU are accessed through the VPU, not directly.

### DMA path graph (from the doc)

Nodes: `HIB`, `HBM`, `IMem`, `SMem`, `BMem`, `VMem`, `ICI`.

Edges (DMA paths) in the diagram:

- `HIB ↔ HBM` (both directions)
- `HBM → IMem`, `HBM → SMem`, `HBM → BMem`, `HBM → VMem`
- `BMem → VMem`
- `BMem → ICI`, `VMem → ICI`, `ICI → VMem`
- `ICI → HBM`, `VMem → HBM`, `BMem → HBM`, `SMem → HBM`
- `HBM → ICI`

### DMA semantics

| Path class | Meaning |
|---|---|
| HBM↔HIB | Host memory ↔ device HBM |
| HBM↔{IMem, SMem, BMem, VMem} | Device HBM ↔ on-chip buffer |
| BMem↔ICI, VMem↔ICI | On-chip buffer ↔ remote chip via ICI |
| ICI↔HBM | Remote HBM over ICI |
| BMem↔VMem | On-chip SRAM-to-SRAM |

Note: `BMem↔VMem` and `BMem↔ICI` share a counter → reported as a combined "BMem to ICI/VMem" bar.

### UI behavior

- Hover over a bar → tooltip with `achieved` and `peak` amounts (instructions for execution units, bytes for DMA BW).
- Utilization % = `achieved / peak`.
- Four bars × two Tensor Nodes × two metric categories = the entire chip view in one screen.

## Techniques referenced

- Hardware-counter-based utilization (not rate-from-static-cost like Graph Viewer).
- Per-Tensor-Node decomposition of a TPU chip.
- Bandwidth accounting per physical DMA path (not per software allocation).
- Shared-counter handling (BMem ICI/VMem reported combined).
- `NF_CTRL` counters as source of DMA telemetry.

## Gaps & caveats

- Nightly-only and TPU-only — expect the tool to change; GPU users do not get an equivalent here.
- Shared counters (BMem ICI/VMem) mean you cannot separately attribute traffic between the two destinations using this tool alone.
- Peak values are **theoretical**; they do not account for contention, power-gating, or thermal constraints.
- Utilization is averaged over the profiling period — short high-utilization bursts can be averaged down by long idle regions.
- Some counts are architecture-specific (`count_s0/s1`, `count_v0/v1`) — interpretation depends on TPU generation.
- Doc links out to `perf_counters.md` and the TPU system architecture for details, not included here.

## Connections

Concept slugs this source informs:

- `tensor-node` — the unit of decomposition (2 per chip).
- `mxu` — matrix unit; 1 inst / 8 cycles throughput.
- `vpu` — vector programmable unit; gateway to MXU/XU/RPU.
- `xu-transpose-unit` — transpose unit throughput.
- `rpu-reduction-permutation` — reduction/permutation unit.
- `scalar-unit` — 2-lane scalar unit on tensor core.
- `vector-alu` — 2-lane vector ALUs.
- `vector-load-store` — 1 inst/cycle throughput.
- `mxu-utilization` — first-class optimization target.
- `dma-paths` — 14-path per-Tensor-Node graph.
- `ici` — inter-chip interconnect: remote HBM / VMEM.
- `hib` — host-interface block (to/from host memory).
- `bmem` — on-chip buffer memory.
- `vmem` — on-chip vector memory.
- `smem` — on-chip scalar memory.
- `imem` — on-chip instruction memory.
- `nf-ctrl-counters` — source of DMA bandwidth telemetry.
- `hw-counter-utilization` — counter-based utilization vs. rate-derived.

## See also

- [xprof](../codebases/xprof.md)
- [xprof memory viewer](2026-xprof-memory-viewer.md)
- [xprof trace viewer](2026-xprof-trace-viewer.md)
- [xprof graph viewer](2026-xprof-graph-viewer.md)
- [xprof overview page](2026-xprof-overview-page.md)
- [xprof terminology](2026-xprof-terminology.md)

## Sources

- `raw/code/xprof/docs/utilization_viewer.md`
