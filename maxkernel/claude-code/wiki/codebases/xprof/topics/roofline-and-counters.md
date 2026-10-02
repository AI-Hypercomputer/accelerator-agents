---
title: "Roofline & performance counters"
type: topic
provenance: docs
updated: 2026-07-03
status: fresh
---
# Roofline & performance counters

Tools and options for understanding whether ops are compute- or memory-bound and for reading low-level hardware counters (compute, memory, PCIe/DCN).

## Tools & options
- **Roofline model** — plots ops against the compute and bandwidth roofs (compute- vs memory-bound); see [Roofline model](../sources/roofline_model.md).
- **Utilization viewer** — device utilization over the timeline; see [Utilization viewer](../sources/utilization_viewer.md).
- **Perf counters** — low-level hardware counters ([how-to-use-the-tool](src:docs/perf_counters.md#how-to-use-the-tool)); see [Perf counters](../sources/perf_counters.md).
- **Kernel profiling** — fine-grained per-kernel counters and event-triggered sampling ([enabling-fine-grained-performance-counters](src:docs/kernel-profiling.md#enabling-fine-grained-performance-counters)); see [Kernel profiling](../sources/kernel-profiling.md).
- **Custom-call profiling** — make hand-written XLA custom calls visible, incl. the LLO utilization line ([enabling-custom-call-visibility](src:docs/custom_call_profiling.md#enabling-custom-call-visibility)); see [Custom-call profiling](../sources/custom_call_profiling.md).
- **DCN/PCIe profiling** — device-side PCIe utilization + host counters ([1-device-side-fw-pcie-utilization-counters](src:docs/dcn_pcie_profiling.md#1-device-side-fw-pcie-utilization-counters)); see [DCN/PCIe profiling](../sources/dcn_pcie_profiling.md).

## See also
- [Advanced profiler options](../sources/advanced_profiler_options.md), [Multi-slice collectives](multislice-collectives.md)

