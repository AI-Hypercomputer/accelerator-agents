---
title: "DCN and PCIe profiling"
type: source
provenance: docs
source_doc: docs/dcn_pcie_profiling.md
updated: 2026-07-03
status: fresh
---
# DCN and PCIe profiling

How to profile data-center-network (DCN) and PCIe traffic: device-side FW PCIe utilization counters and host-side performance counters (via libpfm3 / perf_event_open).

## Key points
- Device-side FW PCIe utilization counters ([1-device-side-fw-pcie-utilization-counters](src:docs/dcn_pcie_profiling.md#1-device-side-fw-pcie-utilization-counters))
- Host-side host performance counters and CPU counter sampling interval ([2-host-side-host-performance-counters](src:docs/dcn_pcie_profiling.md#2-host-side-host-performance-counters))

## Feeds topics
- [Roofline & performance counters](../topics/roofline-and-counters.md)
- [Multi-slice collectives](../topics/multislice-collectives.md)

## Sources
- raw/code/xprof/docs/dcn_pcie_profiling.md
