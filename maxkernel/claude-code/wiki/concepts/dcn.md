---
title: "DCN (Data Center Network)"
type: concept
tags: [stub, hardware, interconnect]
created: 2026-04-22
updated: 2026-04-22
sources: 4
---

Inter-slice network connecting TPU slices; lower bandwidth than ICI, used for multi-slice (megascale) collectives.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [ICI (Inter-Chip Interconnect)](ici.md)
- [Megascale](megascale.md)
- [Multi-slice](multi-slice.md)
- [Collective Communication](collective-communication.md)

## Sources

- [xprof Megascale Stats](../sources/2026-xprof-megascale-stats.md) — `raw/code/xprof/docs/megascale_stats.md`
- [xprof Megascale Viewer](../sources/2026-xprof-megascale-viewer.md) — `raw/code/xprof/docs/megascale_viewer.md`
- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — IB / RoCE GPU analogue; DCN is the cliff for cross-pod all-to-all / PP; EP and PP should be placed across DCN only when unavoidable
