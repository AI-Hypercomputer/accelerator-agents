---
title: "Dimension Alignment"
type: concept
tags: [stub, optimization]
created: 2026-04-22
updated: 2026-04-22
sources: 1
---

Rules for shape divisibility so MXU doesn't waste cycles: batch multiple of 64/1024, hidden multiple of 128/256, sharded dims respect per-chip (not global) alignment.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [MXU (Matrix Unit)](mxu.md)
- [MXU Utilization](mxu-utilization.md)
- [Sharding (GSPMD)](sharding.md)
- [Dtype Strategy](dtype-strategy.md)
- [Tensor Parallelism](tensor-parallelism.md)

## Sources

- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
