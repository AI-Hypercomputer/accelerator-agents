---
title: "Scan Over Layers"
type: concept
tags: [stub, optimization, compile-time]
created: 2026-04-22
updated: 2026-04-22
sources: 1
---

`jax.lax.scan` / torchprime `scan_layers` pattern that folds transformer layers into a scan; compile time O(N)→O(1) with a backward sharding gotcha.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Rematerialization](rematerialization.md)
- [Sharding (GSPMD)](sharding.md)
- [Mark-Step Sync](mark-step-sync.md)
- [Training Memory Budget](training-memory-budget.md)

## Sources

- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
