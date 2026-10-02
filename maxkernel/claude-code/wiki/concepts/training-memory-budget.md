---
title: "Training Memory Budget"
type: concept
tags: [stub, training, memory]
created: 2026-04-22
updated: 2026-04-22
sources: 1
---

Rule of thumb for bf16 parameters + AdamW optimizer: ~16 bytes/param (2 param + 2 grad + 4 m + 4 v + 4 master).

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Rematerialization](rematerialization.md)
- [Host Offload](host-offload.md)
- [HBM (High-Bandwidth Memory)](hbm.md)
- [Dtype Strategy](dtype-strategy.md)
- [FSDP (Fully Sharded Data Parallelism)](fsdp.md)

## Sources

- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
