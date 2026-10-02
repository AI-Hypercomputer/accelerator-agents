---
title: "FSDP (Fully Sharded Data Parallelism)"
type: concept
tags: [stub, parallelism]
created: 2026-04-22
updated: 2026-04-22
sources: 1
---

Shards optimizer states, gradients, and parameters across data-parallel replicas; uses all-gather + reduce-scatter on ICI.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Tensor Parallelism](tensor-parallelism.md)
- [Sharding (GSPMD)](sharding.md)
- [ICI (Inter-Chip Interconnect)](ici.md)
- [All-Gather](all-gather.md)
- [Collective Communication](collective-communication.md)

## Sources

- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — ZeRO-1/2/3 progression; ZeRO-3 = FSDP pays 1 all-gather per layer fwd + 1 all-gather + 1 reduce-scatter per layer bwd; on TPU this is emergent from PartitionSpec over an fsdp axis, not a stage dial
