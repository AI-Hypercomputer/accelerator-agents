---
title: "Async Collectives"
type: concept
tags: [stub, compiler, optimization]
created: 2026-04-22
updated: 2026-04-22
sources: 1
---

XLA flags that fuse and asynchronously schedule all-reduce / all-gather collectives so they overlap with compute.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Latency-Hiding Scheduler](latency-hiding-scheduler.md)
- [XLA Flags](xla-flags.md)
- [All-Gather](all-gather.md)
- [All-Reduce](all-reduce.md)
- [Send/Recv-Done Quartet](send-recv-done.md)

## Sources

- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — Appendix A3 compute/comm overlap math for DP/ZeRO-3/TP/PP; on GPU user-side knob is DDP bucket size, on TPU it's sharding annotations + XLA flags (bucketing is implicit)
