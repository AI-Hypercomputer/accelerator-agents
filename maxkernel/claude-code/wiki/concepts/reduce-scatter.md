---
title: "Reduce-Scatter"
type: concept
tags: [stub, parallelism, communication]
created: 2026-04-22
updated: 2026-04-22
sources: 3
---

Collective op that reduces (sums) a tensor across replicas and distributes the result as equal shards, one per replica. Conceptually an all-reduce whose output is sharded rather than replicated. FSDP/ZeRO-3 pairs an all-gather on the forward pass with a reduce-scatter on the backward pass (one of each per layer). Sequence parallelism replaces the TP block's internal all-reduce with a reduce-scatter + all-gather pair of the same aggregate volume.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Collective Communication](collective-communication.md)
- [All-Gather](all-gather.md)
- [All-Reduce](all-reduce.md)
- [FSDP (Fully Sharded Data Parallelism)](fsdp.md)
- [Send/Recv-Done Quartet](send-recv-done.md)

## Sources

- [xprof HLO Op Stats](../sources/2026-xprof-hlo-op-stats.md) — `raw/code/xprof/docs/hlo_op_stats.md`
- [xprof Megascale Stats](../sources/2026-xprof-megascale-stats.md) — `raw/code/xprof/docs/megascale_stats.md`
- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — ZeRO-3 backward reduce-scatter + Appendix A0 collectives primer; sequence-parallelism replaces the TP all-reduce with reduce-scatter + all-gather
