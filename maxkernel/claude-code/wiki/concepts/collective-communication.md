---
title: "Collective Communication"
type: concept
tags: [stub, parallelism, communication]
created: 2026-04-22
updated: 2026-04-22
sources: 4
---

Cross-replica operations (all-reduce, all-gather, reduce-scatter, etc.) that synchronize or redistribute data.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [All-Gather](all-gather.md)
- [All-Reduce](all-reduce.md)
- [Send/Recv-Done Quartet](send-recv-done.md)
- [Async Collectives](async-collectives.md)
- [Megascale](megascale.md)

## Sources

- [xprof Megascale Stats](../sources/2026-xprof-megascale-stats.md) — `raw/code/xprof/docs/megascale_stats.md`
- [xprof HLO Op Stats](../sources/2026-xprof-hlo-op-stats.md) — `raw/code/xprof/docs/hlo_op_stats.md`
- [JAX HuggingFace Part 2](../sources/2026-jax-huggingface-part-2.md) — gSPMD inserts collectives implicitly from sharding annotations; one all-reduce per attention/MLP block under Megatron-style TP
- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — Appendix A0 collectives primer (broadcast, all-reduce, all-gather, reduce-scatter, ring-all-reduce); same primitives on TPU via `jax.lax.*` over a mesh, substrate is ICI not NCCL
