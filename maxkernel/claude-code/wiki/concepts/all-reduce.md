---
title: "All-Reduce"
type: concept
tags: [stub, parallelism, communication]
created: 2026-04-22
updated: 2026-04-22
sources: 2
---

Collective op summing tensors across replicas and broadcasting the result.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [All-Gather](all-gather.md)
- [Collective Communication](collective-communication.md)
- [Tensor Parallelism](tensor-parallelism.md)
- [Async Collectives](async-collectives.md)
- [Send/Recv-Done Quartet](send-recv-done.md)

## Sources

- [xprof HLO Op Stats](../sources/2026-xprof-hlo-op-stats.md) — `raw/code/xprof/docs/hlo_op_stats.md`
- [JAX HuggingFace Part 2](../sources/2026-jax-huggingface-part-2.md) — Megatron column/row TP inserts exactly one all-reduce per attention block and one per MLP block; attention-O and MLP-down are the row-parallel matmuls
