---
title: "Source: applied-training.md"
type: source-page
doc: applied-training.md
updated: 2026-07-03
---
# applied-training.md — Training LLaMA 3 on TPUs

Part 6. Applies the Part 4/5 formulas to a concrete model, LLaMA 3-70B, showing the 3 big MLP
weight matrices dominate parameter/FLOPs accounting (56B of 70B total params)
([counting-parameters-and-flops](src:applied-training.md#counting-parameters-and-flops)). Derives a
concrete sharding recipe — 1024-way data parallelism, 2-way sequence parallelism, 4-way tensor
parallelism on a full TPU v5p pod, chosen because pure FSDP or FSDP+sequence-parallelism alone would
be communication-bound at this batch size
([how-to-shard-llama-3-70b-for-training](src:applied-training.md#how-to-shard-llama-3-70b-for-training)).

## Feeds
- [Training parallelism strategies](../topics/training-parallelism-strategies.md) — the worked
  example this chapter contributes.
