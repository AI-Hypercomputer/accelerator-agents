---
title: "Source: training.md"
type: source-page
doc: training.md
updated: 2026-07-03
---
# training.md — Training on TPU (Parallelism 101)

Part 5. Derives when pure data parallelism and FSDP become communication-bound: both become
bandwidth-bound on TPU v5 when per-device batch size is below $2550 / M_X$ ($M_X$ = number of mesh
axes) ([fully-sharded-data-parallelism-fsdp](src:training.md#fully-sharded-data-parallelism-fsdp)).
Shows tensor parallelism becomes communication-bound when $Y > M_Y \cdot F / 2550$ (typically 8-16
way for most models) ([tensor-parallelism](src:training.md#tensor-parallelism)). Derives the optimal
FSDP degree $X_{opt} = \sqrt{(B/F)(M_X/M_Y)N}$ and shows combining FSDP with tensor parallelism
lets the model handle a per-chip batch as low as $2550^2/2F$ — about 8x smaller than FSDP alone
([combining-fsdp-and-tensor-parallelism](src:training.md#combining-fsdp-and-tensor-parallelism)).
Introduces context/sequence parallelism as "just another kind of data parallelism" for the MLP,
requiring ring attention for the attention block's cross-sequence computation
([what-do-we-mean-by-scaling](src:training.md#what-do-we-mean-by-scaling)).

## Feeds
- [Training parallelism strategies](../topics/training-parallelism-strategies.md) — the core topic
  this chapter defines.
- [Sharding notation and collective communication](../topics/sharding-notation-and-collectives.md) —
  the collective-cost formulas this chapter's thresholds are derived from.
