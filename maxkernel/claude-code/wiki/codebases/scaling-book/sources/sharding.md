---
title: "Source: sharding.md"
type: source-page
doc: sharding.md
updated: 2026-07-03
---
# sharding.md — Sharded Matrices and How to Multiply Them

Part 3. Introduces the named-axis sharding notation ($A[I_X, J_Y]$: array axis $I$ partitioned
across mesh axis $X$) used throughout the rest of the book
([a-unified-notation-for-sharding](src:sharding.md#a-unified-notation-for-sharding)). Derives the
four cases of sharded matrix multiplication and the communication each requires — no comms when
neither multiplicand has a sharded contracting dimension; an AllGather when one does; an AllReduce
when both do; and a forced AllGather when both have a non-contracting dimension sharded along the
same axis
([case-1-neither-multiplicand-has-a-sharded-contracting-dimension](src:sharding.md#case-1-neither-multiplicand-has-a-sharded-contracting-dimension)
through
[case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis](src:sharding.md#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis)).
Derives costs for AllGather/ReduceScatter/AllReduce/AllToAll over a bidirectional ICI ring, noting
AllToAll is 4x cheaper than AllGather since it doesn't replicate data
([a-deeper-dive-into-tpu-communication-primitives](src:sharding.md#a-deeper-dive-into-tpu-communication-primitives)),
and that AllGather and ReduceScatter are forward/backward-pass duals of each other
([more-about-the-reducescatter](src:sharding.md#more-about-the-reducescatter)).

## Feeds
- [Sharding notation and collective communication](../topics/sharding-notation-and-collectives.md) —
  the core topic this chapter defines.
- [Training parallelism strategies](../topics/training-parallelism-strategies.md) — applies these
  collective costs to derive FSDP/tensor-parallelism communication-bound thresholds.
