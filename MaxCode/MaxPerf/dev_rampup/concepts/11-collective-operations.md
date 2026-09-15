<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Collective Operations

## Prerequisites

None — this is a starting concept for the distributed section (though [TPU v6e Architecture](01-tpu-v6e-architecture.md) provides helpful hardware context).

## Leads to

- [Commutativity & Graph Commutation](12-commutativity-and-graph-commutation.md)
- [Payload Minimization](13-payload-minimization.md)
- [Pipeline Stage Balancing](17-pipeline-stage-balancing.md)

## Used by agents

- [MaxShard](../../agents/max_shard.md) — restructures collective patterns for efficiency
- [DeepResearch](../../agents/deep_research.md) — identifies redundant or suboptimal collectives

## What it is

Collective operations are communication primitives that synchronize data across multiple TPU chips. The primary collectives on TPU are:
- **All-reduce**: every chip contributes a tensor, all chips receive the sum (or other reduction)
- **All-gather**: every chip contributes a shard, all chips receive the full concatenated tensor
- **Reduce-scatter**: every chip contributes a tensor, each chip receives a different reduced shard
- **All-to-all**: general permutation — each chip sends a different piece to each other chip

Collectives execute over the ICI (inter-chip interconnect) network. Their cost is determined by payload size, number of chips, network topology (ring, mesh, hypercube), and whether they can overlap with compute. On TPU v6e, ICI provides ~4.5 TB/s bisection bandwidth per pod slice, but each individual collective is limited by the slowest link in its communication pattern.

## Why it matters for MaxPerf

Collectives are often the dominant cost in distributed training. A single all-reduce of a large gradient tensor can take longer than the corresponding compute. Method M7 (Payload Minimization) reduces the bytes that flow through collectives. Method M6 (Pipeline Stage Balancing) minimizes time chips spend idle waiting for collectives to complete. MaxShard rewrites the graph to use more efficient collective patterns (e.g., replacing all-gather + matmul with collective-matmul).

## Worked example

A model-parallel layer has weight shape [8192, 32768] sharded across 8 chips (each holding [8192, 4096]). The forward pass requires an all-gather to reconstruct the full weight: payload = 8192 × 32768 × 2 bytes = 512 MB. At 400 GB/s effective ICI bandwidth (accounting for ring overhead): 512 MB / 400 GB/s = 1.28 ms. The corresponding matmul takes 0.9 ms. The collective dominates — this layer is communication-bound. MaxShard could overlap the all-gather with the matmul by pipelining the gather and computation of each shard.

## See also

- [TPU v6e Architecture](01-tpu-v6e-architecture.md) — ICI hardware that carries collectives
- [Commutativity & Graph Commutation](12-commutativity-and-graph-commutation.md) — reordering ops around collectives
- [Payload Minimization](13-payload-minimization.md) — reducing collective traffic
- [Pipeline Stage Balancing](17-pipeline-stage-balancing.md) — hiding collective latency
