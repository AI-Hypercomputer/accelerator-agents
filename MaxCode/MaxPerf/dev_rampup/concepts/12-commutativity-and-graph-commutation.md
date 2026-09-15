<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Commutativity & Graph Commutation

## Prerequisites

- [Collective Operations](11-collective-operations.md)

## Leads to

- [Payload Minimization](13-payload-minimization.md)
- [Graph Rewrite Patterns](23-graph-rewrite-patterns.md)

## Used by agents

- [DeepResearch](../../agents/deep_research.md) — identifies commutable operation pairs
- [MaxShard](../../agents/max_shard.md) — applies commutation rewrites to the graph

## What it is

Commutativity in this context means: if two operations can be reordered without changing the final result, they commute. Graph commutation is the technique of reordering operations in the HLO graph to move expensive collectives past cheap local operations, or to merge multiple collectives into one.

The key insight: `all-reduce(A + B) = all-reduce(A) + all-reduce(B)` (linearity), but more usefully: `all-reduce(f(x)) = f(all-reduce(x))` when f is a linear operation applied identically on all chips. This means you can push an all-reduce earlier or later in the graph if the surrounding operations are linear.

Common commutable pairs:
- all-reduce commutes with scalar multiplication: `all-reduce(α·x) = α·all-reduce(x)`
- all-reduce commutes with addition of replicated tensors: `all-reduce(x + b) = all-reduce(x) + N·b` (where b is replicated)
- reduce-scatter commutes with element-wise ops that partition cleanly

Non-commutable: all-reduce does NOT commute with non-linear operations (ReLU, softmax), or operations with cross-element dependencies (convolution with halos).

## Why it matters for MaxPerf

Method M7 (Payload Minimization via Algebraic Commutativity) exploits commutativity to restructure the graph so that collectives operate on smaller tensors or are merged. By pushing an all-reduce past a projection that reduces dimensions, the collective payload shrinks. DeepResearch identifies these opportunities algebraically, and MaxShard implements the graph rewrite.

## Worked example

Original graph: `gradient [8192, 32768]` → `all-reduce [8192, 32768]` → `project [8192, 4096]` (multiply by a [32768, 4096] projection matrix).

The projection is a linear operation, so we can commute: `gradient [8192, 32768]` → `project [8192, 4096]` → `all-reduce [8192, 4096]`.

All-reduce payload drops from 8192 × 32768 × 2 = 512 MB to 8192 × 4096 × 2 = 64 MB — an 8x reduction in communication volume. At 400 GB/s ICI: from 1.28 ms to 0.16 ms.

## See also

- [Collective Operations](11-collective-operations.md) — the operations being reordered
- [Payload Minimization](13-payload-minimization.md) — the goal of commutation
- [Graph Rewrite Patterns](23-graph-rewrite-patterns.md) — structural patterns for implementing commutation
- [Numeric Equivalence](10-numeric-equivalence.md) — ensuring commutation preserves correctness
