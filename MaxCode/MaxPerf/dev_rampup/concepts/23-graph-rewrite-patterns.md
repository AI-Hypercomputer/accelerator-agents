<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Graph Rewrite Patterns

## Prerequisites

- [HLO Intermediate Representation](07-hlo-intermediate-representation.md)
- [Commutativity & Graph Commutation](12-commutativity-and-graph-commutation.md)

## Leads to

- [HLO Barrier Removal](24-hlo-barrier-removal.md)

## Used by agents

- [MaxShard](../../agents/max_shard.md) — applies structural rewrites to the HLO graph
- [DeepResearch](../../agents/deep_research.md) — identifies applicable patterns

## What it is

Graph rewrite patterns are templates that match subgraphs in HLO and replace them with semantically equivalent but more efficient subgraphs. A pattern has three components: (1) a match template that identifies the target subgraph, (2) a validity check that confirms the rewrite is safe for this instance, and (3) a replacement template that produces the optimized subgraph.

Common patterns in MaxPerf:

**Collective-commutation pattern**: `all-reduce → linear_op` becomes `linear_op → all-reduce` (reduces payload)

**Fusion-merge pattern**: two adjacent fusion computations sharing a producer become one larger fusion (eliminates intermediate materialization)

**Scatter-gather elimination**: `reduce-scatter → compute → all-gather` becomes `all-reduce → compute` when compute is element-wise (one collective instead of two)

**Resharding pattern**: `all-to-all → transpose → slice` becomes a direct resharded read when the target layout is known

Each pattern has preconditions (e.g., "the linear_op must have no cross-chip dependencies") and postconditions (e.g., "result is bitwise identical" or "result is within bf16 tolerance").

## Why it matters for MaxPerf

Graph rewrites are the primary implementation mechanism for methods M1 (Symbolic Graph Refactoring) and M7 (Payload Minimization). DeepResearch identifies which patterns apply to the current model's HLO, and MaxShard implements the actual rewrite. The pattern library grows as MaxPerf encounters new models — each successful optimization can be generalized into a reusable pattern.

## Worked example

DeepResearch identifies the following subgraph in a model's backward pass:

```
%grad = f32[4096, 8192] ...
%all-reduce.1 = f32[4096, 8192] all-reduce(%grad)
%slice.1 = f32[4096, 4096] slice(%all-reduce.1, [0:4096, 0:4096])
```

This matches the "reduce-then-slice" antipattern: all-reduce produces 64 MB but only 32 MB is used. Pattern: replace with reduce-scatter that only produces the needed shard:

```
%grad = f32[4096, 8192] ...
%reduce-scatter.1 = f32[4096, 4096] reduce-scatter(%grad, dim=1, shard=0)
```

Collective payload: 64 MB → 32 MB (50% reduction). Validity: the slice boundaries align with reduce-scatter shard boundaries. MaxShard applies the rewrite.

## See also

- [HLO Intermediate Representation](07-hlo-intermediate-representation.md) — the graph being rewritten
- [Commutativity & Graph Commutation](12-commutativity-and-graph-commutation.md) — algebraic foundation
- [Payload Minimization](13-payload-minimization.md) — goal of many rewrites
- [Numeric Equivalence](10-numeric-equivalence.md) — correctness requirement for all rewrites
