<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Iterative Constraint Narrowing

## Prerequisites

- [Roofline Model & Bottleneck Analysis](14-roofline-model-and-bottleneck-analysis.md)
- [Numeric Equivalence](10-numeric-equivalence.md)

## Leads to

- [Algebraic ISA Modeling](16-algebraic-isa-modeling.md)
- [Hypothesis Classes & Origination](21-hypothesis-classes-and-origination.md)

## Used by agents

- [MaxKernel](../../agents/max_kernel.md) — narrows tile-size search space
- [MaxShard](../../agents/max_shard.md) — narrows sharding configuration space
- [DeepResearch](../../agents/deep_research.md) — derives constraints algebraically

## What it is

Iterative constraint narrowing is the method of progressively eliminating infeasible regions of the optimization search space by applying constraints derived from hardware limits, numeric requirements, and empirical measurements. Instead of searching all possible configurations, each iteration adds a constraint that removes a class of options.

The process:
1. Start with the full space of possible optimizations (all tile sizes, all sharding strategies, all fusion decisions)
2. Apply hard constraints: numeric equivalence eliminates certain reorderings, register limits eliminate certain tile sizes, sharding divisibility eliminates certain partitionings
3. Apply soft constraints from profiling: measured bandwidth utilization eliminates configurations that would exceed HBM capacity, observed latencies eliminate configurations whose predicted cost is worse than current
4. Each constraint monotonically shrinks the feasible set
5. Iterate until the feasible set is small enough to evaluate exhaustively or a clear optimum emerges

This is not gradient descent — it's logical elimination. Each constraint is provably correct, so the optimum is guaranteed to be in the remaining feasible set.

## Why it matters for MaxPerf

Method M5 (Iterative Constraint Narrowing) is the core search strategy that distinguishes MaxPerf from brute-force autotuning. Instead of running thousands of experiments, MaxPerf derives constraints that eliminate large regions of the space without running any experiments. A single algebraic observation ("tile sizes that cause VREG spill are always worse") can eliminate 60% of the search space in one step.

## Worked example

Optimizing tile size for a Pallas matmul kernel. Initial space: all divisors of [4096, 8192] → 96 possible tile configs.

Constraint 1 (register budget): tile m × n × 2 bytes ≤ 192 KB → eliminates tiles where m×n > 98304. Removes 40 configs, leaving 56.

Constraint 2 (MXU alignment): m and n must be multiples of 128. Removes 31 configs, leaving 25.

Constraint 3 (HBM bandwidth): total data loaded per tile must achieve ≥50% bandwidth utilization. For tiles where payload < 8 KB, DMA overhead dominates. Removes 8 configs, leaving 17.

Constraint 4 (measured baseline): any config must beat current 1.2 ms. Algebraic cost model predicts 6 configs exceed this. Removes 6, leaving 11.

From 96 candidates to 11 in four steps. Evaluate the 11 remaining configurations empirically.

## See also

- [Roofline Model & Bottleneck Analysis](14-roofline-model-and-bottleneck-analysis.md) — provides initial constraints
- [Numeric Equivalence](10-numeric-equivalence.md) — constraint source (correctness)
- [Algebraic ISA Modeling](16-algebraic-isa-modeling.md) — deriving cost constraints without experiments
- [Hypothesis Classes & Origination](21-hypothesis-classes-and-origination.md) — organizing hypotheses within constraints
