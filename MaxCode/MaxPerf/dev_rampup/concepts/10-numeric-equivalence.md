<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Numeric Equivalence

## Prerequisites

- [HLO Intermediate Representation](07-hlo-intermediate-representation.md)

## Leads to

- [Hypothesis Classes & Origination](21-hypothesis-classes-and-origination.md)
- [Experiment Protocol & Halt Rules](25-experiment-protocol-and-halt-rules.md)

## Used by agents

- [AutoRefactor](../../agents/auto_refactor.md) — validates that pass changes preserve numerics
- [MaxShard](../../agents/max_shard.md) — ensures graph rewrites don't change results
- [MaxKernel](../../agents/max_kernel.md) — verifies Pallas kernels match reference implementations
- [Orchestrator](../../agents/maxperf_orchestrator.md) — enforces numeric equivalence as a hard constraint

## What it is

Numeric equivalence means that an optimized program produces the same outputs (within acceptable tolerance) as the reference unoptimized program for the same inputs. This is the fundamental constraint on all MaxPerf optimizations: any change that breaks numeric equivalence is invalid, regardless of how much faster it is.

Floating-point arithmetic is not associative — `(a + b) + c ≠ a + (b + c)` in general. Reordering operations, changing reduction trees, or altering fusion boundaries can change intermediate rounding, producing different final values. MaxPerf must distinguish between:
- **Bitwise equivalence**: exactly the same bits (required for some operations)
- **Approximate equivalence**: within an acceptable tolerance (e.g., max |diff| < 1e-5 for fp32, or matching within bf16 precision)

The tolerance depends on the model: training usually tolerates larger differences (they average out over steps) while inference serving requires tighter bounds.

## Why it matters for MaxPerf

Every optimization hypothesis in MaxPerf includes a numeric validation step. Before declaring an optimization successful, the system runs the modified program on reference inputs and compares outputs. Method M5 (Iterative Constraint Narrowing) uses numeric equivalence as a binary pass/fail gate — hypotheses that violate it are immediately discarded, narrowing the search space.

## Worked example

DeepResearch proposes reordering a reduction: instead of reducing a [8192, 4096] tensor along dim=1 in a single pass, split into 4 chunks of 1024 and reduce each, then reduce the 4 partial results. This changes the summation order. On reference inputs, the max absolute difference is 2.3e-6 in fp32. For a training workload with bf16 gradients (precision ~1e-3), this is acceptable — the optimization passes. For an inference workload requiring fp32 determinism, it fails and is rejected.

## See also

- [HLO Intermediate Representation](07-hlo-intermediate-representation.md) — the level where equivalence is verified
- [Hypothesis Classes & Origination](21-hypothesis-classes-and-origination.md) — how equivalence constrains hypotheses
- [Experiment Protocol & Halt Rules](25-experiment-protocol-and-halt-rules.md) — validation procedures
- [Iterative Constraint Narrowing](15-iterative-constraint-narrowing.md) — using equivalence to prune search
