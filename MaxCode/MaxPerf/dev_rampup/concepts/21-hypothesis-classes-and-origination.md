<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Hypothesis Classes & Origination

## Prerequisites

- [Iterative Constraint Narrowing](15-iterative-constraint-narrowing.md)
- [Numeric Equivalence](10-numeric-equivalence.md)
- [Pallas Kernels & Sublane Layout](20-pallas-kernels-and-sublane-layout.md)

## Leads to

- [Diagnostic Vector](22-diagnostic-vector.md)

## Used by agents

- [DeepResearch](../../agents/deep_research.md) — generates and classifies hypotheses
- [Orchestrator](../../agents/maxperf_orchestrator.md) — selects which hypotheses to pursue

## What it is

A hypothesis in MaxPerf is a specific, testable claim about how to improve performance: "Fusing ops X and Y will reduce step time by Z ms because it eliminates N MB of HBM traffic." Hypothesis classes categorize these claims by the type of optimization they propose:

- **Fusion hypotheses**: "ops A, B, C should be in one kernel"
- **Sharding hypotheses**: "tensor X should be partitioned differently"
- **Scheduling hypotheses**: "op A should execute before op B to enable overlap"
- **Collective hypotheses**: "this all-reduce can be payload-minimized"
- **Kernel hypotheses**: "a custom Pallas kernel for subgraph S will outperform XLA's codegen"

Origination is the process by which hypotheses are generated. Sources include:
1. **Profile-driven**: TPUDiagnoseAgent identifies a bottleneck → generates hypotheses to address it
2. **Pattern-driven**: DeepResearch recognizes a known graph pattern → generates the associated optimization hypothesis
3. **Constraint-driven**: iterative narrowing reveals that only certain hypotheses are feasible → generates the feasible subset
4. **Analogy-driven**: a previous model had similar structure and benefited from optimization X → hypothesis: same optimization applies here

Each hypothesis has: a class, an origination source, a predicted impact (ms saved), a confidence level, and a validation criterion.

## Why it matters for MaxPerf

The orchestrator must decide which hypotheses to pursue given limited experiment budget. Hypothesis classification enables prioritization: high-confidence, high-impact hypotheses from profile-driven origination go first. Low-confidence hypotheses from analogy are deprioritized. This structure prevents the system from wasting experiments on speculative optimizations when reliable ones are available.

## Worked example

TPUDiagnoseAgent profiles a model and produces:

| # | Class | Hypothesis | Origin | Predicted impact | Confidence |
|---|-------|-----------|--------|-----------------|------------|
| H1 | Fusion | Fuse softmax components (3 ops → 1 kernel) | Profile: 3 unfused ops each memory-bound | 2.1 ms/step | High (90%) |
| H2 | Collective | Commute all-reduce past projection | Pattern: matches linear-post-collective | 0.8 ms/step | High (85%) |
| H3 | Kernel | Replace attention with Pallas flash-attn | Analogy: worked on similar model | 5.0 ms/step | Medium (60%) |
| H4 | Scheduling | Overlap gradient all-reduce with backward compute | Pattern: standard gradient overlap | 12.0 ms/step | Medium (70%) |

Orchestrator orders: H4 (highest expected value = 12×0.7 = 8.4), H3 (5×0.6 = 3.0), H1 (2.1×0.9 = 1.9), H2 (0.8×0.85 = 0.7). Pursue in that order, halt when time budget exhausted.

## See also

- [Iterative Constraint Narrowing](15-iterative-constraint-narrowing.md) — constraining which hypotheses are feasible
- [Numeric Equivalence](10-numeric-equivalence.md) — validation criterion for all hypotheses
- [Diagnostic Vector](22-diagnostic-vector.md) — structured output driving origination
- [Experiment Protocol & Halt Rules](25-experiment-protocol-and-halt-rules.md) — how hypotheses are tested
