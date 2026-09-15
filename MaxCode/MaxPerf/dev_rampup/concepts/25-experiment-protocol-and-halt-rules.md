<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Experiment Protocol & Halt Rules

## Prerequisites

- [Numeric Equivalence](10-numeric-equivalence.md)
- [Diagnostic Vector](22-diagnostic-vector.md)
- [Hypothesis Classes & Origination](21-hypothesis-classes-and-origination.md)

## Leads to

None — this is a terminal concept (operational knowledge).

## Used by agents

- [Orchestrator](../../agents/maxperf_orchestrator.md) — enforces protocol and halt conditions
- All agents follow the protocol for their experiments

## What it is

The experiment protocol defines how MaxPerf validates optimization hypotheses. Every experiment follows a fixed structure:

1. **Baseline measurement**: run the unmodified model for N steps, record step time (mean, p50, p99), numeric outputs on reference inputs
2. **Modification**: apply the proposed optimization (flag change, graph rewrite, kernel replacement)
3. **Validation measurement**: run the modified model for N steps, record same metrics
4. **Numeric check**: compare outputs against baseline within tolerance
5. **Performance comparison**: compute speedup with confidence intervals
6. **Decision**: accept (if faster + numerically valid), reject (if slower or invalid), or inconclusive (if within noise)

Halt rules define when to stop the optimization loop:
- **Target achieved**: step time meets the performance target
- **Diminishing returns**: last K experiments each improved < threshold (e.g., < 0.5%)
- **Budget exhausted**: experiment count or wall-clock time exceeded
- **No viable hypotheses**: all remaining hypotheses are low-confidence or low-impact
- **Regression detected**: an accepted optimization caused degradation in a subsequent measurement (triggers rollback)

## Why it matters for MaxPerf

Without strict protocol, optimization becomes unscientific: you can't distinguish real improvements from noise, you waste time on marginal gains, and you risk accepting changes that degrade rarely-measured metrics. The halt rules prevent infinite optimization loops — they force the system to declare "good enough" and ship results. The orchestrator enforces these rules as hard constraints on all agent behavior.

## Worked example

Orchestrator runs the optimization loop on a model with baseline step time 1420 ms, target 1200 ms:

| Exp # | Hypothesis | Result | Step time | Decision |
|-------|-----------|--------|-----------|----------|
| 1 | Commute all-reduce past projection (M7) | -38 ms | 1382 ms | Accept |
| 2 | Fuse softmax components (M9) | -15 ms | 1367 ms | Accept |
| 3 | Remove DMA barriers in attention (M4) | -52 ms | 1315 ms | Accept |
| 4 | Custom flash-attn Pallas kernel (M9) | -89 ms | 1226 ms | Accept |
| 5 | Pipeline stage rebalance (M6) | -31 ms | 1195 ms | Accept → **TARGET MET** |

Halt: target 1200 ms achieved at experiment 5. Total improvement: 225 ms (15.8%). Remaining hypotheses (H6-H9) are not pursued — target is met.

Alternative halt scenario: if experiment 4 had been only -2 ms and experiments 5-7 similarly small, the "diminishing returns" rule (3 consecutive < 0.5% improvement) would trigger halt at experiment 7, accepting 1365 ms as the best achievable.

## See also

- [Numeric Equivalence](10-numeric-equivalence.md) — validation criterion within protocol
- [Diagnostic Vector](22-diagnostic-vector.md) — measurement structure for baseline/comparison
- [Hypothesis Classes & Origination](21-hypothesis-classes-and-origination.md) — what feeds into experiments
- [Pipeline Stage Balancing](17-pipeline-stage-balancing.md) — example of a method applied within protocol
