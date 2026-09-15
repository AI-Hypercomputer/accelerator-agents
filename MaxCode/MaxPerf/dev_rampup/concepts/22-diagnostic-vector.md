<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Diagnostic Vector

## Prerequisites

- [Hypothesis Classes & Origination](21-hypothesis-classes-and-origination.md)
- [Roofline Model & Bottleneck Analysis](14-roofline-model-and-bottleneck-analysis.md)

## Leads to

- [Experiment Protocol & Halt Rules](25-experiment-protocol-and-halt-rules.md)

## Used by agents

- [TPUDiagnoseAgent](../../agents/tpu_diagnose.md) — produces the diagnostic vector
- [Orchestrator](../../agents/maxperf_orchestrator.md) — consumes it to route optimization work
- [DeepResearch](../../agents/deep_research.md) — uses it to constrain hypothesis search

## What it is

The diagnostic vector is a structured summary of a model's performance state at a point in time. It captures the key measurements and classifications that drive optimization decisions. Rather than passing raw xprof traces (which are massive and unstructured), the diagnostic vector distills them into actionable fields:

```
DiagnosticVector {
  step_time_ms: float
  mxu_utilization: float (0-1)
  hbm_bw_utilization: float (0-1)
  ici_bw_utilization: float (0-1)
  bottleneck_class: enum {compute, memory, communication, pipeline_bubble}
  top_ops: list[(op_name, time_ms, bottleneck_class)]  // top 5 by time
  fusion_boundary_cost_ms: float  // total materialization overhead
  collective_time_ms: float  // total time in collectives
  bubble_fraction: float  // pipeline idle fraction
  dma_idle_fraction: float  // fraction of time DMA has no outstanding requests
}
```

The vector is compact (fits in a single message), comparable across runs (same fields), and sufficient for routing decisions (the orchestrator can select agents/methods purely from this vector without reading raw traces).

## Why it matters for MaxPerf

The diagnostic vector is the interface between profiling and optimization. TPUDiagnoseAgent produces it; every other agent consumes it. It answers the question "what should we work on?" in structured form. When `bottleneck_class = memory` and `fusion_boundary_cost_ms` is high, route to AutoRefactor/MaxKernel for fusion work. When `bottleneck_class = communication` and `collective_time_ms` dominates, route to MaxShard/DeepResearch for collective optimization.

## Worked example

TPUDiagnoseAgent profiles a 70B parameter model training step:

```
DiagnosticVector {
  step_time_ms: 1420
  mxu_utilization: 0.41
  hbm_bw_utilization: 0.62
  ici_bw_utilization: 0.78
  bottleneck_class: communication
  top_ops: [
    ("all-reduce.gradient", 380ms, communication),
    ("dot.attention_qk", 210ms, compute),
    ("dot.ffn_up", 185ms, compute),
    ("custom-call.flash_attn", 160ms, compute),
    ("all-gather.weight", 95ms, communication)
  ]
  fusion_boundary_cost_ms: 42
  collective_time_ms: 520
  bubble_fraction: 0.12
  dma_idle_fraction: 0.08
}
```

Interpretation: 520/1420 = 36.6% of step time is collectives. The system is communication-bound. Orchestrator routes to MaxShard (M7: payload minimization) targeting the gradient all-reduce (380 ms). Secondary: DeepResearch (M6: pipeline balancing) to address the 12% bubble fraction.

## See also

- [Roofline Model & Bottleneck Analysis](14-roofline-model-and-bottleneck-analysis.md) — theoretical foundation for classification
- [Hypothesis Classes & Origination](21-hypothesis-classes-and-origination.md) — what the vector drives
- [Experiment Protocol & Halt Rules](25-experiment-protocol-and-halt-rules.md) — using vectors to measure progress
- [Fusion Boundary Cost Accounting](19-fusion-boundary-cost-accounting.md) — computing fusion_boundary_cost_ms
