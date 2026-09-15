<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Roofline Model & Bottleneck Analysis

## Prerequisites

- [TPU v6e Architecture](01-tpu-v6e-architecture.md)
- [HBM & Memory Hierarchy](02-hbm-and-memory-hierarchy.md)

## Leads to

- [Iterative Constraint Narrowing](15-iterative-constraint-narrowing.md)
- [Pipeline Stage Balancing](17-pipeline-stage-balancing.md)

## Used by agents

- [TPUDiagnoseAgent](../../agents/tpu_diagnose.md) — classifies bottlenecks from profiles
- [DeepResearch](../../agents/deep_research.md) — determines which optimization method applies
- [Orchestrator](../../agents/maxperf_orchestrator.md) — routes work to the correct agent based on bottleneck type

## What it is

The roofline model plots achievable performance (FLOP/s) as a function of arithmetic intensity (FLOPs/byte). Two ceilings define the roofline: the compute ceiling (peak FLOP/s of the hardware) and the memory bandwidth ceiling (peak bytes/s × arithmetic intensity). An operation's performance is bounded by whichever ceiling it hits first.

On TPU v6e: compute ceiling ≈ 920 TFLOPS (bf16), memory ceiling ≈ 820 GB/s HBM bandwidth. The ridge point — where the two ceilings meet — is at 920 TFLOPS / 820 GB/s ≈ 1122 FLOPs/byte. Operations above this intensity are compute-bound; below it, memory-bound.

Bottleneck analysis extends the roofline to distributed systems by adding a third ceiling: ICI bandwidth. An operation can be compute-bound, memory-bound, or communication-bound. Identifying which ceiling is active determines which MaxPerf method to apply.

## Why it matters for MaxPerf

Method M1 (Bottleneck Analysis & Symbolic Graph Refactoring) starts here. The orchestrator uses roofline classification to route: memory-bound ops go to fusion optimization (AutoRefactor, MaxKernel), compute-bound ops go to kernel optimization (MaxKernel, MaxTile), and communication-bound ops go to collective optimization (MaxShard, DeepResearch). Misclassification wastes optimization effort — you can't fuse your way out of a compute bottleneck.

## Worked example

An xprof trace shows an operation taking 2.1 ms. From the HLO: it's a matmul [4096, 16384] × [16384, 4096] = 2 × 4096 × 16384 × 4096 = 549.8 GFLOP. It loads two inputs: 4096 × 16384 × 2 + 16384 × 4096 × 2 = 256 MB. Arithmetic intensity = 549.8 GFLOP / 256 MB = 2.15 GFLOP/GB ≈ 2148 FLOPs/byte.

This is above the ridge point (1122), so the op is compute-bound. Achieved FLOP/s = 549.8 / 2.1 ms = 261.8 TFLOPS = 28.5% of peak. The optimization target is MXU utilization — tile sizes, accumulation depth, or pipeline efficiency — not memory access patterns.

## See also

- [TPU v6e Architecture](01-tpu-v6e-architecture.md) — hardware ceilings
- [HBM & Memory Hierarchy](02-hbm-and-memory-hierarchy.md) — memory ceiling details
- [Iterative Constraint Narrowing](15-iterative-constraint-narrowing.md) — systematic optimization within a bottleneck regime
- [Diagnostic Vector](22-diagnostic-vector.md) — structured representation of bottleneck findings
