<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# XLA Compilation Passes

## Prerequisites

- [HLO Intermediate Representation](07-hlo-intermediate-representation.md)

## Leads to

- [XLA Fusion & Custom-Call Boundaries](09-xla-fusion-and-custom-call-boundaries.md)
- [Compile-Time Specialization Ladder](18-compile-time-specialization-ladder.md)
- [HLO Barrier Removal](24-hlo-barrier-removal.md)

## Used by agents

- [AutoRefactor](../../agents/auto_refactor.md) — manipulates pass flags and ordering to improve output
- [TPUDiagnoseAgent](../../agents/tpu_diagnose.md) — identifies which pass introduced overhead

## What it is

XLA compiles HLO through a pipeline of passes — each pass reads the HLO graph, applies a transformation, and produces modified HLO. The pipeline includes: algebraic simplification, constant folding, dead code elimination, fusion, layout assignment, memory scheduling, buffer assignment, and code generation.

Passes are ordered and sometimes interdependent: fusion decisions affect buffer sizes, which affect scheduling, which affects whether certain optimizations are profitable. Key passes for TPU performance include:
- **Fusion pass**: groups ops into fused computations (single kernels)
- **Layout assignment**: chooses memory layout (row-major, tiled, etc.)
- **Scheduling**: orders operations to overlap DMA and compute
- **Buffer assignment**: assigns HBM addresses, decides where to double-buffer

Each pass can be configured via XLA flags (e.g., `--xla_tpu_enable_aggressive_fusion`). AutoRefactor experiments with these flags systematically.

## Why it matters for MaxPerf

The XLA pass pipeline is where most performance is won or lost. A bad fusion decision can leave memory-bound ops unfused. A conservative scheduler can fail to overlap DMA with compute. Method M8 (Compile-Time Specialization Ladder) systematically explores pass configurations from conservative to aggressive, measuring the impact of each change. Method M4 (Execution Graph Unblocking) specifically targets scheduling passes that insert unnecessary barriers.

## Worked example

AutoRefactor identifies that a model's attention block runs 25% slower than expected. Dumping HLO before/after the fusion pass reveals that the softmax (exp + reduce + divide) is split across two fusion computations because an intermediate exceeds the fusion heuristic's memory threshold. Setting `--xla_tpu_fusion_level=2` (more aggressive) fuses them into one kernel, eliminating an HBM write/read pair of 32 MB. Time improvement: 22%.

## See also

- [HLO Intermediate Representation](07-hlo-intermediate-representation.md) — what passes transform
- [XLA Fusion & Custom-Call Boundaries](09-xla-fusion-and-custom-call-boundaries.md) — fusion pass details
- [Compile-Time Specialization Ladder](18-compile-time-specialization-ladder.md) — systematic flag exploration
- [HLO Barrier Removal](24-hlo-barrier-removal.md) — removing scheduling barriers
