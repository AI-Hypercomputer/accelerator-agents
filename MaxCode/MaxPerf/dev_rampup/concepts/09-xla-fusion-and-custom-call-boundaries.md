<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# XLA Fusion & Custom-Call Boundaries

## Prerequisites

- [XLA Compilation Passes](08-xla-compilation-passes.md)

## Leads to

- [Fusion Boundary Cost Accounting](19-fusion-boundary-cost-accounting.md)
- [Compile-Time Specialization Ladder](18-compile-time-specialization-ladder.md)
- [Pallas Kernels & Sublane Layout](20-pallas-kernels-and-sublane-layout.md)

## Used by agents

- [AutoRefactor](../../agents/auto_refactor.md) — adjusts fusion boundaries via pass flags
- [MaxKernel](../../agents/max_kernel.md) — writes custom-calls that replace suboptimal fusions

## What it is

Fusion is XLA's mechanism for combining multiple HLO operations into a single kernel that executes without materializing intermediate results to HBM. A fused computation keeps data in VMEM/VREG between operations, avoiding expensive memory round-trips. XLA's fusion heuristics decide which ops to group based on: data dependencies, memory footprint of intermediates, and estimated register pressure.

Custom-calls are opaque operations from XLA's perspective — they call into external implementations (like cuDNN for GPU, or Pallas kernels for TPU). Custom-call boundaries are fusion barriers: XLA cannot fuse across them because it doesn't understand the internal computation. This means every custom-call forces materialization of its inputs to HBM and reads its outputs from HBM.

The tension: custom-calls often contain highly optimized implementations (hand-tuned GEMM), but their opacity prevents broader fusion. When a flash-attention custom-call sits between two fusible ops, those ops cannot be fused together through it.

## Why it matters for MaxPerf

Method M9 (Fusion Boundary Cost Accounting) quantifies the HBM traffic cost of each fusion boundary. When a custom-call boundary forces 128 MB of materialization that could be avoided, MaxKernel can write a Pallas kernel that internalizes the surrounding ops — replacing the opaque custom-call with a transparent fused implementation. AutoRefactor can alternatively try adjusting fusion heuristic thresholds to capture more ops in a single fusion.

## Worked example

A model's MLP block has: LayerNorm (fused) → Linear (custom-call GEMM) → GeLU (fused) → Linear (custom-call GEMM). The two custom-call boundaries force LayerNorm output (64 MB) and GeLU output (64 MB) to materialize in HBM. Method M9 calculates cost: 256 MB of HBM traffic at 820 GB/s = 0.31 ms overhead. MaxKernel writes a Pallas kernel fusing LayerNorm + Linear + GeLU into one operation, eliminating the first boundary. Savings: 128 MB traffic = 0.16 ms per layer × 96 layers = 15.4 ms total.

## See also

- [XLA Compilation Passes](08-xla-compilation-passes.md) — the broader pass pipeline context
- [Fusion Boundary Cost Accounting](19-fusion-boundary-cost-accounting.md) — quantifying boundary costs
- [Pallas Kernels & Sublane Layout](20-pallas-kernels-and-sublane-layout.md) — replacing custom-calls
- [HLO Intermediate Representation](07-hlo-intermediate-representation.md) — reading fusion structure in HLO
