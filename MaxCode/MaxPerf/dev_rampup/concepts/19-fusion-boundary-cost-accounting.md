<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Fusion Boundary Cost Accounting

## Prerequisites

- [XLA Fusion & Custom-Call Boundaries](09-xla-fusion-and-custom-call-boundaries.md)
- [HBM & Memory Hierarchy](02-hbm-and-memory-hierarchy.md)

## Leads to

- [Pallas Kernels & Sublane Layout](20-pallas-kernels-and-sublane-layout.md)

## Used by agents

- [MaxKernel](../../agents/max_kernel.md) — decides when custom kernels justify their development cost
- [AutoRefactor](../../agents/auto_refactor.md) — quantifies benefit of fusion flag changes

## What it is

Fusion boundary cost accounting is the technique of quantifying the exact HBM traffic cost imposed by each fusion boundary in the compiled graph. Every fusion boundary forces intermediate tensors to be materialized to HBM — the cost is: bytes_written + bytes_read_by_consumer = 2 × tensor_size (write then read). When expressed in time: cost = 2 × tensor_size / HBM_bandwidth.

The accounting process:
1. Dump the post-fusion HLO graph
2. Identify all fusion boundaries (edges that cross fusion computation boundaries)
3. For each boundary edge, compute the tensor size from its shape and dtype
4. Sum: total materialization cost = Σ (2 × size_i) / BW
5. Rank boundaries by cost — the most expensive are optimization targets

This creates a prioritized list: "removing boundary X saves Y ms." This guides whether to pursue fusion flag changes (AutoRefactor) or custom kernel replacement (MaxKernel).

## Why it matters for MaxPerf

Method M9 (Fusion Boundary Cost Accounting) is the decision framework for kernel writing investment. If the top fusion boundary costs 0.5 ms per step and the model runs 1000 steps/training, that boundary costs 500 ms of training time. A Pallas kernel that eliminates it is worth writing if development time < value of time saved. The accounting also tells AutoRefactor whether a fusion flag experiment is worth trying: if all boundaries together cost only 2% of step time, other methods offer more gain.

## Worked example

Post-fusion HLO for a single transformer layer reveals these boundary edges:

| Boundary | Shape | Dtype | Size | Cost @820 GB/s |
|----------|-------|-------|------|---------------|
| LayerNorm → Attention | [4096, 8192] | bf16 | 64 MB | 0.156 ms |
| Attention → Dropout | [4096, 8192] | bf16 | 64 MB | 0.156 ms |
| GeLU → Linear2 | [4096, 32768] | bf16 | 256 MB | 0.624 ms |
| Linear2 → Residual | [4096, 8192] | bf16 | 64 MB | 0.156 ms |

Total boundary cost: 1.09 ms/layer. With 96 layers: 104.6 ms/step. The GeLU→Linear2 boundary alone costs 60 ms/step — this is the highest-value target for MaxKernel. Writing a fused GeLU+Linear Pallas kernel eliminates 60 ms, a 4.2% improvement on a 1.4s step.

## See also

- [XLA Fusion & Custom-Call Boundaries](09-xla-fusion-and-custom-call-boundaries.md) — what creates boundaries
- [HBM & Memory Hierarchy](02-hbm-and-memory-hierarchy.md) — bandwidth determining cost
- [Pallas Kernels & Sublane Layout](20-pallas-kernels-and-sublane-layout.md) — eliminating boundaries with custom kernels
- [Roofline Model & Bottleneck Analysis](14-roofline-model-and-bottleneck-analysis.md) — whether boundary removal helps
