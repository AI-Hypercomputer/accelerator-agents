<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# HBM & Memory Hierarchy

## Prerequisites

- [TPU v6e Architecture](01-tpu-v6e-architecture.md)

## Leads to

- [VREG Spill & Register Pressure](05-vreg-spill-and-register-pressure.md)
- [Roofline Model & Bottleneck Analysis](14-roofline-model-and-bottleneck-analysis.md)

## Used by agents

- [TPUDiagnoseAgent](../../agents/tpu_diagnose.md) — measures bandwidth utilization and identifies memory-bound ops
- [MaxKernel](../../agents/max_kernel.md) — designs kernels to maximize data reuse in VMEM

## What it is

The TPU v6e memory hierarchy has three levels: HBM (high-bandwidth memory) at ~820 GB/s, VMEM (on-chip vector memory / scratchpad) at multi-TB/s internal bandwidth, and VREG (vector registers) directly feeding the compute units. Data must travel HBM → VMEM → VREG before computation, and results flow back VREG → VMEM → HBM.

HBM is the primary bottleneck for memory-bound operations. Unlike CPU caches, VMEM is software-managed — the compiler (or Pallas kernel author) explicitly stages data through DMA transfers. This means memory performance is deterministic but depends on correct scheduling.

The arithmetic intensity (FLOPs per byte loaded from HBM) determines whether an operation is compute-bound or memory-bound. Operations below ~1120 FLOPs/byte (920 TFLOPS / 820 GB/s) are memory-bound on v6e.

## Why it matters for MaxPerf

When the diagnostic vector shows low MXU utilization with high HBM bandwidth usage, the workload is memory-bound. The optimization strategy shifts from parallelism improvements to data reuse: fusing operations to keep intermediates in VMEM, recomputing cheap values instead of loading them, or restructuring layouts to improve DMA efficiency. This determines whether MaxKernel (custom kernel with better tiling) or AutoRefactor (fusion pass changes) should act.

## Worked example

A layer normalization over a [4096, 8192] tensor loads 64 MB, computes mean/variance/normalize (roughly 5 FLOPs/element × 33.5M elements = 167.5 MFLOP), and writes 64 MB. Arithmetic intensity = 167.5 MFLOP / 128 MB ≈ 1.3 FLOPs/byte. This is far below the 1120 FLOPs/byte ridge point, confirming it is memory-bound. MaxPerf should fuse this with adjacent operations to avoid the HBM round-trip.

## See also

- [TPU v6e Architecture](01-tpu-v6e-architecture.md) — the compute units this memory feeds
- [DMA Idle Time](03-dma-idle-time.md) — what happens when memory transfers stall
- [VREG Spill & Register Pressure](05-vreg-spill-and-register-pressure.md) — when register demand exceeds capacity
- [Pallas Kernels & Sublane Layout](20-pallas-kernels-and-sublane-layout.md) — explicit VMEM management
