<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Pallas Kernels & Sublane Layout

## Prerequisites

- [ISA Latency Tables](04-isa-latency-tables.md)
- [VREG Spill & Register Pressure](05-vreg-spill-and-register-pressure.md)
- [XLA Fusion & Custom-Call Boundaries](09-xla-fusion-and-custom-call-boundaries.md)
- [Algebraic ISA Modeling](16-algebraic-isa-modeling.md)

## Leads to

- [Hypothesis Classes & Origination](21-hypothesis-classes-and-origination.md)

## Used by agents

- [MaxKernel](../../agents/max_kernel.md) — primary author of Pallas kernels
- [MaxTile](../../agents/max_tile.md) — tunes kernel parameters (grid, block sizes)

## What it is

Pallas is JAX's kernel-writing language for TPU. It exposes the TPU's software-managed memory hierarchy: the programmer explicitly specifies DMA transfers between HBM and VMEM, computation on VREG-resident data, and the tiling/gridding strategy. This gives full control over data movement — unlike XLA's automatic scheduling, Pallas kernels execute exactly as written.

Sublane layout refers to how data is arranged within TPU VREG tiles. The TPU's vector unit operates on sublanes (groups of 8 or 16 elements processed in lock-step). Data layout in memory must match the expected sublane arrangement for zero-cost loads; mismatched layouts require costly transpose/permute operations. Key layouts:
- **(128, 128) tiled**: standard MXU-compatible layout
- **(8, 128) sublane**: natural vector processing layout
- **Transposed variants**: for operations needing the other dimension first

A well-written Pallas kernel: (1) tiles computation to fit in VMEM, (2) double-buffers DMA to overlap transfers with compute, (3) respects sublane layout to avoid transposes, and (4) stays within the VREG budget to avoid spills.

## Why it matters for MaxPerf

When fusion boundary cost accounting identifies expensive boundaries that XLA's heuristics won't fuse, MaxKernel writes a Pallas kernel to replace the subgraph. This kernel can fuse operations across custom-call boundaries, use layouts optimized for the specific problem, and employ tiling strategies that XLA's general-purpose algorithm wouldn't attempt. Method M9 provides the justification (cost), and Pallas provides the implementation mechanism.

## Worked example

MaxKernel writes a fused LayerNorm + Linear kernel. The default XLA compilation materializes the LayerNorm output (64 MB) to HBM before the Linear reads it. The Pallas kernel:

```
Grid: (batch_tiles=32, output_tiles=64)
Block: (128, 128) per tile
Pipeline:
  Stage 1: DMA load input tile [128, 8192] to VMEM buf_a
  Stage 2: Compute mean/var on buf_a (128 elements → scalar)
  Stage 3: Normalize buf_a in-place in VMEM
  Stage 4: DMA load weight tile [8192, 128] to VMEM buf_b
  Stage 5: MXU matmul buf_a × buf_b → accumulator in VREG
  Stage 6: DMA store result [128, 128] to HBM
```

The intermediate normalized tensor never touches HBM — it stays in VMEM between LayerNorm and Linear. Savings: 64 MB × 2 / 820 GB/s = 0.156 ms per layer.

## See also

- [ISA Latency Tables](04-isa-latency-tables.md) — instruction costs driving kernel design
- [VREG Spill & Register Pressure](05-vreg-spill-and-register-pressure.md) — register budget constraint
- [Fusion Boundary Cost Accounting](19-fusion-boundary-cost-accounting.md) — quantifying the value of fusion
- [Algebraic ISA Modeling](16-algebraic-isa-modeling.md) — predicting kernel performance
