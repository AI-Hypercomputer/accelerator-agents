---
title: "Source: roofline.md"
type: source-page
doc: roofline.md
updated: 2026-07-03
---
# roofline.md — All About Rooflines

Part 1 of the book. Establishes the roofline model: computation time $T_\text{math} = \text{FLOPs} /
\text{Accelerator FLOPs/s}$, communication time $T_\text{comms} = \text{Bytes} / \text{Bandwidth}$,
and arithmetic intensity (FLOPs per byte) as the metric that predicts whether an operation is
compute-bound or bandwidth-bound
([where-does-the-time-go](src:roofline.md#where-does-the-time-go)). Derives the critical arithmetic
intensity of TPU v5e's MXU (~240 FLOPs/byte) and shows that a bf16 matmul is compute-bound once the
per-replica token batch size exceeds ~240
([matrix-multiplication](src:roofline.md#matrix-multiplication)). Extends the same model to
inter-chip network communication rooflines, showing the compute-bound threshold for a 2-chip
sharded matmul depends on the contracting dimension size $D$, not the batch size
([network-communication-rooflines](src:roofline.md#network-communication-rooflines)).

## Feeds
- [Rooflines and arithmetic intensity](../topics/rooflines-and-arithmetic-intensity.md) — the core
  topic this chapter defines.
- [TPU hardware architecture](../topics/tpu-hardware-architecture.md) — consumes the 240
  FLOPs/byte critical intensity number for TPU v5e.
