---
title: "two-level chunk/subchunk recomputation (SSD pattern)"
type: concept
tags: [pallas, ssm, mamba, mamba2, ssd, rematerialization, axlearn, stub]
created: 2026-04-23
updated: 2026-04-23
---

Memory-saving pattern for Pallas linear-recurrence kernels: store coarse-grained "chunk-level" carries, recompute fine-grained "subchunk-level" states on backward. Originates in AxLearn's Mamba2 SSD kernel (`ssd_kernels.py`). *Stub — expand when more sources are available.*

## Definition

Split the sequence axis into **chunks** and each chunk into **subchunks**. During forward, save only per-chunk carries (`h_chunk_end`). During backward, re-run the forward scan within each chunk from `h_chunk_start` to rebuild the subchunk-level states exactly when the gradients need them. Intermediate per-subchunk state is never held in HBM across the forward/backward boundary.

## Why it matters for TPU perf

Linear recurrences have state-per-token `O(seq_len × dim)` — far too large to checkpoint fully. Subchunk-level recomputation shrinks the persisted footprint to `O(num_chunks × dim)`, trading VPU FLOPs for HBM bandwidth. On memory-bandwidth-bound TPU training, this is almost always a win.

## Mechanism

1. Partition seq axis into C chunks × S subchunks.
2. Forward: compute subchunk states `h_{c,s}`; persist only `h_{c, S}` (chunk boundary carry).
3. Backward: for each chunk c, restart from `h_{c, 0}` (equal to previous chunk's `h_{c-1, S}`), recompute `h_{c, s}` for s=0..S, feed into the VJP.
4. Never store `h_{c, s}` for s<S to HBM during forward.

## When it applies / when it doesn't

- **Applies** to any linear recurrence with associative carry (LRU / S4 / Mamba1 / Mamba2 / RAttention) and long sequences (where full checkpointing is expensive).
- **Does not apply** when full checkpointing fits comfortably (short-sequence or low-state-dim recurrences).

## Known results

| Reference | Repo | Notes |
|---|---|---|
| `ssd_kernels.py` (Mamba2 / SSD) | [axlearn](../codebases/axlearn.md) | Canonical implementation; docstring: "two-level chunking algorithm to balance memory consumption and running speed" |
| `linear_attention_kernels.py` (RAttention) | [axlearn](../codebases/axlearn.md) | Chunking strategy mirrors SSD |

## Connections

- [rematerialization](rematerialization.md) — more general gradient-checkpointing concept.
- [scan-over-layers](scan-over-layers.md) — companion pattern at the layer axis.
- [pallas-kernel](pallas-kernel.md)

## Sources

- [axlearn codebase](../codebases/axlearn.md) "Performance-relevant surfaces §2".
- [Pallas kernel directory §4.1](../analyses/pallas-kernel-directory/04-research-labs.md#41-appleaxlearn).
- Mamba2 paper: [arXiv:2405.21060](https://arxiv.org/abs/2405.21060).
