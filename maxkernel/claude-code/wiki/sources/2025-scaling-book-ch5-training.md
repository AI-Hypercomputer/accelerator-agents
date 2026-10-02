---
title: "How to Scale Your Model — Ch 5: How to Parallelize a Transformer for Training"
type: source
tags: [docs, book, scaling-book, parallelism, fsdp, tensor-parallelism, pipeline-parallelism, data-parallelism, training]
author: DeepMind / JAX ML team
book: "How to Scale Your Model"
book_date: "2025-02-04"
chapter: 5
upstream: https://jax-ml.github.io/scaling-book/training
created: 2026-04-23
updated: 2026-04-23
---

Chapter 5 of the scaling-book. **Derives compute-vs-comm thresholds for each of the four main parallelism schemes** (data, FSDP, tensor, pipeline) from Ch 1 rooflines + Ch 3 collective costs.

## Key claims

1. **Data Parallelism**: activations sharded `B_X`, weights replicated. No forward comms; AllReduce gradients in backward. Compute-bound when `B/X > C/W_ici ≈ 2550` (v5p 1D) or `≈ 850` (v5p 3D, three parallel axes).
2. **FSDP / ZeRO-3**: weights sharded `D_X`; AllGather in forward (prefetchable from prior layer), ReduceScatter in backward. Same threshold as DP but much lower per-device memory.
3. **Tensor Parallelism (Megatron-style)**: activations sharded `D_Y`, weights sharded `F_Y`; AllGather input before matmul 1, ReduceScatter output after matmul 2. Compute-bound when `B > 2550 / M_Y` (M_Y = number of mesh axes dedicated to TP).
4. **Pipeline Parallelism**: weights sharded `L_Z` (layer axis), microbatches pipelined; minimal inter-stage comms (activations over single hop). Bubble cost ≈ `(num_stages - 1) / num_stages` for GPipe; smaller with 1F1B / interleaved / zero-bubble / DualPipe schedules.
5. **Hybrid** is the usual answer: `X_opt = sqrt(2 B N / F)` balances FSDP and TP for a given batch/model shape.
6. **Sequence parallelism** (a TP-companion): shards LayerNorm/Dropout along sequence; converts internal AllReduce → ReduceScatter + AllGather (same total cost, lower peak memory).

## Key data points

### Communication-compute threshold per scheme (v5p)

| Scheme | 1D ring threshold | 3D mesh threshold |
|---|---:|---:|
| DP | `B/X > 2550` | `B/X > 850` |
| FSDP | `B/X > 2550` | `B/X > 850` |
| TP (1-axis) | `B > 2550` | `B > 850` |

(`C/W` = peak FLOPs / ICI bandwidth for the applicable mesh).

### Pipeline bubble

- GPipe: bubble fraction `(S-1) / (S+M-1)` where S=stages, M=microbatches.
- 1F1B: similar; reduces pipeline-stage memory.
- Zero-bubble / DualPipe: schedules named; perf claims not deeply quantified here.

## Techniques referenced

- Megatron tensor parallelism (Q/K/V column / O row split); see [concepts/tensor-parallelism](../concepts/tensor-parallelism.md).
- FSDP / ZeRO gradient/optimizer/parameter sharding.
- Pipeline parallelism (GPipe, 1F1B, interleaved, zero-bubble, DualPipe).
- Sequence / context parallelism (ring attention); see [codebases/ringattention](../codebases/ringattention.md).
- Gradient accumulation for effective batch size.
- Async AllReduce / ReduceScatter overlap.

## Gaps & caveats

- 1D ring / 2D-3D mesh ICI only; doesn't model DCN multi-slice scaling.
- Ignores pipeline bubble for small stage counts.
- Compute-comm overlap assumed perfect; XLA typically under-overlaps by 1.5–2×.
- **Sparse / MoE** models need AllToAll (different cost model) — only briefly noted.
- Assumes v5p numbers; **v6e / v7 thresholds differ**.
- Gemma4 [exp 32 (2D mesh)](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp32-2d-mesh-tp2-rejected.md) showed 2D TP is **economic only above certain chip counts** — consistent with this chapter's thresholds but the **v6e-4 regime sits below the sweet spot**, which the book doesn't cover.

## Connections

- [concepts/fsdp](../concepts/fsdp.md) / [tensor-parallelism](../concepts/tensor-parallelism.md) / [sequence-parallelism](../concepts/sequence-parallelism.md) / [pipeline-parallelism](../concepts/pipeline-parallelism.md) / [context-parallelism](../concepts/context-parallelism.md) / [expert-parallelism](../concepts/expert-parallelism.md).
- [concepts/async-collectives](../concepts/async-collectives.md) / [latency-hiding-scheduler](../concepts/latency-hiding-scheduler.md).
- Gemma4 experiments on TP / FSDP: [exp 24](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp24-splash-seq-minor-accepted.md), [exp 25](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp25-splash-block1024-accepted.md), [exp 32](../experiments/gemma4_autoresearch_optimization/2026-04-23-exp32-2d-mesh-tp2-rejected.md).

## See also

- [Ch 3 — Sharding](2025-scaling-book-ch3-sharding.md)
- [Ch 6 — Applied Training](2025-scaling-book-ch6-applied-training.md)

## Sources

- `raw/code/scaling-book/training.md`
- Upstream: <https://jax-ml.github.io/scaling-book/training>
