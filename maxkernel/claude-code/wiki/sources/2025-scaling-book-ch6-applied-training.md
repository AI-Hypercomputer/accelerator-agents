---
title: "How to Scale Your Model — Ch 6: Training LLaMA 3 on TPUs"
type: source
tags: [docs, book, scaling-book, llama3, llama-3-70b, training, applied, v5p, tpu-pod, mfu]
author: DeepMind / JAX ML team
book: "How to Scale Your Model"
book_date: "2025-02-04"
chapter: 6
upstream: https://jax-ml.github.io/scaling-book/applied-training
created: 2026-04-23
updated: 2026-04-23
---

Chapter 6 of the scaling-book. **Concrete worked example** — LLaMA-3-70B training on a v5p pod. Applies Ch 1–5 to produce time-to-train, memory, and optimal-sharding numbers for a real production model.

## Key claims

1. **LLaMA-3-70B**: L=80, D=8192, F=28672, N=64 heads, K=8 KV heads, H=128, V=128256 vocab → ~70.4 B params (56.3 B MLP, 12 B attn, 2.1 B embed).
2. **Training budget for 15 T tokens**: `6 × 70.4e9 × 15e12 = 6.3e24 FLOPs`.
3. **Single v5p**: 435 years. **Full v5p pod (8960 chips) at 40% MFU**: **44 days**.
4. **Memory**: 140 GB params + 560 GB AdamW optimizer state + 20.9 TB activations (4 gradient checkpoints, batch 4 M tokens) = **~21.6 TB total**.
5. **Minimum chip count to fit**: only 225 (memory alone); 8960 needed for time.
6. **Optimal sharding**: `X_opt = sqrt(2 B N / F) ≈ 2048-way FSDP + 4-way TP`. Validates Ch 5's `B/X < C/(W_ici × 2) = 850` threshold.
7. **Per-chip memory at optimum**: 2.4 GB weights (negligible); **gradient checkpointing dominates memory**.

## Key data points

| Quantity | Value |
|---|---|
| Model | LLaMA-3-70B |
| Layers / D / F / heads / kv / head_dim | 80 / 8k / 28.6k / 64 / 8 / 128 |
| Params | 70.4 B |
| Training tokens | 15 T |
| Total FLOPs | 6.3×10²⁴ |
| Hardware | v5p pod (8960 chips) |
| MFU target | 40% |
| Time to train | **44 days** |
| Topology | 2048 FSDP × 4 TP |

### Memory breakdown

- Params: 140 GB (fp16)
- Optimizer state: 560 GB (fp32 AdamW m + v + master)
- Activations (4 checkpoints, 4 M-token batch): **20.9 TB**
- Total: 21.6 TB

## Techniques referenced

- Gradient checkpointing (per-layer block remat).
- Mixed parallelism: FSDP + TP + sequence parallelism.
- Per-device batch sizing to hit the `B/X` threshold.
- MFU measurement.

## Gaps & caveats

- **40% MFU is aspirational**; real large-scale training hits 30–50%.
- Assumes compute-comm overlap is perfect; XLA under-overlaps by ~20%.
- Doesn't model convergence/tuning overhead (LR schedules, validation, checkpoints, dataset preprocessing).
- **Multi-pod (DCN) scaling not analyzed** — would require 4+ v5p pods to saturate.
- **v5p numbers, pre-v6e/v7** — same derivation applies, constants shift.
- LLaMA-3-70B is the example; different FLOPs/param ratios (MoE, very long context) change the optimal topology.

## Connections

- [concepts/mfu](../concepts/mfu.md) / [step-time](../concepts/step-time.md) / [tokens/s]
- [concepts/training-memory-budget](../concepts/training-memory-budget.md)
- [concepts/fsdp](../concepts/fsdp.md) / [tensor-parallelism](../concepts/tensor-parallelism.md)
- [codebases/maxtext](../codebases/maxtext.md) — reference implementation of a LLaMA trainer on TPU.
- This wiki's gemma4 program is the **E4B-scale** analogue of this chapter's 70B workflow.

## See also

- [Ch 5 — Parallelism](2025-scaling-book-ch5-training.md)
- [Ch 8 — Applied Inference (LLaMA-3-70B serving)](2025-scaling-book-ch8-applied-inference.md)

## Sources

- `raw/code/scaling-book/applied-training.md`
- Upstream: <https://jax-ml.github.io/scaling-book/applied-training>
