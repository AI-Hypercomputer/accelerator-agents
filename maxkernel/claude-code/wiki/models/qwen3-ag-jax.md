---
title: "Qwen3 8B — jax"
type: model
architecture: qwen3-ag
lane: jax
sizes: ["8B"]
hardwares: ["v6e-8"]
tags: [qwen3-ag, jax]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 8B — jax

Qwen3 8B (`Qwen/Qwen3-8B`) as a native-JAX (Flax NNX) port on **TPU v6e-8**.
**Scaffold only** — the trainer is not yet written. This lane lights up once the
[torchax](qwen3-ag-torchax.md) baseline is stable and a from-scratch JAX port
becomes a ranked hypothesis; it then serves as the reference ceiling for
cross-lane comparison.

## Target metrics

- Primary: MFU at seq_len=8192 (v6e bf16 peak ≈ 918 TFLOPS/chip).
- Secondary: tokens/sec (and tok/s/chip), step time, peak HBM.

## How to run

```
TODO: native-JAX trainer not yet written.
```

## Variant matrix

| Size | Hardware | Status | Baseline (step / TPS / MFU) | Current best (step / TPS / MFU) | Open hyps | Frontier exp |
|------|----------|--------|-----------------------------|----------------------------------|-----------|--------------|
| 8B | v6e-8 | live | 512 ms / 31,955 TPS / 20.5% MFU @ seq2048 bs8 | 4077.5 ms / 48,218 TPS / 34.6% MFU @ seq8192 bs24 | 2 | [2026-06-02 v036](../experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v036-sc-bs3.md) |

*Baseline captured at seq 2048 (global batch 8). **Current frontier runs at seq 8192 (global batch 24) on v6e-8, climbing to 34.6% MFU / 6,027 tok/s/chip** via Splash attention, layer-scan, selective remat, and custom cross-entropy. See [v033b](../experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v033b-maxtext-ce-s8k-bs1.md), [v034](../experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v034-maxtext-ce-s8k-bs2.md), [v035](../experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v035-maxtext-ce-s8k-bs3.md), and [v036](../experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v036-sc-bs3.md).*

## Cross-variant open hypotheses

Ranked after the [2026-06-02 v036](../experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v036-sc-bs3.md):

1. [Per-chip batch scaling](../hypotheses/qwen3-jax-batch-scaling.md) — scale to bs=4 to find the OOM limit. Effort S.
2. [Selective SparseCore offload](../hypotheses/qwen3-jax-selective-sparsecore-offload.md) — selectively offload FSDP collectives to minimize transfer overhead. Effort S.

## Variant-specific open hypotheses

(none yet)

## Retired hypotheses

- [SparseCore collective offload](../hypotheses/qwen3-jax-sparsecore-offload.md) — Refuted by [2026-06-02 v036](../experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v036-sc-bs3.md). Offloading all collectives slightly degraded step time (+0.6%) due to SC-TC memory transfer/PCIe overhead.
- [Splash attention](../hypotheses/qwen3-jax-splash-attention.md) — GQA-native kernel; avoids `[B,H,S,S]`. Status: **Confirmed** (via `v004-gradient-checkpointing`).
- [Gradient Checkpointing](../hypotheses/qwen3-jax-gradient-checkpointing.md) — Status: **Confirmed**.
- [tokamax streamed cross-entropy](../hypotheses/qwen3-jax-tokamax-ce.md) — drop `[B,L,V]` logits at the lm_head. Status: **Blocked** (refuted during compilation due to SDPA OOM).
- [Per-chip batch scaling](../hypotheses/qwen3-jax-batch-scaling.md) — fill MXU occupancy at 8K context. Status: **Refuted** (compilation OOM; activation + Adam state exceeds memory limit at bs=2).
- [tokamax-ce + batch scaling](../hypotheses/qwen3-jax-tokamax-batch.md) — combine `splash-attention`, `gradient-checkpointing`, and `tokamax-ce` to unlock `batch_size=2` at 8K context. Status: **Refuted** (compiles, but performance regressed to 29.7% MFU).
- [Async FSDP collectives](../hypotheses/qwen3-jax-fsdp-async-overlap.md) — overlap FSDP reduce-scatter and all-gather with compute via XLA flags. Status: **Refuted** (silent no-op).
- [Fused RMSNorm+RoPE Pallas kernel](../hypotheses/qwen3-jax-pallas-rmsnorm-rope.md) — Status: **Refuted** (Pallas autodiff crash; pure-TPU Norms are an anti-pattern as XLA optimally fuses them).
- [Ring Attention Pallas kernel](../hypotheses/qwen3-jax-ring-attention.md) — Status: **Confirmed** (unlocks memory for batch=8 seq=8192 via sp=2 2D FSDP, but performance regressed to 22.8% MFU due to collective overhead and suboptimal block sizes).
- [Fused GLU Pallas kernel](../hypotheses/qwen3-jax-pallas-fused-glu.md) — Status: **Refuted** (MFU regressed to 30.6% due to redundant forward matmuls caused by `jax.remat` and excessive HBM traffic in the custom JAX backward pass).

## Knobs translation matrix

| Technique | 8B/v6e-8 | Notes |
|-----------|----------|-------|
(populated after the lane is implemented)

## Iteration ladder

1. Port Qwen3 8B to native JAX (Flax NNX), matching the torchax baseline number.
2. Apply the MaxText-style XLA flag stack + collective offload.
3. Cross-lane comparison vs the torchax frontier.

## See also

- [Qwen3 8B — torchax](qwen3-ag-torchax.md) — primary lane (live frontier).
- Program: [`../experiments/qwen3_ag_autoresearch_optimization/program.md`](../experiments/qwen3_ag_autoresearch_optimization/program.md)
- Lane README: [`../experiments/qwen3_ag_autoresearch_optimization/jax/README.md`](../experiments/qwen3_ag_autoresearch_optimization/jax/README.md)

## Sources

- `wiki/experiments/qwen3_ag_autoresearch_optimization/jax/` (scaffold).
