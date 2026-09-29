---
title: "Qwen3 jax — per-chip batch scaling"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "1.5-2.5x throughput / MFU"
confidence: high
effort: S
origin: 2026-06-02-qwen3-jax-v6e8-baseline
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 jax — per-chip batch scaling

> [!warning] Refuted 2026-06-02 by [v001 (bs=2)](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v001-batch-bs2.md) + [v002 (bs=4)](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v002-batch-bs4.md): both OOM at compile (bs=2 → 43.12 G, bs=4 → 91.72 G vs 31.25 G HBM). Batch cannot scale on the current path — the dense `[B,H,S,S]` scores + no gradient remat cap it at bs=1. **Blocked behind [grad-remat](qwen3-jax-grad-remat.md) + [splash](qwen3-jax-splash-attention.md)**; revisit batch after those land.

*Hypothesis*: Raising `--batch_size` (per-chip batch > 1) lifts MXU occupancy
above the baseline 19.9% with no code change, until HBM (Adam state +
activations + `[B,L,V]` logits) binds.

*Mechanism*: The [baseline](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v6e8-baseline.md)
ran per-chip batch 1 (global 8 on fsdp=8) at ~20% MXU. More work per step
amortizes fixed per-step overhead (collectives + dispatch).

*Falsification criterion*: throughput/chip does not rise with batch, or OOM
before any gain → refuted.
