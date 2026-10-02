---
title: "Qwen3 jax — vocab-sharded chunked CE (dw [H, V/8] per chip)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: open
expected_gain: "−3.2 GB at the CE (dw 2.34→0.3 GB + fwd pad sharded); marginal alone for bs3 (31.4 vs 31.25)"
confidence: low
effort: L
origin: 2026-06-12-v014-bs3-ce
hlo_prefilter: passed
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — vocab-sharded chunked CE

*Hypothesis*: Sharding the chunked CE over the vocab axis (w shard `[V/8, H]`
per chip; per-shard partial lse combined via psum-of-logsumexp) shrinks the
f32 dw buffer from `[H,V]` 2.34 GB to `[H,V/8]` 0.3 GB and the fwd pad
likewise — recovering ~3.2 GB at bs3 ([v014](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v014-bs3-ce.md)
missed by 3.40 GB, so this is marginal alone; combine with
[save_attn remat](qwen3-jax-remat-save-attn.md) if v015 lands).

*Mechanism*: shard_map over the fsdp axis with w sharded on V; cross-shard
logsumexp combination (all_gather of per-shard lse + logsumexp — NOT
all_gather inside the same-axis collectives; needs the canonical dedicated-
axis treatment per the CE topic's refuted-pattern list).

*Falsification criterion*: numerics offset beyond bf16 noise, or net memory
saving < 2 GB (XLA re-materializes elsewhere), or step-time regression > 3%.

*Stub — expand if v015's remat path doesn't cover the remaining gaps.*

> [!warning] Deprioritized 2026-06-12 (post v018)
> (a) tokamax CE has no partial-vocab semantics — cross-shard lse requires a
> hand-rolled kernel (effort L, not M); (b) [v018](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v018-prem4.md)
> showed negative returns when spending freed HBM near the ceiling.
