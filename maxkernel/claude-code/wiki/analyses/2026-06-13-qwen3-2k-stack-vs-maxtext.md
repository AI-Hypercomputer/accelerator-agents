---
title: "Qwen3-8B @ seq-2048 on v6e-8: jax lane vs MaxText — best-vs-best verdict and the remaining-levers ledger"
type: analysis
tags: [qwen3-cc5, jax, maxtext, v6e-8, seq2048, charles-v6e, stack-comparison]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3-8B @ 2k: jax lane vs MaxText — verdict

User directive: *"keep optimizations going, compare to maxtext, find ways
to push it further."* Both stacks were taken to their measured ceilings at
seq2048 on charles-v6e (identical architecture, identical chips, tok/s/chip
as the accounting-free metric).

## Best-vs-best scoreboard

| Stack | Best config | Step | tok/s/chip | MFU (our acct.) | Ceiling evidence |
|---|---|---|---|---|---|
| MaxText | bs12, flash-2048-fused, remat custom + host offload, **bf16 logits**, 23-flag recipe | 3.106 s | **7,915** | ≈40.7% | bs20 = 6,601 (−16.6%, offload DMA stops hiding); bs24 OOM (+129 MiB) — [mt-v002](../experiments/qwen3_cc5_autoresearch_optimization/maxtext/experiments/2026-06-13-mt-v002-2k.md)/[mt-v003](../experiments/qwen3_cc5_autoresearch_optimization/maxtext/experiments/2026-06-13-mt-v003-2kbs24.md) |
| **jax lane** | bs5, splash-2048/1024-fused, scan + save_qkv_ctx_hbm + context ckpt, **f32 logits (plain CE)**, 21-flag | 1.301 s | **7,871 (50-step certified)** | **40.5%** | bs6 OOM (f32 CE pair, [v059](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v059-2kbs6.md)); bs8-via-chunked-CE = +1.5% but **invalid** on numerics ([v061](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v061-2kce8.md)/[v062](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v062-2kce5par.md)) |

**Δ = −0.56% (7,871 vs 7,915) — a statistical tie.** Per-token cost
0.1270 vs 0.1264 ms/token/chip. Both stacks converge to the same wall from
opposite directions: MaxText runs huge batch with offloaded activations and
bf16 logits; we run modest batch with in-HBM context-checkpointed remat and
exact f32 CE.

## What this wave measured (2026-06-13, v063 + mt-v002/v003)

| Probe | Result |
|---|---|
| [scan unroll=2](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-13-v063-2kunroll2.md) (jax) | **refuted −4.1%**: ARS bucket FLAT (157.3 vs 157.8 ms — XLA schedules the unrolled body atomically; zero overlap); regression = body overhead (+34.9 loop fusion, +19.7 state unpack). Bonus: S(5) unroll bug confirmed offload-specific. |
| MaxText bs ladder | bs12 optimum; bs20 −16.6%; bs24 OOM. Their 2k wall = bf16 logits buffer (13.91 G @ bs24). |

## Why the stacks tie — structural reading

- **Compute core is identical-grade**: both use the same splash/flash kernel
  family at the same block sizes; conv-fusion-class matmul time per token is
  equal within noise (our MXU 51.9% at bs5; their TFLOP/s implies similar).
- **Each stack's residual inefficiency is a different ~5%**: ours is the
  **exposed gradient RS (158 ms = 12.1%)** that scan surfaces (unrolled XLA
  hid it, but unrolled+flags doesn't compile in budget); theirs is **offload
  DMA pressure** that bounds their batch curve and bleeds at bs>12.
- Accounting note: MaxText self-reports 42.3% MFU (388/918) — their
  per-token FLOP convention is ~15% higher than ours; tok/s/chip is the
  comparable number throughout.

## Remaining-levers ledger ("push it further" candidates)

| Lever | Expected | Effort | Gate |
|---|---|---|---|
| Manual async RS pipelining (explicit RS-start/done interleave in the scan body, collective-matmul-style) | up to +12% ceiling, realistically less | **L** | Low prior: 8k manual-prefetch lineage refuted; v063 proves flags/unroll can't do it |
| bf16-logits plain CE (MaxText's trick — halves CE memory, reopens bs6–bs8) | +1.5–3% | M | **Rule-8 numerics gate**: logit quantization; requires the v062-style same-data parity test |
| Chunked CE with masked padding (fix the [+0.006 phantom-mass offset](../observations/chunked-ce-padded-vocab-loss-offset.md)) | +1.5% (bs8 route) | M (kernel fix, likely upstream tokamax) | Human review of the offset finding (also touches the 8k certified stack) |
| int8 weight-only AQT | conv-BW headroom | L | Human numerics waiver (standing) |
| 2 more v6e-8 slices on charles-v6e | parallel experiments, not step-time | — | Foreign zombie Kueue reservations (charles-rl-mh-2026-01-03-*) hold 16 of 24 chips |

Everything S-effort at this op-point has been spent: batch (closed both
stacks), flags (subsumed/closed), io layout (shape-gated null), splash
blocks (closed at 8k, same kernel), unroll (refuted), CE swaps
(invalid/retired).

## Bottom line

**The jax lane matches MaxText at seq-2048 (−0.56%, within noise) just as
it does at seq-8192 (+0.09%)** — the user's stack is at parity with the
reference at both op-points, 40.5% MFU certified at 2k. Further gains
require either L-effort structural work (manual RS overlap), numerics-gated
changes (bf16 logits / fixed chunked CE), or policy decisions (int8 waiver,
quota cleanup).

## Sources

- Experiments: v052–v063 (jax), mt-v002/mt-v003 (maxtext) — all on charles-v6e, 2026-06-12/13.
- Profiles: GCS under `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-1{2,3}-qwen3-*`.

## See also

- [Model page](../models/qwen3-cc5-jax.md) · [8k phase summary](2026-06-12-maxtext-gap-closing-phase-summary.md) · [chunked-CE offset observation](../observations/chunked-ce-padded-vocab-loss-offset.md)
