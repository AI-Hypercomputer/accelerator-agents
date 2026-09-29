---
title: "Exp 7 — Splash + bs=4 + MaxText XLA recipe flags (REJECTED, −0.7 pp MFU vs exp 3)"
type: experiment
tags: [llama3, torchax, splash, xla-flags, rejected]
hypothesis: llama3-torchax-xla-recipe-flags
model: llama3-8b-torchax
created: 2026-04-25
updated: 2026-04-25
commit: "v6e8-llama3-8b-torchax-20260425-exp7-splash-xla-bs4"
branched_from: v6e8-llama3-8b-torchax-20260425-exp3-splash-bs4
verdict: refuted
---

Re-test of the [MaxText recipe XLA flag set](../../../../hypotheses/llama3-torchax-xla-recipe-flags.md)
on top of the [exp 3 winner](2026-04-25-exp3-splash-bs4-accepted.md) — so
per-chip B·L = 4,096 (vs exp 1's 2,048). Hypothesis: at higher per-chip
compute density, the recipe's collective-overlap flags now have something
to overlap. Result: **flat (-0.7 pp MFU vs exp 3, within noise)**.

## Setup

Identical to [exp 3](2026-04-25-exp3-splash-bs4-accepted.md) (image `hf-v2`,
splash, `bs=4 seq=1024`) plus the [recipe LIBTPU flags from exp 1](../../../../hypotheses/llama3-torchax-xla-recipe-flags.md)
(VMEM 98304 KiB + LAYOUT_FOR_ALL_REDUCE_SCATTER + DATA_PARALLEL_OVERLAP +
CF_FOR_ALL_GATHER, 11 flags total).

## Results

| Metric | Exp 3 (no flags) | **Exp 7 (with flags)** | Δ |
|---|---|---|---|
| Cold compile | 72 s | **78 s** | +8 % |
| Steady step time | 572 ms | **584 ms** | +2.1 % |
| Throughput (aggregate) | 57,154 TPS | **56,059 TPS** | **−1.9 %** |
| Per-chip TPS | 7,144 | 7,007 | −1.9 % |
| MFU | 35.7 % | **35.0 %** | **−0.7 pp** |

Δ within noise (±2 %). Compile time +6 s.

## Verdict + reasoning

**Refuted again.** Even at exp 3's higher per-chip compute density (B·L =
4,096, ~2× exp 1's), the recipe XLA flags don't move the needle on torchax.

The MaxText 44.6 % MFU baseline that ships these flags runs at B·L = 24,576
(`bs=3 seq=8192`) — ~6× our current density. We are still on the wrong side
of the threshold where collective overlap matters; the FSDP all-gather of
each layer's weights is still hidden behind compute.

This generalizes the heuristic from [exp 1](2026-04-25-exp1-xla-recipe-flags-rejected.md):
**the MaxText recipe XLA flags require per-chip B·L ≳ ~10,000 to pay for
themselves on this trainer.** Don't enable speculatively.

## Next hypotheses generated

- The 9-pp MFU gap to MaxText's 44.6 % is unlikely to come from XLA flags
  alone — most of it is (a) compute density (Maxtext bs=3 seq=8192 vs ours
  bs=4 seq=1024), (b) framework overhead (JittableModule + torchax interop
  vs MaxText's pure-JAX hand-tuned graph). Memory-saving optimizations to
  unlock seq=8192 or bs≥6 are higher EV than further flag combinations.
- Defer revisiting flags until after **scan-over-layers** lands or
  **selective remat** (so we can fit `seq=8192 bs≥1`).

## Profile

Not staged — workload deleted before profile pull. The decision was
already-flat after step 5.

## See also

- [Exp 1 (XLA flags @ baseline)](2026-04-25-exp1-xla-recipe-flags-rejected.md) — same flags, no splash. Same flat result.
- [Exp 3 (splash bs=4)](2026-04-25-exp3-splash-bs4-accepted.md) — current best (without flags).
