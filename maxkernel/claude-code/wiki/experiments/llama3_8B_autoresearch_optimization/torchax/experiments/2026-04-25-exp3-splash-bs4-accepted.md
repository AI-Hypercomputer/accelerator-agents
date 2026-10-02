---
title: "Exp 3 — Splash + bs=4 (ACCEPTED, +55.6% TPS, +12.8 pp MFU)"
type: experiment
tags: [llama3, torchax, splash, fsdp, batch4, v6e, gke, accepted]
hypothesis: llama3-torchax-splash-attention
model: llama3-8b-torchax
created: 2026-04-25
updated: 2026-04-25
commit: "v6e8-llama3-8b-torchax-20260425-exp3-splash-bs4"
branched_from: v6e8-llama3-8b-torchax-20260425-exp2-splash-bs2
verdict: supported
---

With splash attention installed (from [exp 2](2026-04-25-exp2-splash-bs2-potential.md)),
double the per-chip batch from 2 to **4**. Bs=4 was OOM on the baseline by
1.05 GiB; splash's `O(L)` attention activation memory frees just enough HBM
to fit. **The combined effect is the largest single optimization step in
this program so far: +55.6 % TPS, +12.8 pp MFU.**

## Setup

Same as [exp 2](2026-04-25-exp2-splash-bs2-potential.md) (image `hf-v2`,
splash override active) with `--batch_size=4` (env-only diff).
`global_batch = 4 × 8 = 32 tokens × seq=1024 = 32,768 tokens / step`.

## Results

| Metric | Baseline (bs=2) | Exp 2 (splash bs=2) | **Exp 3 (splash bs=4)** | Δ vs baseline |
|---|---|---|---|---|
| Cold compile | 92 s | 92 s | **72 s** (−22 %) | |
| Steady step time | 446 ms | 437 ms | **572 ms** | |
| Throughput (aggregate) | 36,729 TPS | 37,299 TPS | **57,154 TPS** | **+55.6 %** |
| Per-chip TPS | 4,591 | 4,662 | **7,144** | **+55.6 %** |
| MFU | 22.9 % | 23.3 % | **35.7 %** | **+12.8 pp** |

Step time grows ~28 % (446 → 572 ms) for **2× the tokens-per-step** — the
non-attention compute (FFN, lm_head) is amortized over twice the work, so
per-token cost drops sharply.

## Profile

- **xprof browser URL**: [2026-04-25-llama3-8b-exp3-splash-bs4](http://localhost:8791/?run=2026-04-25-llama3-8b-exp3-splash-bs4) (point xprof at `gs://<your-bucket>/autoresearch/`).
- **Run name**: `2026-04-25-llama3-8b-exp3-splash-bs4`
- **GCS**: `gs://<your-bucket>/autoresearch/2026-04-25-llama3-8b-exp3-splash-bs4/plugins/profile/2026_04_25_18_38_42/`
- **On-disk**: [`raw/profiles/2026-04-25-llama3-8b-exp3-splash-bs4/`](../../../../../raw/profiles/2026-04-25-llama3-8b-exp3-splash-bs4/) (gitignored, 617 MiB, both ranks).
- **Steps captured**: 7.

## Verdict + reasoning

**Supported (accepted).** Both component changes contribute, and they are
synergistic: splash unlocks bs=4 which lifts compute density, which lifts
the per-chip MFU floor. This becomes the new program target — subsequent
experiments branch from here.

## Observations

- **HBM headroom at bs=4 + splash** — fits with margin (no OOM during compile).
  Bs=8 still OOMs by 7.36 GiB ([exp 4](2026-04-25-exp4-splash-bs8-rejected.md))
  — the headroom is not enough to double again.
- **The MaxText recipe XLA flags still don't help here** —
  [exp 7](2026-04-25-exp7-splash-xla-bs4-rejected.md) replicates this config
  and adds the recipe XLA flags; result is essentially flat (35.0% vs 35.7% MFU).
  At per-chip B·L = 4,096 the FSDP all-gather is still hidden behind compute.

## Next hypotheses generated

- Try **fp32 master weights with bf16 compute (AMP)** — the program target
  is fp32. Memory budget for the optimizer state ~doubles; will likely
  require dropping batch to 3 or moving to bs=2.
- **bs=4 + seq=2048 = bs=4 seq=2048**: see [exp 6](2026-04-25-exp6-splash-bs4-seq2k-accepted.md)
  for whether the doubled compute-per-step still fits HBM.
- Per-chip scaling vs MaxText's 44.6 % MFU still leaves ~9 pp on the table.
  Most likely sources: kernel choice (HF Llama's matmul ordering vs
  MaxText's), framework overhead in the JittableModule wrapper, or
  scheduler differences. Worth a profile-led HLO comparison.

## See also

- [Exp 2 (splash bs=2)](2026-04-25-exp2-splash-bs2-potential.md) — precondition.
- [Exp 4 (splash bs=8)](2026-04-25-exp4-splash-bs8-rejected.md) — bs=8 OOM.
- [Exp 5 (splash bs=2 seq=2048)](2026-04-25-exp5-splash-seq2k-accepted.md) —
  same B·L=4096 a different way; ~equivalent MFU.
- [Exp 7 (splash + XLA flags bs=4)](2026-04-25-exp7-splash-xla-bs4-rejected.md) — XLA flags don't help.

## Sources

- [`raw/profiles/2026-04-25-llama3-8b-exp3-splash-bs4/`](../../../../../raw/profiles/2026-04-25-llama3-8b-exp3-splash-bs4/)
- [`gs://<your-bucket>/autoresearch/2026-04-25-llama3-8b-exp3-splash-bs4/`]
