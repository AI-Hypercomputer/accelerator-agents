---
title: "Exp 9 — Apply exp 8 autotune-best splash config (ACCEPTED, +1.1 % TPS / +0.4 pp MFU)"
type: experiment
tags: [llama3, torchax, splash, kernel-autotune, accepted]
hypothesis: llama3-torchax-splash-kernel-autotune
model: llama3-8b-torchax
created: 2026-04-25
updated: 2026-04-25
commit: "v6e8-llama3-8b-torchax-20260425-exp9-splash-autotuned-bs4"
branched_from: v6e8-llama3-8b-torchax-20260425-exp3-splash-bs4
verdict: supported
---

Full-training validation of the [exp 8 kernel autotune](2026-04-25-exp8-splash-kernel-autotune-potential.md):
swap the splash `BlockSizes` in [`splash_attn.py`](../splash_attn.py) for the
universal-winner config (symmetric `block_q=block_kv=1024`, `block_kv_dkv =
seq_len`, `use_fused_bwd_kernel=True`). Result: **+1.1 % TPS, +0.4 pp MFU vs
exp 3** — clean above noise, lands at the low end of the +1-3 % prediction
based on attention's share of step time at this shape.

## Setup

Same image-and-config as [exp 3](2026-04-25-exp3-splash-bs4-accepted.md)
**except**:
- Image: **`hf-v3`** (just `splash_attn.py` changed; everything else
  identical to `hf-v2`).
- Splash `BlockSizes`: production was `block_kv=512` + `use_fused_bwd_kernel=False`;
  exp 9 uses `block_kv=1024` + `use_fused_bwd_kernel=True` + `block_kv_dkv=2048`.

Diff: 12 lines in [`splash_attn.py:42-58`](../splash_attn.py).

## Results

| Metric | Baseline (no splash, bs=2) | Exp 3 (stale splash, bs=4) | **Exp 9 (autotuned splash, bs=4)** | Δ vs exp 3 |
|---|---|---|---|---|
| Cold compile | 92 s | 72 s | 72 s | flat |
| Steady step time | 446 ms | 572 ms | **560 ms** | −2.1 % |
| Throughput (aggregate) | 36,729 TPS | 57,154 TPS | **57,799 TPS** | **+1.1 %** |
| Per-chip TPS | 4,591 | 7,144 | **7,225** | +1.1 % |
| MFU (MaxText formula) | 22.9 % | 35.7 % | **36.1 %** | **+0.4 pp** |
| vs MaxText reference 44.6 % MFU | 21.7 pp | 8.9 pp | **8.5 pp** gap | −0.4 pp closer |

Δ is above the ±0.5 % step-to-step noise observed across exp 3's run, and
matches exp 8's [back-of-envelope prediction](2026-04-25-exp8-splash-kernel-autotune-potential.md#o6-end-to-end-tps-impact-is-bounded-by-attentions-share-of-step-time)
of +1.5–3.5 % at this shape. Lands at the low end (1.1 % vs 2.4 % predicted)
because the in-loop attention call is fast enough that the per-step ~12 ms
saved is ~half what the kernel benchmark suggested — likely some of the
saving is absorbed by the FSDP all-gather scheduling around attention
(which the kernel-only sweep doesn't see).

## Profile

- **xprof browser URL**: [2026-04-25-llama3-8b-exp9-splash-autotuned](http://localhost:8791/?run=2026-04-25-llama3-8b-exp9-splash-autotuned).
- **GCS**: `gs://<your-bucket>/autoresearch/2026-04-25-llama3-8b-exp9-splash-autotuned/plugins/profile/<session>/`.
- **On-disk**: [`raw/profiles/2026-04-25-llama3-8b-exp9-splash-autotuned/`](../../../../../raw/profiles/2026-04-25-llama3-8b-exp9-splash-autotuned/) (gitignored, both ranks).
- **Steps captured**: 7.

## Verdict + reasoning

**Supported (accepted).** Becomes the new program HEAD for `bs=4 seq=1024`
runs. Loss trajectory bit-identical (11.7500 throughout, same as exp 3). Step
0 cold-step has the same ~1.2 s warmup overhead.

The kernel autotune from exp 8 lands an end-to-end gain that survives XLA
fusion + the surrounding FSDP collectives — the scenario that
[gemma4 exp 33 (Pallas RMSNorm, −8.1 %)](../../../gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp33-pallas-rmsnorm-rejected.md)
warned us about. Splash is *not* a fresh custom-call replacing an XLA fusion —
it's already a custom-call; we only changed its internal tiling — so the
"swap-fusion-for-Pallas" tax doesn't apply.

## Observations

- **Methodology validated**: 3-min kernel-only autotune (171 configs ×
  3 shapes) → +1.1 % end-to-end TPS that would have cost ~10× the cluster
  time to find via direct training A/Bs. Pattern transfers to the next
  Pallas kernel we add (cross-entropy, RMSNorm, etc.).
- **Most of the kernel-side win is NOT visible end-to-end** — exp 8 measured
  +30 % on fwd+bwd kernel time (from 1.42 → 0.96 ms/layer), which works out
  to ~13.6 ms / step saved at exp 3 shape. We see ~12 ms / step saved
  end-to-end → the saving is real but slightly less than the kernel-only
  reading. Likely cause: exp 3 was already partially overlapping attention
  with adjacent collectives, so a faster attention kernel doesn't fully
  translate to a faster step.

## Next hypotheses generated

1. ✅ Done — kernel autotune at seq=8192 (program target) → exp 10 (running).
2. **Selective remat** to recompute the FFN intermediate `[B, L, 14336]`
   (the OOM driver in exp 4 / exp 6) — would unlock bs=8 or seq=2048+
   at bs=4. Effort: M.
3. **Tokamax `LinearSoftmaxCrossEntropyLoss`** — Llama 3 has no
   logit-softcapping, so the [gemma4 exp 43 blocker](../../../gemma4_autoresearch_optimization/jax/experiments/2026-04-23-exp43-jax-tokamax-ce-rejected.md)
   does NOT apply. Direct memory saving ~128 MiB / chip (avoids materializing
   `[B, V] = [4×1024, 128256]` logits at fwd). Effort: M.
4. **Scan-over-layers** via `torchax.train.ScannedModule` — collapses the
   32-layer unrolled HLO into a 1-iteration scan body for ~32× compile-time
   reduction; steady throughput likely neutral. Effort: M.
5. **fp32 master weights** (program target) once memory unlocks. Effort: M.

## See also

- [Exp 3 (splash bs=4, stale config)](2026-04-25-exp3-splash-bs4-accepted.md) — the regression target.
- [Exp 8 (kernel-only autotune)](2026-04-25-exp8-splash-kernel-autotune-potential.md) — the source of the new config.
- [`splash_attn.py` on this branch](../splash_attn.py) — the production kernel wrapper with the autotuned values.

## Sources

- [`raw/profiles/2026-04-25-llama3-8b-exp9-splash-autotuned/`](../../../../../raw/profiles/2026-04-25-llama3-8b-exp9-splash-autotuned/)
- `gs://<your-bucket>/autoresearch/2026-04-25-llama3-8b-exp9-splash-autotuned/`
- Workload (XPK): `llama3-8b-exp9-exp9-splash-autotuned-bs4`
