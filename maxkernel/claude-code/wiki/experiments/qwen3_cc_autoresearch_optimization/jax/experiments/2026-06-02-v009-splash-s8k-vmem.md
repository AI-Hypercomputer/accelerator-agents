---
title: "Qwen3-8B jax v009 — splash @ seq8192 + scoped-VMEM fix"
type: experiment
hypothesis: qwen3-jax-splash-attention
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-splash-2026-06-02
verdict: supported
tags: [qwen3-cc, jax, splash, remat, seq8192, target-seq, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v009 — splash @ seq8192 + scoped-VMEM fix

## Hypothesis under test

**Hypothesis**: splash+remat reaches the **program-target seq 8192** (bs=1) — the
seq the dense path can't run — with sane loss and a usable MFU.

**Mechanism**: `--use_remat --use_splash --batch_size=1 --seqlen=8192` +
`--xla_tpu_scoped_vmem_limit_kib=98304`, image `qwen3-8b-jax:v006-splash`.
(First attempt preempted on spot; runner re-submitted and this one completed.)

## Results

| Metric | Value |
|--------|-------|
| Loss | 12.094 → 12.068 (varies — splash correct at long seq) |
| Throughput | **42,439 tok/s (5,305/chip)** |
| MFU | **30.4%** (v6e bf16 peak) |
| Steady step | ~1,524 ms (global 8, seq 8192) |
| Exit | 0 |

This is the **program-target sequence length (8192)** — unreachable on the dense
XLA-SDPA path (its `[B,H,8192,8192]` scores ≈ 4 GiB/layer OOM). Splash makes it
feasible at 30.4% MFU. It's a *different shape* from the seq2048 frontier (32.4% @
bs4), so it's a parallel best-for-target-seq, not a regression.

## Profile

- xprof run `2026-06-02-qwen3-jax-v009-splash-s8k`; GCS profile (steps 12–14) + HLO present.

## HLO Dump
- `gs://…/qwen3_cc/2026-06-02-qwen3-jax-v009-splash-s8k/hlo/`.

## Verdict

**supported** — splash+remat reaches the **seq-8192 target** at 30.4% MFU /
5,305 tok/s/chip with stable loss. This is the headline target-shape number for the
lane (dense can't run seq8192 at all). The seq2048 throughput frontier remains
[v008](2026-06-02-v008-splash-vmem-bs4.md) (splash bs4, 32.4%); v009 is the
seq8192-target frontier. Next: push seq8192 batch (bs2 with CE to fit the larger
activations) and seq16384.

## Next hypotheses

- [tokamax CE (jax)](../../../../hypotheses/qwen3-jax-tokamax-ce.md) — CE at seq8192 lets bs rise past 1 (logits at seq8192 are large); test seq8192+CE+bs2.
- [Splash attention (jax)](../../../../hypotheses/qwen3-jax-splash-attention.md) — seq16384 (does splash scale further?).

## Sources
- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v009-splash-s8k/`
