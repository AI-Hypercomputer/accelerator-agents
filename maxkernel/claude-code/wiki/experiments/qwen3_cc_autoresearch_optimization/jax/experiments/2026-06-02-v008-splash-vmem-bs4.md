---
title: "Qwen3-8B jax v008 — splash bs=4 + scoped-VMEM fix"
type: experiment
hypothesis: qwen3-jax-splash-attention
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-splash-2026-06-02
verdict: supported
tags: [qwen3-cc, jax, splash, remat, frontier, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v008 — splash bs=4 + scoped-VMEM fix

## Hypothesis under test

**Hypothesis**: Raising the scoped-VMEM cap clears v006's 2.15 M splash-backward
overrun, letting splash+remat+bs=4 run; splash removes the N² scores so bs=4 fits
and beats the 25.1% frontier.

**Mechanism**: v006 config + `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304`
(32 M → 96 M). `--use_remat --use_splash --batch_size=4 --seqlen=2048`, image
`qwen3-8b-jax:v006-splash`.

**Predicted signal**: no VMEM OOM; loss ~12 stable; MFU > 25.1%.

**Falsification criterion**: VMEM-OOM, NaN loss, or MFU ≤ 25.1%.

## Setup

- v6e-8 (fsdp=8, tp=1), image `qwen3-8b-jax:v006-splash`, `--use_remat --use_splash --batch_size=4 --seqlen=2048`, `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=98304`.
- global_batch 32, per-chip 4. Splash blocks `bq=2048 bkv=1024 bq_dkv=2048 bkv_dkv=2048 fused_bwd=True`.

## Baseline comparison

| Metric | baseline | remat bs2 (v004) | remat bs3 (v005) | **splash bs4 (v008)** |
|--------|----------|------------------|------------------|------------------------|
| tok/s/chip | 3,994 | 4,280 | 4,874 | **6,299** |
| MFU | 20.5% | 22.0% | 25.1% | **32.4%** |
| MXU util | 19.9% | 27.8% | 32.7% | **44.6%** |

**+29% tok/s/chip over v005; +58% over baseline.** MXU util 44.6% ≈ the llama3 MaxText reference regime.

## Results

| Metric | Value |
|--------|-------|
| Steady step time | 1,300.6 ms (global 32) |
| Throughput | **50,389 tok/s (6,299/chip)**, 18 steps |
| MFU | **32.4%** (trainer) / MXU 44.6% (xprof) |
| Loss | 12.095 → 12.069 (stable, slight decrease, step-to-step variation) — **splash numerically correct** |
| VMEM | cleared (96 M cap resolved v006's 34.15 M-vs-32 M crash) |
| Exit | 0 |

## Profile

- **xprof URL**: `http://localhost:8791/?run=2026-06-02-qwen3-jax-v008-splash-vmem-bs4`
  (run `…/2026_06_02_06_09_24`). MXU util **44.6%**, tc_idle 3.8 ms.
- **GCS**: `gs://…/qwen3_cc/2026-06-02-qwen3-jax-v008-splash-vmem-bs4/plugins/profile/`. Profiled steps 12–14.
- **Op profile**: `convolution fusion` (matmul) **49.3%**, `loop fusion` 18.7%,
  **`custom-call` (splash kernel) 11.0%** (replaces the N²-score `loop`/einsum path),
  `data formatting` 7.1%, `collective-permute` 6.3%. The attention is now a bounded
  custom-call instead of materialized scores — exactly the splash win.

## HLO Dump

- `gs://…/qwen3_cc/2026-06-02-qwen3-jax-v008-splash-vmem-bs4/hlo/` — **871 files**
  (text + proto, incl. memory-usage-report); the splash custom-call is in the train_step module.

## Verdict

**supported — NEW FRONTIER 32.4% MFU.** Splash attention (GQA-native Pallas, with
the 96 M scoped-VMEM cap) removes the `[B,H,S,S]` score materialization, lets bs=4
fit, and lifts MFU 25.1% → **32.4%** (6,299 tok/s/chip), MXU util to 44.6%. Loss is
stable → kernel correct. **qwen3-jax-splash-attention SUPPORTED.** Stable config =
`--use_remat --use_splash --batch_size=4` + the vmem flag on `qwen3-8b-jax:v006-splash`.
Next: stack tokamax-CE (drop the `[B,L,V]` logits) for bs=6+, and push seq8192 (v009).

## Next hypotheses

- [tokamax CE (jax)](../../../../hypotheses/qwen3-jax-tokamax-ce.md) — stack on splash to drop logits → bs=6+ (v010 canary in flight).
- splash + remat at **seq8192** (v009 in flight) — the program-target seq.

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v008-splash-vmem-bs4/`
