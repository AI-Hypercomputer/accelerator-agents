---
title: "Qwen3-8B jax v017 — partial remat (skip-every-5) at seq 8192"
type: experiment
hypothesis: qwen3-jax-partial-remat-8192
model: qwen3-cc5-jax
variant: "8B/v6e-8"
commit: fork v6e8-qwen3-8b-jax-20260612-v017-partial-remat
status: supported
verdict: supported
tags: [qwen3-cc5, jax, v6e-8, remat, seq8192]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3-8B jax v017 — partial remat (skip-every-5) at seq 8192

Tests [qwen3-jax-partial-remat-8192](../../../../hypotheses/qwen3-jax-partial-remat-8192.md)
at the [v015 target-shape frontier](2026-06-12-v015-save-attn-remat.md).

## Hypothesis under test

**Hypothesis**: `JAX_REMAT_SKIP_EVERY=5` (8 of 36 layers keep full
activations, ~3.5 GiB) cuts the recompute tax ~22% → step time < 1475.7 ms
beyond noise, within the 31.25 GB budget (27.62 + 3.5 ≈ 31.1, tight).

**Mechanism**: new env knob in the v017 fork (`i % 5 == 0` layers run
unchecked). One mechanism: remat scope; policy and all other flags identical
to v015 phase 2.

**Predicted signal**: `loop fusion` share drops below 14.9%; step time
~1430–1450 ms; HBM ~31.1 GiB (OOM is a live risk — itself a precise headroom
measurement).

**Falsification criterion**: OOM, or tok/s ≤ 44,361 + 0.5% → refuted.

## Setup

- Hardware: v6e-8, fsdp=8, tp=1, 1 slice of `<your-cluster>`.
- Image: `qwen3-8b-jax:v6e8-qwen3-8b-jax-20260612-v017-partial-remat`.
- Workload: `alekseyv-qwen3-cc5-jax-v017-prem5`.
- Command (diff vs v015 ph2: `JAX_REMAT_SKIP_EVERY=5`):
  ```bash
  export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=81920"
  export JAX_REMAT_SKIP_EVERY=5
  python -u train.py --batch_size=1 --seqlen=8192 --train_steps=20 \
      --use_splash=True --use_tokamax_ce=True --tokamax_ce_impl=chunked_xla \
      --remat_policy=save_attn \
      --profile_dir=<gcs_root> --profile_start_step=12 --profile_steps=3
  ```

## Results

20 steps clean (exit 0); no OOM; no NaN; loss smooth 12.10 → 12.07.

| Metric | v015 (full save_attn remat) | This run (skip-every-5) | Δ |
|--------|------------------------------|--------------------------|---|
| Steady step time | 1475.7 ms | **1390.1 ms** | **−5.8%** |
| Throughput | 44,361 tok/s | **47,146 tok/s (5,893/chip)** | **+6.3%** |
| MFU | 31.8% | **≈ 33.8%** (derived from the step-time ratio; xprof FLOP util 34.1% vs 33.4% corroborates; the runner-quoted trainer line of 31.5% is inconsistent with its own tok/s and flagged as transcription noise) | +2 pp |
| Program HBM | 27.62 GiB | **30.68 GiB** (0.57 GiB headroom) | +3.06 GiB |

## Profile

- **xprof URL**: `http://localhost:8791/?run=2026-06-12-qwen3-jax-v017-prem5`
  (run `…/2026_06_12_04_01_03`); profiled steps 12–14.
- **GCS run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v017-prem5/plugins/profile/2026_06_12_04_01_03/`
- **Op profile** (exclude-idle, master-analyzed): `convolution fusion` 47.4%,
  `custom-call` 23.9%, `loop fusion` **14.4%** (down from 14.9 — the
  recompute tax shrank as predicted), `data formatting` 4.9%, collectives
  ~2.5%. **FLOP util 34.1%** (was 33.4%).
- **Predicted signal: CONFIRMED.**

## HLO Dump

- **GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v017-prem5/hlo/` — 51 files;
  `Total bytes used: 30.68 GiB` (the ~0.43 GiB/exempt-layer model holds:
  27.62 + 8 × 0.43 ≈ 31.1 projected vs 30.68 measured).

## Verdict

**supported — TARGET-SHAPE FRONTIER SHIFT**: 1390 ms / 47,146 TPS / ≈33.8%
MFU @ seq 8192. One knob (`JAX_REMAT_SKIP_EVERY=5`), no semantics change
(remat scope only; loss series tracks v015/v016 step-for-step within noise).
Fork merged to trunk; `latest` rebuilt; the seq-8192 recipe gains the knob.

## Observations

- The remat recompute tax is linear in checkpointed-layer count (loop fusion
  14.9 → 14.4% for −8 layers; step −5.8%); HBM cost ≈ 0.43 GiB per exempt
  layer at bs1/seq8192. Headroom now 0.57 GiB ⇒ skip-every=4 (+1 exempt
  layer, +0.43 GiB) is the last rung; probed as v018.

## Next hypotheses

- [qwen3-jax-partial-remat-8192](../../../../hypotheses/qwen3-jax-partial-remat-8192.md) — v018: skip-every=4 (the edge rung; 0.57 GiB headroom vs +0.43 needed — decisive either way).
- [qwen3-jax-vocab-sharded-ce](../../../../hypotheses/qwen3-jax-vocab-sharded-ce.md) — would free ~2.2 GB → 5 more exempt layers (~+3-4% further); now the highest-leverage open M-effort item.

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v017-prem5/`
- Prior: [v015 (frontier)](2026-06-12-v015-save-attn-remat.md), [v016 (bkv refuted)](2026-06-12-v016-splash-bkv2048.md)
