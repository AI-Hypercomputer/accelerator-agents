---
title: "Qwen3-8B jax v022 — tokamax splash knobs at the frontier shape"
type: experiment
hypothesis: qwen3-jax-tokamax-splash-knobs
model: qwen3-cc5-jax
variant: "8B/v6e-8"
commit: 11e8f9d (trunk; env-only diff)
status: refuted
verdict: refuted
tags: [qwen3-cc5, jax, v6e-8, splash-attention, tokamax, seq8192]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3-8B jax v022 — tokamax splash knobs at the frontier shape

Tests [qwen3-jax-tokamax-splash-knobs](../../../../hypotheses/qwen3-jax-tokamax-splash-knobs.md)
at the exact [v019 frontier recipe](2026-06-12-v019-val50.md).

## Hypothesis under test

**Hypothesis**: tokamax splash + base2_exp + fuse_reciprocal beats the
upstream-jax splash by enough to push past 47,182 tok/s (the splash share is
23.7% of step; llama3 measured +4.4% from this swap).

**Mechanism**: env-only: `USE_TOKAMAX_SPLASH=1 TOKAMAX_USE_BASE2_EXP=1
TOKAMAX_FUSE_RECIPROCAL=1` on the v019 recipe (skip5, no scan). The dispatch
already exists in `splash_attn.py`; the dev tokamax in the image ships the
experimental splash kernel.

**Predicted signal**: `splash_mha_*` custom-call share drops below 23.7%;
step < 1389 ms. Loss parity vs the well-known v019 series (same seed).

**Falsification criterion**: tok/s ≤ 47,182 + 0.5%, or kernel
crash/incompat → refuted; splash topic closes fully on this stack.

## Setup

- Hardware: v6e-8, fsdp=8, tp=1, 1 slice of `<your-cluster>`.
- Image: `qwen3-8b-jax:latest` (trunk `11e8f9d`).
- Workload: `alekseyv-qwen3-cc5-jax-v022-tksp`.
- Command: v019's + the three tokamax-splash envs.

## Results

20 steps clean (exit 0). **Kernel swap proven**: `[tokamax-splash] config:
bq=2048 bkv=1024 … base2=True fuse_recip=True` ×36 layers; no Mosaic errors.

| Metric | v019 (upstream splash) | This run (tokamax splash + knobs) | Δ |
|--------|------------------------|-----------------------------------|---|
| Steady step | 1389.0 ms | 1382.3 ms | −0.48% (noise) |
| Throughput | 47,182 tok/s | 47,411 tok/s | +0.49% — just under the +0.5% bar |
| Loss parity (steps 2–7) | reference | max Δ 0.0002 | ✓ numerics-equivalent |

## Profile

- **xprof URL**: `http://localhost:8791/?run=2026-06-12-qwen3-jax-v022-tksp`
  (run `…/2026_06_12_05_18_48`); profiled steps 12–14.
- **GCS run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v022-tksp/plugins/profile/2026_06_12_05_18_48/`
- Flat-shape comparison is the complete signal.

## HLO Dump

- **GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v022-tksp/hlo/` — 51 files.

## Verdict

**refuted** (marginal — +0.49% sits exactly at the noise bar; treat as
neutral). base2_exp + fuse_reciprocal don't reproduce llama3's +4.4% here —
that lane's gain was measured against the jax-experimental splash on a
different stack era/shape. **llama3 splash-knob transfers are now 0-for-3 on
this QK-norm GQA stack** (blocks v002-tuning, bkv v016, perf knobs v022).
Splash topic fully closed. The tokamax splash is numerics-equivalent and
~at-par — kept as a non-default option.

## Next hypotheses

- None from this run — splash topic closed. The lane's remaining open items
  are the retrospective's L/M tail (host offload, fused QK-norm+RoPE kernel,
  scan-body save policy) and the data-formatting drilldown, executed as a
  pure profile analysis (no TPU run) — see the lane log entry of this date.

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v022-tksp/`
- Prior: [v019 (frontier)](2026-06-12-v019-val50.md), [v016 (block transfer weak)](2026-06-12-v016-splash-bkv2048.md)
