---
title: "Qwen3-8B jax v028 — splash block sweep @ bs4/seq8192 frontier shape"
type: experiment
hypothesis: qwen3-jax-splash-fused-bwd
model: qwen3-cc5-jax
variant: "8B/v6e-8"
commit: trunk 174efd7 (image latest = v024-offload; launch-only env-var diff)
status: complete
verdict: refuted
tags: [qwen3-cc5, jax, v6e-8, splash, kernel-blocks, seq8192]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3-8B jax v028 — splash block sweep at the bs4 frontier

Launch-only sweep of the splash block configuration (env vars in
`splash_attn.py`; fused bwd already default-on) at the
[v027](2026-06-12-v027-bs4-mtfl.md) frontier op-point.

## Hypothesis under test

[qwen3-jax-splash-fused-bwd](../../../../hypotheses/qwen3-jax-splash-fused-bwd.md):
a block config beats the current 2048/1024 hybrid by ≥ 1% step at
bs4/seq8192 (> 6,100 tok/s/chip).

**Phases:**
- **A — llama3 autotune winner, symmetric 1024**: `SPLASH_BQ=1024 SPLASH_BKV=1024 SPLASH_BKV_COMPUTE=1024 SPLASH_BQ_DKV=1024 SPLASH_BKV_DKV=1024 SPLASH_BKV_DKV_COMPUTE=1024`
- **B — MaxText match, all 2048**: `SPLASH_BKV=2048 SPLASH_BKV_COMPUTE=2048` (bq/dkv already 2048)

**Falsification criterion**: both phases within ±1% of v027's 6,040
tok/s/chip → refuted (third flat result on this stack; splash block-config
direction closed).

## Setup

- Hardware: v6e-8, fsdp=8, tp=1, 1 slice of `<your-cluster>`.
- Image: `qwen3-8b-jax:v6e8-qwen3-8b-jax-20260612-v024-offload` (= latest).
- Workload: `alekseyv-qwen3-cc5-jax-v028-spblk`.
- Both phases: bs4 seq8192, scan + offload_attn + splash + chunked CE f32-x + 25-flag stack; 20 steps each; phase B profiled steps 12–14 + HLO.

```bash
# Phase A (prepended to v027 launch command):
SPLASH_BQ=1024 SPLASH_BKV=1024 SPLASH_BKV_COMPUTE=1024 \
SPLASH_BQ_DKV=1024 SPLASH_BKV_DKV=1024 SPLASH_BKV_DKV_COMPUTE=1024 \
python -u train.py ... [v027 args]

# Phase B:
SPLASH_BKV=2048 SPLASH_BKV_COMPUTE=2048 \
python -u train.py ... [v027 args, + profile at steps 12-14]
```

## Results

Both phases clean (exit 0, no OOM, no NaN). **Neither phase reaches the 6,100
bar.** Phase A is slower than baseline; Phase B is neutral.

**Key comparison (both vs v027 baseline 6,040 tok/s/chip / 5,425.0 ms):**

| Metric | Phase A (sym-1024) | **Phase B (all-2048)** | v027 baseline | v016 (bkv2048, bs1) |
|--------|--------------------|------------------------|---------------|---------------------|
| Block config | bq=bkv=dkv=1024 | bq=bkv=dkv=2048 | bq=2048, bkv=1024, dkv=2048 | bkv=2048 (fwd only, bs1) |
| Median step (ms) | **5,561.8** | **5,419.2** | 5,425.0 | flat vs bkv=1024 |
| tok/s/chip | **5,892** | **6,044** | 6,040 | flat |
| vs baseline | **−2.5%** (−148/chip) | **+0.1%** (+4/chip) | — | flat |
| vs 6,100 PASS bar | **FAIL** | **FAIL** (−56/chip) | — | — |
| Phase B vs Phase A step delta | — | **−142.6 ms faster** | — | — |

**Phase-by-phase step tables:**

Phase A (sym-1024), worker-0:

| Step | Step ms | tok/s total |
|------|---------|-------------|
| 0 | 54,912 | compile |
| 1 | 44,750 | warmup |
| 2–11 | 5,556–5,566 | steady |
| 12–19 | 5,560–5,567 | steady |
| **Median 2–19** | **5,561.8** | **47,132 (5,892/chip)** |

Phase B (all-2048), worker-0:

| Step | Step ms | tok/s total |
|------|---------|-------------|
| 0 | 59,133 | compile |
| 1 | 49,288 | warmup |
| 2–11 | 5,414–5,422 | steady |
| 12 | 5,454 | profile capture (+34 ms) |
| 13–14 | 5,419–5,420 | profiled |
| 15–19 | 5,419–5,425 | steady |
| **Median 2–11+15–19** | **5,419.2** | **48,355 (6,044/chip)** |

Loss sanity: Phase A step-0 loss = 12.1033 → 12.0480 (step 19); Phase B 12.1033 → 12.0481. Both identical at step 0 and within bf16 noise throughout — semantics unchanged. No NaN in either phase.

**Cross-phase insight**: bq=2048 beats bq=1024 by ~142 ms/step (+2.6%). This confirms the query block size of 2048 is correct for this workload; the v027 hybrid (bq=2048, bkv=1024, dkv=2048) already had the right bq. The only open dimension was bkv, and Phase B's result (bkv=2048 ≈ neutral, Phase A's bkv=1024 symmetric ≈ −2.5%) confirms bkv=1024 was optimal — consistent with v016's finding at bs1.

## Profile

**xprof URL**: http://localhost:8791/?run=2026-06-12-qwen3-jax-v028-spblk
**Run name**: `2026-06-12-qwen3-jax-v028-spblk` (Phase B captured at `2026_06_12_07_51_46`)
**xprof direct URL**: http://localhost:8791/?run=2026-06-12-qwen3-jax-v028-spblk/2026_06_12_07_51_46
**GCS path**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v028-spblk/plugins/profile/2026_06_12_07_51_46/` (4 files)
**Local path**: [`raw/profiles/2026-06-12-v028-spblk/`](../../../../../raw/profiles/2026-06-12-v028-spblk/)
**Steps captured**: 12–14 (Phase B, steady-state steps)
**Contents**: xprof trace (TPU v6 Lite, Phase B all-2048 config), 2 hosts × 4 chips = 8 chips.

Phase B profile at 5,419 ms/step is nearly identical to v027's 5,425 ms profile — the all-2048 config is effectively a neutral swap of the fwd-bkv dimension only.

**Kernel-level A/B (profile-analyzer; per-chip ms/step):**

| Metric | v027 (bkv=1024) | v028-B (bkv=2048) | Delta |
|---|---|---|---|
| splash custom-call bucket | 1,448 (26.7%) | 1,445 (26.6%) | −0.4% |
| `splash_mha_fwd_residuals` | 816 | 810 | −0.8% |
| `splash_mha_dkv` | 636 | 636 | 0 (already bkv_dkv=2048 in both) |
| Step | 5,425 | 5,419 | −0.1% |

Genuine insensitivity, not a masked win: no offsetting regression in any
other bucket (conv 46.1%, async-done 11.5%, loop fusion 7.7% — all match
v027).

## HLO Dump

**GCS path**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v028-spblk/hlo/` (50 files)

Main module `module_0145.jit_train_step` (~1.36 MB). **Block config
verified in kernel metadata** (the env took effect):
`splash_mha_fwd_residuals` shows `block_kv=2048, block_kv_compute=2048`
(v027: 1024/1024); `splash_mha_dkv` shows `block_kv_dkv=2048` in BOTH runs
(the bwd was already at 2048 — phase B only moved the fwd tile). Module
structure otherwise identical to v027 (same scan whiles, custom-call and
copy-start/done counts).

## Verdict

**refuted**

The falsification criterion is exactly met: both phases are within ±1% of the v027 baseline (Phase A −2.5%, Phase B +0.1%). Neither reaches the 6,100 PASS bar.

This is the **third flat splash block-config result** on this stack:
1. [v016](2026-06-12-v016-splash-bkv2048.md): bkv=2048 flat at bs1/seq8192 (−0.2%).
2. [v022](2026-06-12-v022-tokamax-splash.md): tokamax perf knobs (base2/fuse_recip) neutral at bs1 (−0.48%).
3. **v028**: sym-1024 −2.5%; all-2048 +0.1% — both within noise.

**Interpretation**: the splash kernel cost (26.7% / 1,448 ms/step at bs4) is a fundamental arithmetic cost for the GQA + QK-norm attention at this shape, not a VMEM-tile-scheduling artifact. There is no env-var-accessible block configuration that changes it materially. The only remaining lever against this bucket would be a fundamentally faster attention algorithm — e.g., a custom Mosaic kernel that fuses QK-norm into the attention kernel (avoiding the pre-scale + re-project memory round-trip), or a sub-quadratic attention approximation (out of scope for this optimization loop's phase).

**Confirmed finding**: bq=2048 is strictly better than bq=1024 by ~142 ms/step (+2.6%). This is consistent with the larger query-block reducing the number of softmax reductions per forward pass. The v027 hybrid config (bq=2048, bkv=1024, dkv=2048) was already optimal on the query dimension. The kernel-config direction is closed.

## Next hypotheses

- [qwen3-jax-bs5-probe](../../../../hypotheses/qwen3-jax-bs5-probe.md) — with splash configs exhausted, batch scaling is the highest-ROI remaining lever; actual bs4 HBM = 23.72 GiB (slope 4.06 GiB/bs), bs5 projects 27.78–28.19 GiB (89–90% cap); low confidence but S effort; probe before moving to L-effort directions.
- [qwen3-jax-mlp-only-remat](../../../../hypotheses/qwen3-jax-mlp-only-remat.md) — if bs5 OOMs, MLP-only checkpointing (skip splash layers to avoid the v012 NaN) can recover ~8 GiB/step for bs6; also reduces the peak of the dominant conv-fusion bucket; medium effort.

## Sources

- Profile (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v028-spblk/plugins/profile/2026_06_12_07_51_46/`
- HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v028-spblk/hlo/`
- Prior: [v027 (frontier baseline)](2026-06-12-v027-bs4-mtfl.md), [v016 (bkv2048 flat at bs1)](2026-06-12-v016-splash-bkv2048.md), [v022 (tokamax knobs neutral)](2026-06-12-v022-tokamax-splash.md)
