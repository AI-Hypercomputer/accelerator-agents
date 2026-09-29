---
title: "Qwen3-8B jax v037 — scan unroll mechanism probe on the no-offload substrate (bs2 + save_attn)"
type: experiment
hypothesis: qwen3-jax-manual-weight-prefetch
model: qwen3-cc5-jax
variant: "8B/v6e-8"
commit: fork .repo/2026-06-12-v034-wprefetch; image digest bc8d7950
status: complete
verdict: refuted
tags: [qwen3-cc5, jax, v6e-8, scan, unroll, fsdp-collectives, seq8192]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3-8B jax v037 — unroll mechanism probe @ bs2/save_attn

The XLA bug ([v034](2026-06-12-v034-scan-unroll.md)/[v035](2026-06-12-v035-unroll-noacf.md))
blocks unroll≥2 only when pinned-host offload buffers exist. This probe
removes offload: **bs2 + save_attn** keeps the saved attention tensors in
HBM (~9 GB at bs2, fits) — no `S(5)` buffers, so unroll compiles. It can't
beat the bs4 frontier directly (batch deficit), but it PROVES or REFUTES
the AG-overlap mechanism and quantifies the prize for an upstream XLA bug
report. (v037: v036 skipped — user's manual namespace.)

## Hypothesis under test

[qwen3-jax-manual-weight-prefetch](../../../../hypotheses/qwen3-jax-manual-weight-prefetch.md)
mechanism check: at the same substrate, unroll=2 cuts the exposed AG wait
(async-done bucket) vs unroll=1 by ≥ 30% and gains ≥ +2% tok/s/chip.

**Arms (both bs2/seq8192, scan + save_attn + splash + chunked CE + 21-flag
stack [ACF flags dropped — proven worthless in v035]):**
- **A — unroll=1 control** (20 steps)
- **B — unroll=2** (20 steps, profiled 12–14, HLO)

**Outcome mapping**: B ≥ A+2% with async-done collapse → mechanism PROVEN;
hypothesis resolves "blocked by XLA bug at the frontier op-point" (file
upstream; revisit on libtpu update). B ≈ A → mechanism REFUTED (the
scheduler doesn't exploit the unrolled window even without offload) →
hypothesis closes refuted; next = retrospective #3 (int8-AQT).

## Setup

- Hardware: v6e-8, fsdp=8, tp=1, 1 slice of `<your-cluster>`.
- Image: digest-pinned `qwen3-8b-jax@sha256:bc8d7950…`.
- Workload: `alekseyv-qwen3-cc5-jax-v037-unsv2`.

## Results

Both arms clean (exit 0, no OOM, no S(5) error, no NaN; loss series
bit-identical between arms within bf16 noise ≤0.0004).

| Metric | **Arm A: unroll=1 (control)** | Arm B: unroll=2 | B/A |
|--------|-------------------------------|------------------|-----|
| Step (median) | **2,625.7 ms** | 2,713.5 ms | **+3.35% SLOWER** |
| tok/s/chip (131,072 tok/step) | **6,240** | 6,038 | −3.24% |
| MFU | **35.8%** | 34.6% | — |
| Compile | ~17 min | ~38 min (2.2×) | — |

**Mechanism FAIL** (threshold was B ≥ A+2%): even with no offload buffers
in play, unrolling the scan makes things *worse* — the latency-hiding
scheduler does not use the 2-layer window to overlap the next AG; it just
pays a bigger body.

**Side discovery — ARM A BEATS THE bs4 FRONTIER**: bs2 + save_attn + scan
+ 21-flag stack = **6,240 tok/s/chip / 35.8% MFU** vs v027's 6,040/34.6%
(+3.3%). The old bs2 figure (v021: 5,701) was uniform-remat under the
pre-v026 flags; save_attn + the completed stack is worth +9.5% at this
op-point, and the no-offload substrate is more per-token-efficient than
bs4+offload. Validation dispatched as
[v038](2026-06-12-v038-bs2-saveattn-val.md).

## Profile

- **Arm B**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v037-unsv2/plugins/profile/2026_06_12_12_28_30/`
  (steps 12–14; xprof run `2026-06-12-qwen3-jax-v037-unsv2`,
  http://localhost:8791/?run=2026-06-12-qwen3-jax-v037-unsv2). Kept as the
  mechanism-refutation record; deep analysis superseded by the arm-A
  discovery — the v038 validation run carries the frontier-candidate
  profile.
- Arm A: unprofiled (control); its throughput + loss parity are the
  finding; v038 profiles the configuration.

## HLO Dump

`gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v037-unsv2/hlo_b/` (74 files, arm B).

## Verdict

**refuted** (the unroll mechanism — final ledger across the hypothesis:
flags v030–v032 exhausted; unroll×offload = XLA bug v034/v035;
unroll-no-offload = −3.24% here; carry-prefetch = +16 GB infeasible).
**[qwen3-jax-manual-weight-prefetch](../../../../hypotheses/qwen3-jax-manual-weight-prefetch.md)
closes refuted on this libtpu generation** — the 624 ms AG bucket at bs4 is
unreachable by any tested or implementable means; revisit on libtpu update
(the XLA bug evidence package: error verbatim in v034/v035 pages +
before-opt HLO at `.../v034-unr/hlo_a/`).

## Next hypotheses

- [qwen3-jax-bs2-saveattn-frontier](../../../../hypotheses/qwen3-jax-bs2-saveattn-frontier.md) — validate the arm-A discovery as the new frontier (v038, 50 steps + profile; >6,200 sustained).
- None else — the unroll/prefetch branch is closed.

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v037-unsv2/`
- Prior: [v035 (ACF exoneration + bug isolation)](2026-06-12-v035-unroll-noacf.md), [v015 (save_attn numerics)](2026-06-12-v015-save-attn-remat.md)
