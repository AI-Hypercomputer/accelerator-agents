---
title: "Qwen3-8B jax v045 — save norm rsqrt stats on-device @ seq8192 bs3 (marginal frontier)"
type: experiment
hypothesis: qwen3-jax-save-norm-stats
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-save-norm-stats-2026-06-05
status: supported
verdict: supported
tags: [qwen3-cc, jax, remat, rmsnorm, qk-norm, save-only-these-names, on-device, scan, splash, maxtext-ce, seq8192, bs3, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3-8B jax v045 — save norm rsqrt stats on-device @ seq8192 bs3

## Hypothesis under test

**Hypothesis**: tagging the RMSNorm/QK-norm rsqrt reduction stat with `checkpoint_name("norm_rsqrt")` + a
`save_only_these_names("norm_rsqrt")` remat policy keeps the tiny `(B,T,1)`/`(B,T,H,1)` stats ON-DEVICE, so the
backward skips recomputing the f32 mean-square reductions (xprof: QK-norm reduce remat 685ms + RMSNorm
`add_rsqrt_fusion`), lifting the frontier above v035 (34.6% / 6,030). On-device → sidesteps the offload-pipeline
wall (v039).

**Mechanism**: image `qwen3-8b-jax:v045-save-norm-stats` (FROM v033-maxtext-ce, ONLY: tag rsqrt + env-gated
`save_only_these_names` policy). v035 flags + env `SAVE_NORM_STATS=1`. **Value-preserving** (saving an
intermediate never changes values → equivalence holds by construction).

**Predicted signal**: tok/s/chip > 6,030 beyond noise.

**Falsification criterion**: ≤ 6,030 within noise (XLA already fuses the reduction with the elementwise → save is a no-op).

## Setup

- Image `qwen3-8b-jax:v045-save-norm-stats`, `--use_remat --use_splash --use_scan --use_maxtext_ce --batch_size=3
  --seqlen=8192` + v035 LIBTPU bundle + `SAVE_NORM_STATS=1`. Startup: `[mesh] fsdp=8 tp=1`, `global_batch=24
  per_chip_batch=3`, `[scan]/[attn]/[remat]/[ce]` ON. (`[remat]` prints the cosmetic `nothing_saveable` label;
  the actual policy is selected by the `SAVE_NORM_STATS` env in the model — the consistent −26 ms confirms it took effect.)

## Results — marginal new frontier (+0.63%)

| run | norm-reduction remat | tok/s/chip | MFU | steady step (ms) |
|-----|----------------------|-----------|-----|------------------|
| v035 | recompute (nothing_saveable) | 6,030 | 34.6% | 4,075 |
| **v045** | **save rsqrt stat on-device** | **6,068** | **34.8%** | **4,049** |

Δ = **+0.63% tok/s/chip (+38), −26 ms (−0.64%), +0.2 pp MFU** — **consistent across all 18 steady-state steps
(~5× the ±5 ms noise band)**, so a real-if-small gain. Loss 12.0994 → 12.0508 monotone, no NaN, exit 0, no OOM
(stats are tiny f32 reductions — negligible HBM).

## Profile

- **Run name**: `2026-06-05-qwen3-jax-v045-rsqrt` · xprof [`http://localhost:8791/?run=2026-06-05-qwen3-jax-v045-rsqrt`](http://localhost:8791/?run=2026-06-05-qwen3-jax-v045-rsqrt) · GCS `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v045-rsqrt/plugins/profile/2026_06_05_06_08_55/` (2 hosts, steps 12–14). On-disk: [`raw/profiles/2026-06-05-qwen3-jax-v045-rsqrt/`](../../../../../raw/profiles/2026-06-05-qwen3-jax-v045-rsqrt/). The −26 ms is the skipped f32 mean-square reduction recompute in the backward (the saved `norm_rsqrt` stat replaces the recomputed `add_rsqrt_fusion`/QK-norm `reduce_sum`).

## HLO Dump

- **GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v045-rsqrt/hlo/` (29 files incl. `module_0109.jit_train_step.*`). Backward should show the saved `norm_rsqrt` value reused instead of a recomputed reduction.

## Verdict

**Supported (marginal new seq8192 frontier, 34.8% / 6,068).** Keeping the norm reduction statistic on-device
skips the f32 mean-square reduction recompute, for a small but consistent −26 ms (+0.63%). It's value-preserving
and — unlike host-offload (v039 wall) — works because the saved tensor is tiny and stays on-device. **Caveat**:
the gain is single-run and small (~5× noise); a confirm run would solidify it. This is the *only* lever from the
"try-all" gap-closing sweep that moved the frontier (tp refuted, RMSNorm-bf16 parity). **frontier v035 → v045
(34.8% / 6,068 = 87.4% of MaxText 6,942).**

## Next hypotheses

- [save norm rsqrt stats — confirm run](../../../../hypotheses/qwen3-jax-save-norm-stats.md) — re-run v045 to confirm the +0.63% is reproducible (small single-run win).
- None new on the recompute axis — the dominant SiLU double-recompute (3,985 ms) is walled (can't save on-device 21.7 GB; can't offload, v039). The residual to MaxText is its host-offload pipelining (a runtime/build capability).

## Sources

- `raw/profiles/2026-06-05-qwen3-jax-v045-rsqrt/` (gitignored; GCS path above)
