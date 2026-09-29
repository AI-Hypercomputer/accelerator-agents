---
title: "Qwen3-8B jax v032 — scan + overlap @ seq2048 bs4 (lift the v018 frontier)"
type: experiment
hypothesis: qwen3-jax-scan-layers
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-scanfull-2026-06-02
status: refuted
verdict: refuted
tags: [qwen3-cc, jax, scan, overlap, splash, remat, seq2048, bs4, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v032 — scan + overlap @ seq2048 bs4

## Hypothesis under test

**Hypothesis**: scan + the overlap flags lifted the seq8192 frontier (v028, +6.2%); applying the same to
the **seq2048** frontier (v018, 35.8% / 6,964 tok/s/chip) should similarly improve it toward MaxText's
seq2048 (38.0% / 7,116) — a quick win on the shape where we're already near parity (97.9%).

**Mechanism**: image `qwen3-8b-jax:v030-scan-full`. v018 frontier shape + scan + overlap:
`--use_remat --use_splash --use_scan --batch_size=4 --seqlen=2048` (no offload, no CE — plain CE fits at
seq2048 bs4 like v018) + 7 scheduler + 4 overlap flags.

**Predicted signal**: MFU/tok-s-chip > v018 35.8% / 6,964, toward MaxText 7,116; loss parity; exit 0.

**Falsification criterion**: ≤ v018 6,964 beyond noise (scan/overlap don't help at seq2048 — its
collectives are already a smaller share at the shorter seq, so less to overlap).

## Setup

- Image `qwen3-8b-jax:v030-scan-full`; `--use_remat --use_splash --use_scan --batch_size=4 --seqlen=2048` (NO offload, NO CE — plain CE fits at seq2048 bs4 like v018) + 7 scheduler + 4 overlap flags. scan/splash/remat all confirmed active in log (`[scan] lax.scan over stacked decoder layer (1 compiled body) ON`; splash bq=2048 bkv=1024 fused_bwd=True; remat nothing_saveable). global batch 32.

## Results — BELOW frontier

| run | config | tok/s/chip | MFU | step (ms) |
|-----|--------|-----------|-----|-----------|
| **v018** | 7-flag sched, **no scan** | **6,964** | **35.8%** | — |
| **v032** | **scan + overlap**, bs4 | **6,731** | 34.6% | 1,217 |
| MaxText | seq2048 bs4 ref | 7,116 | 38.0% | — |

v032 = **−3.4% tok/s/chip vs v018** (−233/chip, −1.2 pp MFU). Loss 12.098→12.069 monotone over 20 steps, no NaN/Inf, exit 0. Compile 39.3 s.

## Profile

- **Run name**: `2026-06-02-qwen3-jax-v032-scan-overlap-s2k-bs4` · xprof [`http://localhost:8791/?run=2026-06-02-qwen3-jax-v032-scan-overlap-s2k-bs4`](http://localhost:8791/?run=2026-06-02-qwen3-jax-v032-scan-overlap-s2k-bs4) · GCS `.../plugins/profile/2026_06_02_16_43_42/` (steps captured mid-run, steps 2–19 steady). On-disk: [`raw/profiles/2026-06-02-qwen3-jax-v032-scan-overlap-s2k-bs4/`](../../../../../raw/profiles/2026-06-02-qwen3-jax-v032-scan-overlap-s2k-bs4/).

## HLO Dump

- **GCS**: `.../2026-06-02-qwen3-jax-v032-scan-overlap-s2k-bs4/hlo/` (31 objects; scanned while-body train-step).

## Verdict

**Refuted** (≤ v018 6,964 beyond noise). Scan + the overlap flags — which lifted the **seq8192** frontier +6.2% (v028) — **regress at seq2048** (−3.4%). The falsification criterion predicted this: at the shorter sequence the FSDP collectives are already a smaller share of step time (less compute to hide them behind, and they overlap less of a shorter matmul tail), so the overlap flags buy little; meanwhile the `lax.scan` while-body adds loop-carry/boundary overhead that, without a large collective tail to amortise it against, is net negative. **seq2048 frontier stays v018 (35.8% / 6,964, no scan).** MaxText seq2048 (38.0% / 7,116) remains the reference; the residual ~2.2 pp is again its better-pipelined per-step kernel efficiency, not a config lever we hold.

**Conclusion for the lane**: scan+overlap is a **seq8192-specific** win (long-sequence collective tail is large enough to overlap-amortise the scan body), NOT a universal lever. The Knobs matrix records it as `seq8192: win / seq2048: regress`.

## Next hypotheses

- None — scan+overlap refuted at seq2048; v018 remains the frontier. seq8192 frontier is v028 (documented hard wall on batch). This closes the scan+overlap arc. See the [closing analysis](../../../../analyses/2026-06-02-qwen3-cc-jax-maxtext-closing.md).
