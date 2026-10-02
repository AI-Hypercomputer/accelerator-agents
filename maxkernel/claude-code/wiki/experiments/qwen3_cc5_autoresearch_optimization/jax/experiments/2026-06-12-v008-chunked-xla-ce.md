---
title: "Qwen3-8B jax v008 — chunked_xla CE, bf16 inputs (+ splash)"
type: experiment
hypothesis: qwen3-jax-chunked-xla-ce-bf16
model: qwen3-cc5-jax
variant: "8B/v6e-8"
commit: fork v6e8-qwen3-8b-jax-20260612-v006-tkmce-bf16 (same image; flag-only diff)
status: invalid
verdict: invalid
tags: [qwen3-cc5, jax, v6e-8, tokamax, chunked-xla, cross-entropy]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3-8B jax v008 — chunked_xla CE, bf16 inputs (+ splash)

Tests [qwen3-jax-chunked-xla-ce-bf16](../../../../hypotheses/qwen3-jax-chunked-xla-ce-bf16.md)
— the pivot from [v006](2026-06-12-v006-tkmce-bf16.md) (mosaic_tpu bf16-Ref
crash) — on the v006 image with `--tokamax_ce_impl=chunked_xla`. No rebuild.

## Hypothesis under test

**Hypothesis**: chunked_xla CE with bf16 inputs computes the loss in
chunk-sized vocab transients (no monolithic `[BL,V]` buffer), with loss parity
vs plain CE and step time ≤ the 453 ms splash frontier — establishing the
streamed-CE building block that bs2/seq-8192 need.

**Mechanism**: `tokamax.linear_softmax_cross_entropy_loss(implementation=
"chunked_xla")` — pure-XLA chunked logsumexp; bf16-safe (no Pallas dtype
refs). llama3-lane precedent: chunked_xla beat mosaic_tpu (+1.6%) and
bf16-native added +0.3% (its exps 62b/66/66b).

**Predicted signal**: no `[2048,151936]`-sized buffer in the train_step buffer
assignment (chunk-sized instead); no `f32[4096,151936]` cast; program total
< the plain-CE 29.04 GiB at bs1. Step time within noise of 453 ms or better.

**Falsification criterion**: loss diverges from phase-1 reference beyond bf16
noise → invalid; step time > 453 ms + noise or program HBM ≥ plain CE →
refuted.

## Setup

- Hardware: v6e-8, fsdp=8, tp=1, 1 slice of `<your-cluster>`.
- **Attempt 1** (`alekseyv-qwen3-cc5-jax-v008-cxla`): crashed pre-compile —
  `ValueError: Unsupported implementation: chunked_xla`. The v006 image ships
  tokamax **0.0.10** (pip), which only has mosaic_tpu/xla; `chunked_xla`
  exists in the `raw/code/tokamax` dev checkout. Phase 1 reference ran clean
  (453 ms / 36.2k tok/s — third independent reproduction of the splash bs1
  number).
- **Attempt b** image: `qwen3-8b-jax:v6e8-qwen3-8b-jax-20260612-v008b-cxla` —
  v006 image + dev tokamax layered in (`pip install --no-deps` from the
  checkout). CPU smoke verified `chunked_xla` fwd+bwd with bf16 inputs works.
- Workload: `alekseyv-qwen3-cc5-jax-v008-cxla-b`.
- In-workload A/B (deterministic data, seed 0), both phases splash-on:
  ```bash
  python -u train.py --batch_size=1 --seqlen=2048 --train_steps=8 \
      --use_splash=True --use_tokamax_ce=False   # phase 1: reference
  python -u train.py --batch_size=1 --seqlen=2048 --train_steps=20 \
      --use_splash=True --use_tokamax_ce=True --tokamax_ce_impl=chunked_xla \
      --profile_dir=<gcs_root> --profile_start_step=12 --profile_steps=3
  ```

## Results

Attempt b: both phases exit 0; `[ce] … impl=chunked_xla (bf16 inputs)`
confirmed; 20 steps clean.

| Metric | Phase 1 (plain CE) | Phase 2 (chunked_xla bf16) | Δ |
|--------|--------------------|----------------------------|---|
| Steady step time | 452–453 ms | 467–469 ms | **+3.3%** |
| Throughput | 36,170 tok/s | 35,024 tok/s | −3.2% |
| Program HBM | 23.03 GiB (v002 module) | **23.92 GiB** | **+0.9 GiB** |
| Loss (steps 2–7, seed 0) | 12.089–12.103 (fp32-smooth) | **11.25–11.44, quantized to 1/16** | **−0.73 systematic offset** |

**Numerics analysis**: `chunked_xla.py` sets `dtype = x.dtype` — with bf16
inputs the per-chunk `logsumexp` accumulates in **bf16**. The observed −0.73 ≈
ln 2 offset and the 1/16-quantized loss values are the bf16-lse failure mode
in mild form (no NaN/collapse, but the loss — and hence gradients — are not
the reference computation).

**Memory analysis**: the `[BL,V]` logits buffers ARE gone (0 occurrences),
but the kernel's backward carries a **f32 `dw` accumulator `[H,V]` = 2.49 GB**
(`chunked_xla.py:267`), which more than offsets the bs1 logits saving. The
chunked-CE memory win only materializes where logits scale with L — i.e. at
seq 8192 (~10 GiB logits vs the fixed 2.49 GB accumulator).

## Profile

- **xprof URL**: `http://localhost:8791/?run=2026-06-12-qwen3-jax-v008b-cxla`
  (run `…/2026_06_12_02_37_07`); profiled steps 12–14.
- **GCS run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v008b-cxla/plugins/profile/2026_06_12_02_37_07/`
- Step-time + HLO + loss-series evidence sufficient for the verdict; deep
  bucket attribution deferred to the corrected v011.

## HLO Dump

- **GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v008b-cxla/hlo/`
- `Total bytes used: 23.92 GiB`; 0 × `[2048,151936]` buffers (streaming
  confirmed); f32 `dw` accumulator present per source analysis.

## Verdict

**invalid** — loss diverges from the fp32 reference by a systematic −0.73
(≈ ln 2; bf16 lse accumulation per `dtype = x.dtype`), violating the
falsification criterion's parity bound. Additionally step time regressed
+3.3% and HBM did not improve at this shape. The corrective derivative is
precise and cheap: **cast x (h) only to f32** — 32 MB — so `dtype` (and the
lse) become f32 while w stays bf16 (avoiding v003's 2.49 GB w-cast).
Dispatched as v011.

## Observations

- tokamax `chunked_xla` accumulation dtype follows `x.dtype`; bf16-x ⇒ bf16
  lse ⇒ ~ln 2 loss bias. f32-x (cheap) is the correct usage; f32-w is never
  needed.
- chunked_xla bwd allocates a fixed f32 `[H,V]` dw accumulator (2.49 GB at
  Qwen3-8B) — the memory case for chunked CE is seq-scaling (8192), not
  batch-scaling.

## Next hypotheses

- [qwen3-jax-chunked-xla-ce-f32x](../../../../hypotheses/qwen3-jax-chunked-xla-ce-f32x.md) — chunked_xla with f32-x-only cast (v011): fixes the lse dtype at +32 MB; parity A/B decides.
- [qwen3-jax-seq-8192](../../../../hypotheses/qwen3-jax-seq-8192.md) — the shape where chunked CE's memory math actually pays (~10 GiB logits saved vs 2.49 GB accumulator); follows v011 if parity holds.

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v008-cxla/`
- Prior: [v006 (invalid)](2026-06-12-v006-tkmce-bf16.md), [v003 (refuted)](2026-06-12-v003-tkmce.md), [v002 (frontier)](2026-06-12-v002-splash.md)
