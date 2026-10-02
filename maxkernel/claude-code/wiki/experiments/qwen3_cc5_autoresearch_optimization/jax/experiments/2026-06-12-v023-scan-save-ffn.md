---
title: "Qwen3-8B jax v023 — scan + save_attn + saved FFN intermediate"
type: experiment
hypothesis: qwen3-jax-scan-over-layers
model: qwen3-cc5-jax
variant: "8B/v6e-8"
commit: fork v6e8-qwen3-8b-jax-20260612-v023-sffn
status: refuted
verdict: refuted
tags: [qwen3-cc5, jax, v6e-8, scan, remat, seq8192]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3-8B jax v023 — scan + save_attn + saved FFN intermediate

The scan-perf parity play: spend scan's 11.6 GiB headroom on uniformly
saving the down_proj input, removing the FFN recompute that makes scan
(1471 ms) slower than skip-every-5 (1389 ms).

## Hypothesis under test

**Hypothesis**: Saving `ffn_pre_down` (36 × 192 MB ≈ 6.9 GiB) under scan +
save_attn cuts the backward recompute enough that step time < 1419 ms
(≥60% of the 82 ms scan↔skip5 gap recovered); stretch: ≤ 1389 ms ties v019
and makes scan strictly best (same perf + 3.4× compile + 1.1 GiB headroom).

**Mechanism**: `checkpoint_name(silu(gate)·up, "ffn_pre_down")` in
`Qwen3MLP` + `JAX_REMAT_SAVE_FFN=1` extending the save_attn policy. CPU
smoke: scan + new policy backward OK, loss parity exact.

**Predicted signal**: `loop fusion` share drops vs v020's profile; HBM ≈
19.63 + 6.9 ≈ 26.5 GiB (< 30.5 bar); step ∈ [1389, 1419] ms.

**Falsification criterion**: ph2 ≥ ph1 − 1% (the saved tensor doesn't reduce
recompute time meaningfully — i.e. it was latency-hidden), or OOM → refuted;
scan remains a special-purpose tool.

## Setup

- Hardware: v6e-8, fsdp=8, tp=1, 1 slice of `<your-cluster>`.
- Image: `qwen3-8b-jax:v6e8-qwen3-8b-jax-20260612-v023-sffn`.
- Workload: `alekseyv-qwen3-cc5-jax-v023-sffn`.
- Phases (both: bs1/seq8192, splash + chunked CE f32-x + scan + save_attn):
  ph1 = SAVE_FFN off (v020 reference, 12 steps); ph2 = SAVE_FFN=1 (20 steps,
  profiled).

## Results

Both phases exit 0; loss parity exact through step 11; no OOM.

| Metric | Ph1 (scan + save_attn) | Ph2 (+ SAVE_FFN) | Δ |
|--------|------------------------|-------------------|---|
| Steady median | 1470.8 ms | 1467.3 ms | **−0.24% (noise)** |
| Throughput | 44,557 tok/s | 44,614 tok/s | +0.1% |
| MFU | 31.9% | 32.0% | — |

PASS bar (<1419 ms) not approached.

## Profile

- **xprof URL**: `http://localhost:8791/?run=2026-06-12-qwen3-jax-v023-sffn`
  (run `…/2026_06_12_05_46_26`); profiled steps 12–14.
- **GCS run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v023-sffn/plugins/profile/2026_06_12_05_46_26/`
- The flat A/B is the complete signal.

## HLO Dump

- **GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v023-sffn/hlo/`

## Verdict

**refuted** — saving the FFN intermediate is worth ~0.2%: the FFN recompute
under scan is effectively latency-hidden, so the 82 ms scan↔skip5 gap lives
elsewhere (scan loop mechanics / attention-adjacent recompute). Scan stays a
special-purpose tool (compile 3.4×, −8 GiB); **v019 (skip5, no-scan) remains
the seq-8192 frontier recipe at 33.6% MFU**. With this, every direction from
the 2026-06-12 retrospective is resolved; the lane moves to
maintenance/holding pending new ideas, ingests, or user direction.

## Observations

- Remat recompute under scan is largely overlapped — trading HBM for saved
  intermediates has near-zero step-time value here (corroborates v018's
  memory-pressure lesson from the opposite direction).

## Next hypotheses

- None — retrospective queue fully resolved (scan landed as tool; bs2@8192,
  splash knobs, FFN-save refuted; host-offload mooted; QK-norm+RoPE parked;
  data-formatting closed). The next candidates should come from new ingests
  (e.g. a tokamax upgrade re-opening mosaic-CE bf16) or a future
  retrospective at a changed op-point (hardware, model size, or real-data
  training runs).

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v023-sffn/`
- Prior: [v020 (scan)](2026-06-12-v020-scan.md), [v019 (frontier)](2026-06-12-v019-val50.md), [v021 (bs2-scan refuted)](2026-06-12-v021-bs2-s8k-scan.md)
