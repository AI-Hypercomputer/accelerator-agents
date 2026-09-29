---
title: "Qwen3-8B jax v031 — scan + overlap + CE + bs2 (no offload) @ seq8192"
type: experiment
hypothesis: qwen3-jax-scan-layers
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-scanfull-2026-06-02
status: refuted
verdict: refuted
tags: [qwen3-cc, jax, scan, overlap, tokamax-ce, splash, remat, seq8192, bs2, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v031 — scan + overlap + CE + bs2 (no offload) @ seq8192

## Hypothesis under test

**Hypothesis**: v030 showed bs3 regresses (offload host-transfer drag). Isolate batch amortization from the
offload: bs2 fits at seq8192 with streamed CE and **no offload** (v016 proved bs2+CE fits). With scan +
overlap (which v028 showed help at bs1), does **bs2 beat v028 bs1 (5,632 tok/s/chip)**? If yes, batch
amortizes and the v030 drag is the offload (→ pursue lighter offload for bs3); if no, batch genuinely
doesn't amortize → v028 bs1 is the seq8192 ceiling for our stack.

**Mechanism**: image `qwen3-8b-jax:v030-scan-full`. `--use_remat --use_splash --use_scan --use_tokamax_ce
--tokamax_ce_impl=mosaic_tpu --batch_size=2 --seqlen=8192` (NO `--offload_remat`) + 7 sched + 4 overlap.

**Predicted signal**: fits (v016 bs2+CE fit w/o offload); MFU/tok-s-chip vs v028 5,632 and v016 5,139.

**Falsification criterion**: ≤ v028 5,632 → batch doesn't amortize even clean (v028 bs1 is the frontier);
OOM → bs2 needs offload after all.

## Setup

- Image `qwen3-8b-jax:v030-scan-full`; `--use_remat --use_splash --use_scan --use_tokamax_ce --tokamax_ce_impl=mosaic_tpu --batch_size=2 --seqlen=8192` (NO offload) + 7 sched + 4 overlap. scan/splash/CE active, offload absent — confirmed.

## Results

| run | config | tok/s/chip | MFU | step (ms) |
|-----|--------|-----------|-----|-----------|
| v028 | scan+overlap, **bs1** | **5,632** | 32.3% | 1,452 |
| **v031** | scan+overlap+CE, **bs2**, no offload | **5,553** | 31.9% | 2,950 |
| v016 | CE bs2, no scan/overlap | 5,139 | 29.5% | — |
| v030 | scan+offload+CE, bs3 | 4,595 | 26.4% | 5,347 |

bs2 = **−1.4% vs v028 bs1**; +8% vs v016 (scan+overlap help, but don't make bs2 amortize). Loss 12.106→12.064 monotone, no NaN, exit 0.

## Profile

- **Run name**: `2026-06-02-qwen3-jax-v031-scan-ce-bs2-nooffload-s8k` · GCS `.../plugins/profile/2026_06_02_16_41_56/` (steps 14–16). HLO `.../hlo/`. On-disk: [`raw/profiles/2026-06-02-qwen3-jax-v031-scan-ce-bs2-nooffload-s8k/`](../../../../../raw/profiles/2026-06-02-qwen3-jax-v031-scan-ce-bs2-nooffload-s8k/).

## HLO Dump

- **GCS**: `.../2026-06-02-qwen3-jax-v031-scan-ce-bs2-nooffload-s8k/hlo/`.

## Verdict

**Refuted** (bs2 ≤ v028 bs1). Decisive: with the offload removed, bs2 (5,553) is still **below** v028 bs1
(5,632) — so the v030 bs3 regression was **not** just the offload; **batch genuinely anti-amortizes at
seq8192 on our stack**. The monotonic series **bs1 5,632 > bs2 5,553 > bs3 4,595** confirms it (v016
bs2<bs1 was not a CE artifact). **v028 (scan+overlap, bs1, 32.3% / 5,632) is the achieved seq8192
frontier.** The residual gap to MaxText (45.3% / 6,942, bs3) is MaxText's **bs3 per-token efficiency** —
its better-pipelined offload (<0.1% host-copy) + its own fused CE let bs3 amortize where our
tokamax-CE + JAX-offload path does not. That's a kernel-pipeline-level difference, not a config lever —
**documented hard wall** for the seq8192 batch dimension.

## Next hypotheses

- None on seq8192 batch — bs1 is the frontier; bs2/bs3 refuted; the MaxText gap is structural (its bs3 offload/CE pipelining).
- seq2048 scan (v032, in flight) — the remaining open lever; if it lifts v018, that's the last gain.
- Closing analysis once v032 lands: scan+overlap delivered seq8192 30.4%→32.3%; document the MaxText residual.
