---
title: "Qwen3-8B jax v030 — scan + offload + CE(f32) + bs3 @ seq8192"
type: experiment
hypothesis: qwen3-jax-host-offload-bs3-seq8192
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-scanfull-2026-06-02
status: inconclusive
verdict: inconclusive
tags: [qwen3-cc, jax, scan, offload, tokamax-ce, overlap, seq8192, bs3, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v030 — scan + offload + CE(f32) + bs3 @ seq8192

## Hypothesis under test

**Hypothesis**: v026 (offload + CE + bs3, NO scan) OOM'd by only **+2.34G**. Adding **scan** — which
compiles 1 layer body instead of 36 unrolled, shrinking the concurrent activation/program footprint — frees
enough HBM to **fit bs3**, and with scan+overlap making collectives cheap (v028), bs3 should amortize toward
MaxText (45.3% / 6,942 tok/s/chip). CE weight stays **f32** (mosaic kernel requires it — v029).

**Mechanism**: image `qwen3-8b-jax:v030-scan-full`. `--use_remat --use_splash --use_scan --offload_remat
--use_tokamax_ce --tokamax_ce_impl=mosaic_tpu --batch_size=3 --seqlen=8192` + 7 scheduler + 4 overlap flags.

**Predicted signal**: bs3 fits (scan frees > +2.34G); loss ~12.07 stable; MFU/tok-s-chip > v028 bs1 (5,632),
toward 6,942.

**Falsification criterion**: still OOM (scan didn't free enough → need vocab-sharded CE or bs2 fallback);
MFU ≤ 5,632 (bs3 doesn't amortize even with scan+overlap → v028 bs1 is the seq8192 frontier).

## Setup

- Image `qwen3-8b-jax:v030-scan-full`; `--use_remat --use_splash --use_scan --offload_remat --use_tokamax_ce --tokamax_ce_impl=mosaic_tpu --batch_size=3 --seqlen=8192` + 7 sched + 4 overlap flags. All active; bs3 global batch 24.

## Results — bs3 FITS (milestone) but throughput REGRESSES

| Metric | v028 (scan+overlap, bs1) | **v030 (full stack, bs3)** | MaxText bs3 |
|--------|---------------------------|----------------------------|-------------|
| fits | yes | **yes (scan closed v026's +2.34G)** | yes |
| MFU (918 basis) | 32.3% | **26.4%** | 45.3% |
| tok/s/chip | 5,632 | **4,595** | 6,942 |
| steady step (ms) | 1,452 | 5,347 (bs3) | 3,546 |
| loss | stable | 12.105→12.057 monotone, no NaN | — |
| exit | 0 | 0 | — |

bs3 is **−18% tok/s/chip vs v028 bs1** — it does NOT amortize. (All MFU on the trainer's 918-TFLOPS basis,
same as v028/MaxText; the runner's alt "49% @ bf16-459-peak" is just a different denominator, not comparable.)

## Profile

- **Run name**: `2026-06-02-qwen3-jax-v030-scan-offload-ce-s8k-bs3` · GCS `.../plugins/profile/2026_06_02_16_32_07/` (2 hosts; steps 14–16). HLO `.../hlo/` (`module_0109.jit_train_step`). On-disk: [`raw/profiles/2026-06-02-qwen3-jax-v030-scan-offload-ce-s8k-bs3/`](../../../../../raw/profiles/2026-06-02-qwen3-jax-v030-scan-offload-ce-s8k-bs3/). Profile-analyzer dispatched to attribute the bs3 drag (offload host-transfer vs CE f32-weight gather vs splash bwd).

## HLO Dump

- **GCS**: `.../2026-06-02-qwen3-jax-v030-scan-offload-ce-s8k-bs3/hlo/` (scanned while-body train-step).

## Verdict

**Inconclusive** — a genuine **bs3-fit milestone** (scan's 1-body program closed the +2.34G that OOM'd v026
without scan; CE numerically fine, loss monotone) but a **throughput regression** (4,595 < v028 bs1 5,632).
**bs3 does not amortize on our stack** — consistent with v016 (bs2<bs1); even with scan + overlap, the
named-offload host round-trip (3× data at bs3) + the f32 CE-weight all-gather cost more than the batch
occupancy buys. seq8192 frontier **stays v028 (32.3% / 5,632, bs1)**. The MaxText bs3 advantage (6,942) is
its *better-pipelined* offload (profile: <0.1% host-copy) + its own CE — not something the tokamax-CE +
JAX-offload path reproduces at bs3.

## Next hypotheses

- **Diagnose** (profile-analyzer on v030): is the bs3 drag the named-offload host-transfer (not overlapped) or the f32 CE-weight gather? Determines whether a lighter/better-pipelined offload could make bs3 amortize.
- **bs2 + scan + overlap, NO offload** (v016 fit bs2 w/o offload) — isolates batch amortization from the offload overhead: does bs2 beat v028 bs1 (5,632)? If yes, batch helps and the offload is the drag; if no, batch genuinely doesn't amortize and v028 bs1 is the seq8192 frontier.
- **Retry scan+overlap at seq2048** (v018 35.8% frontier) — scan should lift it too, toward MaxText's 38.0%; likely a quick win independent of the bs3 fight.
