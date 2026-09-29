---
title: "Qwen3 JAX v236 - RS3+AR2 SparseCore latency rerun"
type: experiment
hypothesis: "v235 landed within noise of the confirmed v230 RS2+AR2 frontier while matching or slightly exceeding late/no-trace windows; an exact rerun determines whether RS3+AR2 is a reproducible frontier alternative or just noise."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs3-ar2-rerun
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, latency-model, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8, rerun]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v236 - RS3+AR2 SparseCore latency rerun

## Hypothesis under test

**Hypothesis**: v235's RS3+AR2 schedule is close enough to v230 that a second
sample is needed. If RS3+AR2 is a real alternative frontier, the rerun should
repeat or exceed v230's **59,749 tok/s** full-window average without losing the
late/no-trace profile band.

**Mechanism**: Exact rerun of v235. Keep the v230 SparseCore RS/AR split and
AR latency multiplier `2`, with
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3`.

**Predicted signal**: Support requires clean compile/loss and throughput above
v230's confirmed **59,749 tok/s** band, or repeated lower/equal profile step
time with no HBM/MXU regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed
HLO with worse profile, or clean completion at or below the v230/v235 band.

## Setup

GKE workload `alekseyv-q3-v236-rs3-ar2-r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v236-no-scan-sparsecore-rs-ar-rs3-ar2-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an exact runtime flag rerun and preserves
model math.

## Result

Completed cleanly as workload `alekseyv-q3-v236-rs3-ar2-r` with `EXIT_CODE=0`
on both workers and normal loss ending at 12.0459.

- Worker0 pod: `alekseyv-q3-v236-rs3-ar2-r-slice-job-0-0-vh2jp` averaged
  **59,750 tok/s** over steps 2-19, **59,795 tok/s** over steps 15-19, and
  **59,765 tok/s** excluding profiled steps. Best post-warmup step:
  **59,907 tok/s**.
- Worker1 pod: `alekseyv-q3-v236-rs3-ar2-r-slice-job-0-1-2mvds` averaged
  **59,624 tok/s** over steps 2-19, **59,456 tok/s** over steps 15-19, and
  **59,650 tok/s** excluding profiled steps. Best post-warmup step:
  **59,922 tok/s**. This worker had runtime dips at profiled step 12 and
  late step 16; worker0 is the clean comparable series.

Worker0 is a noise-level **+1 tok/s** over v230's best-worker full-window
average (**59,749 tok/s**) and **+3 tok/s** over v235's best-worker average.
Treat this as a confirmed frontier-band tie, not a material throughput gain.

## Profile

XProf run:
`2026-06-05-qwen3-jax-v236-no-scan-sparsecore-rs-ar-rs3-ar2-rerun/2026_06_05_08_10_30`.

- Step time: **4401.1 ms** average.
- MXU utilization: **67.0%**.
- Peak HBM: **30.93 GiB**.
- Top train-step buckets include `convolution fusion`
  **56,763.3 ms / 53.7%**, `custom-call` **29,411.4 ms / 27.8%**, and
  `loop fusion` **10,683.3 ms / 10.1%**.

## HLO Dump

Optimized train-step HLO is byte-identical to v235 and remains the RS3+AR2
schedule:

- SHA256:
  `d4d46797cc4508667e0aad8b0a75bf6485f046806db6ced027a159f89ee80b5c`
- Size: **19,909,611 bytes**
- Lines: **119,349**

## Verdict

Supported as a frontier-band tie. RS3+AR2 reproduced the same optimized HLO as
v235 and matched the v230/v235 throughput/profile band, with the clean worker
landing at **59,750 tok/s / 42.9% MFU**. The gain over v230 is only **1 tok/s**
and XProf MXU is slightly lower, so carry it as an equivalent confirmed
schedule rather than a meaningful new MFU step.
