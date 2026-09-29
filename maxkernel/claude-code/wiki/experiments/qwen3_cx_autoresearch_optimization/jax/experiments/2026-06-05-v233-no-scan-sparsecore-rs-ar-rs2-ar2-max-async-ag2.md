---
title: "Qwen3 JAX v233 - RS2+AR2 with max concurrent async all-gathers 2"
type: experiment
hypothesis: "The confirmed frontier keeps all-gather on async collective fusion; a less restrictive async all-gather concurrency cap of 2 may reduce scheduling pressure without serializing all-gather as aggressively as v232's cap of 1."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs2-ar2-maxag2
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, all-gather, async-collective-fusion, concurrency, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v233 - RS2+AR2 with max concurrent async all-gathers 2

## Hypothesis under test

**Hypothesis**: v232 showed that forcing a single concurrent async all-gather is
too restrictive or inert on the v230 RS2+AR2 frontier. TPU recipe stacks also
use a cap of `2` in some long-sequence shapes. Setting
`--xla_max_concurrent_async_all_gathers=2` may reduce residual all-gather
scheduling pressure while preserving more overlap than v232.

**Mechanism**: Keep v230 fixed and add only
`--xla_max_concurrent_async_all_gathers=2`.

**Predicted signal**: Support requires clean compile/loss and throughput above
v230's **59,749 tok/s** confirmed value, or a lower profile step time/all-gather
bucket with the same throughput band and no HBM regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO/memory
regression, or clean completion at or below v230's confirmed band.

## Setup

GKE workload `alekseyv-q3-v233-rs2-ar2-ag2`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v233-no-scan-sparsecore-rs-ar-rs2-ar2-max-async-ag2`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an XLA runtime scheduling flag isolate and
preserves model math.

## Result

Completed cleanly as workload `alekseyv-q3-v233-rs2-ar2-ag2` with
`EXIT_CODE=0` on both workers and normal loss ending at 12.0459.

- Worker0 pod: `alekseyv-q3-v233-rs2-ar2-ag2-slice-job-0-0-l6ckn`
  averaged **58,282 tok/s** over steps 2-19, **58,334 tok/s** over steps
  15-19, and **58,323 tok/s** excluding profiled steps. Best post-warmup step:
  **58,437 tok/s**.
- Worker1 pod: `alekseyv-q3-v233-rs2-ar2-ag2-slice-job-0-1-wnz4v`
  averaged **58,311 tok/s** over steps 2-19, **58,335 tok/s** over steps
  15-19, and **58,322 tok/s** excluding profiled steps. Best post-warmup step:
  **58,436 tok/s**.

This is a clear regression from v230's **59,749 tok/s** full-window frontier.

## Profile

XProf run:
`2026-06-05-qwen3-jax-v233-no-scan-sparsecore-rs-ar-rs2-ar2-max-async-ag2/2026_06_05_07_36_21`.

- Step time: **4508.1 ms** average.
- MXU utilization: **65.7%**.
- Peak HBM: **30.90 GiB**.
- Top train-step buckets include `convolution fusion`
  **56,409.5 ms / 52.1%** and exposed direct `all-gather`
  **5,067.2 ms / 4.7%**.

## HLO Dump

Optimized train-step HLO changed from v229/v230/v232:

- SHA256:
  `94c2737b7e9dfa48715821d41919490b32113f315107bdc924d8dcbf714e01ac`
- Size: **17,784,311 bytes**
- Lines: **103,755**

## Verdict

Refuted. `--xla_max_concurrent_async_all_gathers=2` changes the compiler plan,
exposes a large direct all-gather bucket, lowers MXU, and regresses throughput
by about **2.4%** versus v230. Do not carry async all-gather concurrency caps.
