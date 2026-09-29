---
title: "Qwen3 JAX v190 - no MLP intermediate sac, no scan, Splash BKV_DKV1024"
type: experiment
hypothesis: "The no-scan unrolled graph may interact differently with backward DKV Splash tiling, so a smaller DKV KV tile could reduce attention backward pressure despite regressing in scan-family runs."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-no-scan-bkvdkv1024
status: refuted
tags: [qwen3-cc, jax, flax-nnx, scan, no-scan, remat, activation-sharding, mlp, sharding-constraint, splash, splash-tiling, tokamax-splash, collective-matmul, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v190 - no MLP intermediate sac, no scan, Splash BKV_DKV1024

## Hypothesis under test

**Hypothesis**: v157 showed `SPLASH_BKV_DKV=1024` regressed in the scan family,
but the confirmed no-scan frontier is a much larger unrolled program with a
different fusion/copy/custom-call mix. Smaller DKV KV tiles may reduce backward
attention pressure enough to improve no-scan steady step time.

**Mechanism**: Runtime-only backward Splash DKV tiling change on top of the
confirmed v183 frontier:
`SPLASH_BKV_DKV=1024 SPLASH_BKV_DKV_COMPUTE=1024`.

**Predicted signal**: Support requires clean compile/loss and steady throughput
above the confirmed no-scan band: v182/v183 at 58,288-58,330 tok/s confirmed
average and 58,443 best observed. A fresh HLO/hash with lower profile step time
would justify carrying smaller backward DKV tiles only in the no-scan schedule.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the no-scan frontier band.

## Setup

GKE workload `alekseyv-qwen3-v190-noscan-bkvdkv1024`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v190-no-mlp-sac-no-scan-bkvdkv1024`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 SPLASH_BKV_DKV=1024 SPLASH_BKV_DKV_COMPUTE=1024 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit was made. This is a runtime attention-kernel tiling change
only and preserves model math. The current v169 lane equivalence test was rerun
in the dependency image on CPU and passed: logits max delta `2.682e-07`, loss
delta `0`, all 25 gradients OK, max gradient delta `7.823e-08`.

## Results

Refuted. Workload `alekseyv-qwen3-v190-noscan-bkvdkv1024` completed cleanly
with valid loss ending at 12.0460 and `EXIT_CODE=0`, but regressed below the
confirmed no-scan frontier:

- Worker0 stdout: 55,489 tok/s average / 6,936 tok/s/chip / 39.8% MFU.
- Best observed worker0 steady step: 55,648 tok/s at step 2.
- Profile-derived host averages: `gke-tpu-964065d9-9vh8` 4.739211 s / 55,314
  tok/s, `gke-tpu-964065d9-c180` 4.730082 s / 55,421 tok/s.
- Final loss: 12.0460.

## Profile

Profile files:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v190-no-mlp-sac-no-scan-bkvdkv1024/plugins/profile/2026_06_04_22_42_05/`

The profile validates the stdout regression. Device-side
`jit_train_step(14633774807237963760)` spans are about 4.730-4.739 s across the
two hosts, far slower than the v183 no-scan 4.497 s class.

## HLO Dump

Fresh train-step HLO:

- Optimized HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v190-no-mlp-sac-no-scan-bkvdkv1024/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 19,050,551 bytes.
- SHA256: `362c1236acda6c60a5cea43826813a9c8b6545f641d00632d9ac75ee87935ace`.
- Memory report: 27.66 GiB total bytes used.
- Text counts: all-gather 7,039, all-reduce 1,148, collective-permute 6,
  convolution 3,071, custom-call 2,809, copy 21,331, fusion 31,686.

Compared with the v182/v183 frontier HLO (`1ba7015c...`, 19,043,435 bytes),
the smaller DKV tile selects a real changed program. The changed program has
much higher text-level communication/fusion/copy/custom-call counts and slower
wall time.

## Verdict

Refuted. Do not carry `SPLASH_BKV_DKV=1024 SPLASH_BKV_DKV_COMPUTE=1024` in the
no-scan frontier. Both smaller forward KV tiles (v189) and smaller DKV KV tiles
(v190) slow the unrolled no-scan schedule.
