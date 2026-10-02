---
title: "Qwen3 JAX v189 - no MLP intermediate sac, no scan, Splash BKV512/BKV_COMPUTE512"
type: experiment
hypothesis: "Correctly pairing smaller forward Splash KV tiles with matching compute tiles may reduce no-scan attention custom-call pressure enough to improve steady throughput."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-no-scan-bkv512-compute512
status: refuted
tags: [qwen3-cc, jax, flax-nnx, scan, no-scan, remat, activation-sharding, mlp, sharding-constraint, splash, splash-tiling, tokamax-splash, collective-matmul, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v189 - no MLP intermediate sac, no scan, Splash BKV512/BKV_COMPUTE512

## Hypothesis under test

**Hypothesis**: v188 was an invalid launch because `SPLASH_BKV=512` was set
without the matching `SPLASH_BKV_COMPUTE=512`. The corrected pair may still
change the no-scan unrolled attention schedule in a useful way: smaller forward
KV tiles could reduce custom-call pressure enough to beat the confirmed no-scan
frontier, even though the same idea regressed in the scan-family v156 probe.

**Mechanism**: Runtime-only Splash tiling change on top of the confirmed v183
frontier:
`SPLASH_BKV=512 SPLASH_BKV_COMPUTE=512`.

**Predicted signal**: Support requires clean compile/loss and steady throughput
above the confirmed no-scan band: v182/v183 at 58,288-58,330 tok/s confirmed
average and 58,443 best observed. A fresh HLO/hash with lower wall time would
justify carrying the smaller forward KV tile only in the no-scan schedule.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the no-scan frontier band.

## Setup

GKE workload `alekseyv-qwen3-v189-noscan-bkv512c512`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v189-no-mlp-sac-no-scan-bkv512-compute512`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 SPLASH_BKV=512 SPLASH_BKV_COMPUTE=512 TOKAMAX_MAX_LOGIT_CONST=30.0`
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

Refuted. Workload `alekseyv-qwen3-v189-noscan-bkv512c512` completed cleanly
with valid loss ending at 12.0459 and `EXIT_CODE=0`, but regressed below the
confirmed no-scan frontier:

- Worker0 stdout: 56,837 tok/s average / 7,105 tok/s/chip / 40.8% MFU.
- Best observed worker0 steady step: 56,984 tok/s at step 2.
- Profile-derived host averages: `gke-tpu-964065d9-9vh8` 4.629782 s / 56,621
  tok/s, `gke-tpu-964065d9-c180` 4.616597 s / 56,783 tok/s.
- Final loss: 12.0459.

## Profile

Profile files:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v189-no-mlp-sac-no-scan-bkv512-compute512/plugins/profile/2026_06_04_22_33_58/`

The profile validates the stdout regression. Device-side
`jit_train_step(16781811678069776883)` spans are about 4.616-4.630 s across the
two hosts, versus the v183 no-scan profile-derived 4.497 s class.

## HLO Dump

Fresh train-step HLO:

- Optimized HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v189-no-mlp-sac-no-scan-bkv512-compute512/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 19,042,499 bytes.
- SHA256: `668dfe34f5eef201487104c076b15c8a7e999c5be616c7d26948fbf031509a1d`.
- Memory report: 27.66 GiB total bytes used.
- Text counts: all-gather 7,039, all-reduce 1,148, collective-permute 6,
  convolution 3,071, custom-call 2,806, copy 21,271, fusion 31,686.

Compared with the v182/v183 frontier HLO (`1ba7015c...`, 19,043,435 bytes),
the corrected BKV512 schedule is a real changed program, not a persistent-cache
replay. The change increases text-level communication/fusion/copy/custom-call
pressure and slows wall time.

## Verdict

Refuted. Do not carry `SPLASH_BKV=512 SPLASH_BKV_COMPUTE=512` in the no-scan
frontier. The smaller forward KV tile worsens the unrolled no-scan schedule
despite fitting and preserving normal loss.
