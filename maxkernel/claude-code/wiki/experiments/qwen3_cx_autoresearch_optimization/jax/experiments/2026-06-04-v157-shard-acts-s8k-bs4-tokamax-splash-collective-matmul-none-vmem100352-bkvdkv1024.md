---
title: "Qwen3 JAX v157 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 100352 bkv dkv 1024"
type: experiment
hypothesis: "If the remaining backward DKV custom-call bucket reacts differently on the confirmed batch-4 frontier than it did on the older bs3 stack, then BKV_DKV1024 may improve throughput."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, attention-block-size, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v157 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 100352 bkv dkv 1024

## Hypothesis under test

**Hypothesis**: On the confirmed batch-4/global-32 frontier, smaller backward
DKV KV tiles may reduce the DKV custom-call bucket enough to beat the default
DKV tiling.

**Mechanism**: Keep v154's confirmed command, image, XLA flag stack,
`--xla_tpu_scoped_vmem_limit_kib=100352`, mesh, batch, and profile window
fixed. Add only `SPLASH_BKV_DKV=1024 SPLASH_BKV_DKV_COMPUTE=1024`.

**Predicted signal**: Support requires clean loss and a best-worker average
above the confirmed v153/v154 band (57,754-57,757 tok/s), with changed HLO or
profile evidence that DKV custom-call cost falls.

**Falsification criterion**: Compile/runtime failure, invalid loss, unchanged
or slower throughput, or profile showing DKV custom-call grows as it did in the
older v028 bs3 stack.

## Setup

GKE workload `alekseyv-qwen3-v157-bs4-bkvdkv1024`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v157-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352-bkvdkv1024`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0 SPLASH_BKV_DKV=1024 SPLASH_BKV_DKV_COMPUTE=1024`
- **XLA flags**: exact v154 stack, including `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers, but regressed substantially from the
confirmed v153/v154 `--xla_tpu_scoped_vmem_limit_kib=100352` frontier:

- Worker 0: 55,258 tok/s average, 6,907 tok/s/chip, 39.6% MFU; final loss
  12.0461.
- Worker 1: 55,237 tok/s average, 6,905 tok/s/chip, 39.6% MFU; final loss
  12.0461.
- Best steady step was about 55,380 tok/s, far below v154's 57,754 tok/s
  average and v153's 57,888 best observed step.

Loss remained normal, so this is a valid performance refutation rather than a
correctness failure.

## Profile

Profile artifacts were written under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v157-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352-bkvdkv1024/plugins/profile/2026_06_04_17_54_52/`

Both host xplane files are present. XProf HTTP indexing remained unreliable in
this session, so the decision uses worker logs plus direct GCS HLO/profile
artifacts.

## HLO Dump

Fresh HLO was emitted:

- Optimized HLO:
  `module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 783,982 bytes
- SHA256:
  `5be74e908e49455d08150bed79701690304fc72bae18460845d1ecba43d994eb`
- Compiled memory: 26.86 GiB

Splash metadata confirms the changed DKV tiling:

`block_q=2048`, `block_kv=1024`, `block_kv_compute=1024`,
`block_q_dkv=2048`, `block_kv_dkv=1024`,
`block_kv_dkv_compute=1024`.

The DKV custom-call scratch allocation grows to a 4.12 GiB preallocated temp,
including `f32[4,8,32,8192,128]`, which matches the observed throughput
regression.

## Verdict

**Refuted.** Smaller DKV KV tiles on the current batch-4/vmem100352 frontier
reduce throughput to 39.6% MFU. Keep the default forward and DKV Splash block
sizes; do not carry `SPLASH_BKV_DKV=1024` or
`SPLASH_BKV_DKV_COMPUTE=1024`.
