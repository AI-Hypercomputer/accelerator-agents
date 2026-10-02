---
title: "Qwen3 JAX v156 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 100352 bkv512"
type: experiment
hypothesis: "If smaller forward Splash KV blocks reduce the forward residual custom-call bucket on the confirmed batch-4 frontier, then BKV512 may improve throughput despite earlier bs3 regressions."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, attention-block-size, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v156 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 100352 bkv512

## Hypothesis under test

**Hypothesis**: On the confirmed batch-4/global-32 frontier, the remaining
forward Splash residual custom-call cost may prefer smaller forward KV tiles.

**Mechanism**: Keep v154's confirmed command, image, XLA flag stack,
`--xla_tpu_scoped_vmem_limit_kib=100352`, mesh, batch, and profile window
fixed. Add only `SPLASH_BKV=512 SPLASH_BKV_COMPUTE=512`.

**Predicted signal**: Support requires clean loss and a best-worker average
above the confirmed v153/v154 band (57,754-57,757 tok/s), with a changed HLO or
profile showing reduced forward residual custom-call cost.

**Falsification criterion**: Compile/runtime failure, invalid loss, unchanged
or slower throughput, or profile showing the forward custom-call bucket grows
as it did in the older v027 bs3 stack.

## Setup

GKE workload `alekseyv-qwen3-v156-bs4-bkv512`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v156-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352-bkv512`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0 SPLASH_BKV=512 SPLASH_BKV_COMPUTE=512`
- **XLA flags**: exact v154 stack, including `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers.

- Worker 0: average 56,541 tok/s (7,068/chip), approx 40.5% MFU; final loss 12.0458.
- Worker 1: average 56,521 tok/s (7,065/chip), approx 40.5% MFU; final loss 12.0458.
- Best observed steady step: 56,643 tok/s.
- Both pods exited with `EXIT_CODE=0`.

This regresses by roughly 1,213 tok/s (-2.1%) versus the confirmed v154
100352 frontier (57,754 tok/s).

## Profile

Profile files were written under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v156-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352-bkv512/plugins/profile/2026_06_04_17_49_22/`

Files present:

- `gke-tpu-964065d9-c180.xplane.pb`
- `gke-tpu-964065d9-c180.trace.json.gz`
- `gke-tpu-964065d9-9vh8.xplane.pb`
- `gke-tpu-964065d9-9vh8.trace.json.gz`

The local XProf server was unavailable after restart, so no aggregate op
profile was extracted during the loop.

## HLO Dump

Fresh HLO dump was emitted for the train-step program:

- HLO: `module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 783,923 bytes
- SHA256: `9b5c23ed2a0035b7352e87ce0a5dcbbd3e0c89bf45c6d5a66dc4b9e061f044f4`
- Compiled memory: 28,841,540,536 bytes / 26.86 GiB
- Splash config confirmed in logs: `bq=2048 bkv=512 bq_dkv=2048 bkv_dkv=2048`.

## Verdict

**Refuted.** Smaller forward KV blocks are harmful on the confirmed batch-4
frontier. Do not carry `SPLASH_BKV=512`; restore default forward KV tiling.
