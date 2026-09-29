---
title: "Qwen3 JAX v257 - QK-norm after transpose with Tokamax Splash env"
type: experiment
hypothesis: "With the correct v230/v236 Tokamax Splash runtime restored, applying Q/K RMSNorm after the q/k transpose may expose a better QK-norm+RoPE layout without changing math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v254-qknorm-after-transpose-runtime-tokamax
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, qk-norm, rope, layout, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v257 - QK-norm after transpose with Tokamax Splash env

## Hypothesis under test

**Hypothesis**: The v254 launch was invalid because it omitted Tokamax Splash
environment variables and produced HLO with `tokamax` count 0. Rerunning the
same QK-norm-after-transpose image with the correct v230/v236 runtime will
validly test whether applying Q/K RMSNorm in `(B,H,T,D)` layout, immediately
before RoPE, improves the QK-norm+RoPE lowering.

**Mechanism**: Reuse the v254 image unchanged. The only difference from v254 is
restoring:

```text
USE_TOKAMAX_SPLASH=1
TOKAMAX_MAX_LOGIT_CONST=30.0
```

Keep the v230 RS2+AR2 runtime stack fixed: no-scan, remat, MaxText CE,
activation sharding, scoped VMEM 100352 KiB, all-gather on async collective
fusion, SparseCore reduce-scatter/all-reduce offload plus aggregator, RS
latency multiplier 2, AR latency multiplier 2.

**Predicted signal**: Support requires a clean finite-loss run above the current
v230/v236 frontier band (**59,749-59,750 tok/s / 42.9% MFU**) or a clear XProf
reduction in QK-norm/RoPE-adjacent fusion/custom-call cost at in-band
throughput.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax HLO markers, or clean completion below the frontier.

## Setup

GKE workload: `alekseyv-q3-v257-qknorm-xpose`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v254-qknorm-after-transpose`
- **Image digest**: `sha256:53d1fd3f552af51368c14380109c4d7800578db5efffdd43ef5a11b755ec1ef5`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v257-no-scan-sparsecore-rs-ar-rs2-ar2-qknorm-after-transpose-tokamax`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Moving Q/K RMSNorm after transpose preserves the reduced last dimension
(`head_dim`) and leaves RMSNorm params unchanged. Cluster loss and HLO/profile
comparison gate validity.

## Results

Completed cleanly on both workers with `EXIT_CODE=0`. The corrected Tokamax
Splash path was active:

```text
[tokamax-splash] config: bq=2048 bkv=1024 bq_dkv=2048 bkv_dkv=2048 base2=True fuse_recip=True exp_sched=False
```

Throughput:

- Worker 0: **59,296 tok/s**, **7,412 tok/s/chip**, **42.5% MFU**.
- Worker 1: **59,328 tok/s**, **7,416 tok/s/chip**, **42.5% MFU**.
- Loss stayed finite and ended at **12.0458** on both workers.

This is valid, but below the v230/v236 frontier by about **421-454 tok/s**.
The QK-norm-after-transpose layout rewrite does not improve the current
QK-norm+RoPE/Splash lowering.

## Profile

- XProf run:
  `2026-06-05-qwen3-jax-v257-no-scan-sparsecore-rs-ar-rs2-ar2-qknorm-after-transpose-tokamax/2026_06_05_13_08_02`
- Step time average: **4430.7 ms**
- MXU utilization: **67.1%**
- Peak HBM: **30.95 GiB**
- Op profile top buckets: convolution fusion **57,028.9 ms / 53.6%**; custom
  fusion **2,997.0 ms / 2.8%**; visible async all-gather done/start tails about
  **1.2% / 0.6%** for the largest all-gather group.

## HLO Dump

- Optimized train-step HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v257-no-scan-sparsecore-rs-ar-rs2-ar2-qknorm-after-transpose-tokamax/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Local copy:
  `/tmp/qwen3-v257-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `bfd1c54cb811133970fc8a65fd4a266698062d67bac3d10db7b21b05f4bd7e07`
- Size: **19,575,537 bytes / 116,637 lines**
- Key text counts: `all-gather` 6,901; `all-reduce` 165;
  `reduce-scatter` 2,774; `async-start` 361; `async-done` 361;
  `convolution` 3,087; `custom-call` 3,536; `fusion` 30,653;
  `copy` 21,394; `tokamax` 2.

## Verdict

**Refuted.** With the correct Tokamax Splash runtime restored, moving Q/K
RMSNorm after the q/k transpose preserves loss but remains below the native-JAX
frontier. Keep the original QK-norm placement.
