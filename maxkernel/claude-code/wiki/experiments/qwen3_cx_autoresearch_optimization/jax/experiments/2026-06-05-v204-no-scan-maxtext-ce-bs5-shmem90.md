---
title: "Qwen3 JAX v204 - no-scan MaxText CE batch 5 shmem90"
type: experiment
hypothesis: "The no-scan MaxText CE batch-5 run misses HBM by only 877 MiB; lowering the TPU scheduler shared-memory limit to 90 may select a lower-HBM schedule that fits while preserving enough throughput to beat batch 4."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-maxtext-ce-bs5-shmem90
status: invalid
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, ce, maxtext-ce, batch-scaling, scheduler, shared-memory, splash, tokamax-splash, collective-matmul, scoped-vmem, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v204 - no-scan MaxText CE batch 5 shmem90

## Hypothesis under test

**Hypothesis**: v203 direct MaxText CE batch 5/global batch 40 OOMed at
32.10 GiB, only 877 MiB above HBM capacity, with the largest temp being the
11.59 GiB batch-5 full-vocab lm-head/logit tensor. Lowering
`--xla_tpu_scheduler_percent_shared_memory_limit` from 100 to 90 may select a
lower-HBM schedule that fits. If it fits, the faster MaxText CE path plus
larger global batch may still beat the batch-4 frontier despite scheduler
tradeoffs.

**Mechanism**: Reuse v203 and change only
`--xla_tpu_scheduler_percent_shared_memory_limit=100` to `90`.

Keep the v169 no-MLP-intermediate-`_sac` image, no-scan, remat, tokamax Splash,
MaxText CE, activation sharding, `TOKAMAX_MAX_LOGIT_CONST=30.0`, scoped VMEM
100352, collective-matmul modes disabled, and batch 5 fixed.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v182/v183 confirmed 58,288-58,330 tok/s average / 41.8% MFU band.

**Falsification criterion**: Compile HBM OOM, runtime failure, invalid loss, or
clean completion below the frontier.

## Setup

GKE workload `alekseyv-qwen3-v204-noscan-bs5-shmem90`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v204-no-scan-maxtext-ce-bs5-shmem90`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=5 --seqlen=8192 --tp_parallelism=1`
- **Changed XLA flag**: `--xla_tpu_scheduler_percent_shared_memory_limit=90`
- **Global batch**: 40
- **Profile window**: steps 12-14 if the run reaches training.

## Equivalence

No model-code edit is made. This changes only a scheduler memory-pressure flag
relative to v203.

## Results

Invalid. Workload `alekseyv-qwen3-v204-noscan-bs5-shmem90` failed before
step 0 with a compile-time HBM OOM:

```text
RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm.
Used 32.10G of 31.25G hbm. Exceeded hbm capacity by 876.30M.
EXIT_CODE=1
```

Lowering scheduler shared-memory from 100 to 90 did not change the effective
batch-5 MaxText CE memory envelope enough to fit; it missed HBM by essentially
the same amount as v203.

## Profile

Not captured; the run failed during compile.

## HLO Dump

OOM reports were emitted by XLA, but no usable train-step profile was captured.

## Verdict

Invalid. The v203/v204 pair closes this direct batch-5 MaxText CE branch until
a real memory reduction removes roughly 0.9 GiB from the full-vocab lm-head/logit
path.
