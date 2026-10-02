---
title: "Qwen3 JAX v167 - replicate RMSNorm params scoped VMEM 100864"
type: experiment
hypothesis: "If v165's replicated-RMSNorm HLO has a shifted scoped-VMEM optimum, raising scoped VMEM from 100352 to 100864 KiB may recover throughput while preserving the lower all-gather HLO family."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v165-replicate-rmsnorm
status: refuted
tags: [qwen3-cc, jax, flax-nnx, sharding, rmsnorm, collectives, scoped-vmem, maxtext-ce, splash, tokamax-splash, collective-matmul, remat, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v167 - replicate RMSNorm params scoped VMEM 100864

## Hypothesis under test

**Hypothesis**: v165/v166 replicated RMSNorm params, reduced HLO size and
all-gather count, but landed just below the confirmed v153/v154 throughput
band at scoped VMEM 100352 KiB. Because this is a distinct HLO family, the
previous base-stack VMEM bracket may not transfer exactly. Raising scoped VMEM
to 100864 KiB may select a better schedule for the replicated-RMSNorm family.

**Mechanism**: Keep the v165 replicated-RMSNorm image and change only:

```text
--xla_tpu_scoped_vmem_limit_kib=100352 -> 100864
```

All code, batch/sequence shape, CE, attention, and other runtime flags remain
fixed.

**Predicted signal**: Support requires clean loss, HLO in the replicated-RMSNorm
family, and throughput above v165/v166 and ideally above the confirmed v153/v154
frontier band (57,754-57,757 tok/s average).

**Falsification criterion**: Compile/runtime failure, invalid loss, fallback to
the base HLO family, or clean completion at/below v165/v166 without a profile
win.

## Setup

GKE workload `alekseyv-qwen3-v167-repnorm-vmem100864`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v165-replicate-rmsnorm`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v167-replicate-rmsnorm-vmem100864`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v154 stack, but `--xla_tpu_scoped_vmem_limit_kib=100864`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss and `EXIT_CODE=0`.

- Worker 0: 57,724 tok/s average, 7,216 tok/s/chip, 41.4% MFU.
- Worker 1: 57,697 tok/s average, 7,212 tok/s/chip, 41.4% MFU.
- Best observed post-warmup step: 57,854 tok/s.
- Final loss: 12.0461 on both workers.

This is below both v165/v166 replicated-RMSNorm runs and the confirmed v153/v154
frontier. Raising scoped VMEM to 100864 KiB did not recover throughput for the
replicated-RMSNorm HLO family.

## Profile

Profile captured successfully:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v167-replicate-rmsnorm-vmem100864/plugins/profile/2026_06_04_19_16_42/`

Artifacts:

- `gke-tpu-964065d9-9vh8.xplane.pb`
- `gke-tpu-964065d9-c180.xplane.pb`
- matching `trace.json.gz` files for both hosts

## HLO Dump

Fresh optimized train-step HLO was emitted:

- `module_0105.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 778,215 bytes
- SHA256: `f928230a6d0bf402ad387404cace7c42dd989d5cc630b5f2d190f9cdc1eeff56`

This is byte-identical to v165's replicated-RMSNorm HLO. The higher scoped-VMEM
setting did not select a new train-step program.

## Verdict

Refuted. The targeted high-side VMEM follow-up did not improve the
replicated-RMSNorm family:

- v165: 57,730 tok/s best-worker average, 57,889 best observed step.
- v166: 57,735 tok/s best-worker average, 57,846 best observed step.
- v167: 57,724 tok/s best-worker average, 57,854 best observed step.
- Confirmed frontier v153/v154: 57,754-57,757 tok/s repeated average.

Do not carry replicated RMSNorm params or the 100864 KiB VMEM setting. The
current frontier remains v153/v154.
