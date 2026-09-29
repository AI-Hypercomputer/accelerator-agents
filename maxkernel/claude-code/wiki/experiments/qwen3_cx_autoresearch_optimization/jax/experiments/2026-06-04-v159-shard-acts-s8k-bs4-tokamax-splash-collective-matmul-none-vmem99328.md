---
title: "Qwen3 JAX v159 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 99328"
type: experiment
hypothesis: "If the scoped-VMEM optimum is a narrow lower-side point rather than exactly 100352 KiB, then the midpoint between confirmed 98304 and 100352 may select a faster or equally fast schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, scheduler, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v159 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 99328

## Hypothesis under test

**Hypothesis**: v153/v154 confirmed 100352 KiB as the current frontier, v150/v151
confirmed 98304 KiB as lower but still strong, and v155/v152 refuted the high
side. A lower-side midpoint at 99328 KiB may expose whether the optimum is a
narrow schedule choice below 100352 or whether 100352 remains the local peak.

**Mechanism**: Keep v154 fixed except change only:

```text
--xla_tpu_scoped_vmem_limit_kib=99328
```

All other command shape, image, mesh, batch, profile window, Splash settings,
and scheduler flags stay identical to v154.

**Predicted signal**: Support requires clean loss and a best-worker average
above the confirmed v153/v154 band (57,754-57,757 tok/s), ideally with a changed
train-step HLO/profile. A result between v151 and v154 refutes the narrower
optimum hypothesis but helps close the lower-side bracket.

**Falsification criterion**: Compile/runtime failure, invalid loss, unchanged
HLO with no throughput gain, or throughput below the confirmed frontier.

## Setup

GKE workload `alekseyv-qwen3-v159-bs4-vmem99328`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v159-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem99328`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: exact v154 stack except `--xla_tpu_scoped_vmem_limit_kib=99328`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss but below the confirmed
100352 KiB frontier:

- Worker 0: 57,519 tok/s average, 7,190 tok/s/chip, 41.2% MFU; final loss
  12.0460.
- Worker 1: 57,494 tok/s average, 7,187 tok/s/chip, 41.2% MFU; final loss
  12.0460.
- Best steady steps were about 57,625 tok/s, below v154's 57,754 tok/s average
  and v153's 57,888 best observed step.

## Profile

Profile artifacts were written under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v159-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem99328/plugins/profile/`

Both host xplane files are present. XProf HTTP indexing remained unreliable, so
the decision uses worker logs plus direct GCS HLO/profile artifacts.

## HLO Dump

Fresh train-step HLO was emitted as
`module_0267.jit_train_step.cl_854318611.after_optimizations.txt`:

- Size: 784,575 bytes
- SHA256:
  `fd192aaa99e632c9a2affba51d0309a58ff2449fac37c8663517df6d42850f49`
- Compiled memory: 26.86 GiB

This is a real schedule/HLO change from v153/v154, but the changed schedule is
slower.

## Verdict

**Refuted.** The lower-side midpoint at 99328 KiB changes the train-step HLO
but regresses to 57.5k tok/s. Keep 100352 KiB as the current scoped-VMEM
frontier.
