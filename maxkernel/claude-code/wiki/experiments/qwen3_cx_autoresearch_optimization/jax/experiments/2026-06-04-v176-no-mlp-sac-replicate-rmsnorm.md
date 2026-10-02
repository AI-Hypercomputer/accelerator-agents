---
title: "Qwen3 JAX v176 - no MLP intermediate sac plus replicated RMSNorm params"
type: experiment
hypothesis: "If v169's no-MLP-intermediate schedule changes the cost balance enough, the previously refuted replicated-RMSNorm parameter sharding may now reduce small all-gather pressure without losing wall time."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v176-no-mlp-sac-replicate-rmsnorm
status: refuted
tags: [qwen3-cc, jax, flax-nnx, sharding, rmsnorm, collectives, activation-sharding, mlp, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v176 - no MLP intermediate sac plus replicated RMSNorm params

## Hypothesis under test

**Hypothesis**: v165/v166 replicated RMSNorm parameters on the pre-v169 frontier.
That reduced HLO size and all-gather text count but did not improve wall time.
v169 then changed the schedule family by removing only the large MLP intermediate
activation constraint and became the confirmed frontier. This experiment tests
whether the RMSNorm parameter replication is beneficial in the v169 schedule
family.

**Mechanism**: Derive a thin image from `v169-no-mlp-sac` and change only the
RMSNorm parameter sharding entries:

```python
"model.layers.*.input_layernorm.weight":          (),
"model.layers.*.post_attention_layernorm.weight": (),
"model.norm.weight": (),
```

Keep v169's removal of the MLP intermediate `_sac`. Keep all activation
constraints, attention settings, batch/sequence shape, CE path, and XLA flags
fixed.

**Predicted signal**: Support requires clean loss, CPU equivalence, a distinct
HLO/profile with lower small-collective pressure, and throughput above the
confirmed v169/v170 frontier band (57,935-57,947 tok/s average, 58,051 best
observed).

**Falsification criterion**: Equivalence failure, compile/runtime failure,
invalid loss, HLO no-op without throughput gain, or a clean run at/below the
confirmed frontier.

## Setup

GKE workload `alekseyv-qwen3-v176-repnorm-mlpsac`.

- **Base image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v176-no-mlp-sac-replicate-rmsnorm`
- **Image digest**: `sha256:6e2c159f660fac6861fca2cd037afd9d46538c34dbb332b97f29c636a0743bba`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v176-no-mlp-sac-replicate-rmsnorm`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v170 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

CPU equivalence passed before TPU launch.

- Forward logits max |delta|: `2.682e-07`
- Loss max |delta|: `0.000e+00`
- Gradients: all 25 HF parameters matched.
- Max observed gradient |delta|: `7.823e-08`

## Results

Completed cleanly on v6e-8 with normal loss and `EXIT_CODE=0`.

- Worker 0: 57,718 tok/s average, 7,215 tok/s/chip, 41.4% MFU; final loss `12.0461`.
- Worker 1: 57,702 tok/s average, 7,213 tok/s/chip, 41.4% MFU; final loss `12.0461`.
- Best observed steady step: 57,868 tok/s on worker 1 step 2, 57,857 tok/s on worker 0 step 2.
- Frontier comparison: below the confirmed v169/v170 repeated band of 57,935-57,947 tok/s average and 58,051 best observed.

## Profile

Profile captured steps 12-14 under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v176-no-mlp-sac-replicate-rmsnorm/plugins/profile/2026_06_04_20_41_16/`

The profile contains both TPU hosts:

- `gke-tpu-964065d9-9vh8`
- `gke-tpu-964065d9-c180`

## HLO Dump

Fresh optimized train-step HLO:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v176-no-mlp-sac-replicate-rmsnorm/hlo/module_0103.jit_train_step.cl_854318611.after_optimizations.txt`

- Size: 778,035 bytes / 759.8 KiB
- SHA256: `d6839ab4eaed88451aa554ba3b0a53deda0386a08f3a9ae4e5eee9fa6c05ca0f`
- Text counts: all-gather 116, all-reduce 36, collective-permute 3, convolution 66, custom-call 41, copy 324, fusion 753.

Relative to v169/v170, replicated RMSNorm params reduce all-gather text
occurrences from 140 to 116 and slightly shrink HLO size, but copy count rises
from 318 to 324 and throughput regresses.

## Verdict

Refuted. The interaction is real in HLO space, but it does not improve wall
time. Do not carry replicated RMSNorm parameter sharding on top of the v169
no-MLP-intermediate schedule. Keep the v169/v170 frontier unchanged.
