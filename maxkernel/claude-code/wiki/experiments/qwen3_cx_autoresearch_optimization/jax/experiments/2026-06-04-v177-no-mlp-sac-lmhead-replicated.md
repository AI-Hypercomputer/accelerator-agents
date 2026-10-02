---
title: "Qwen3 JAX v177 - no MLP intermediate sac plus replicated lm_head"
type: experiment
hypothesis: "If v169's no-MLP-intermediate schedule changes the final CE/lm-head cost balance, fully replicating lm_head.weight may now reduce final projection or CE collective pressure enough to overcome its copy/fusion overhead."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v177-no-mlp-sac-lmhead-repl
status: refuted
tags: [qwen3-cc, jax, flax-nnx, lm-head, sharding, ce, maxtext-ce, activation-sharding, mlp, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v177 - no MLP intermediate sac plus replicated lm_head

## Hypothesis under test

**Hypothesis**: v168 fully replicated `lm_head.weight` on the pre-v169 frontier.
It produced a distinct HLO and fewer all-gather text occurrences but regressed
wall time. v169 later changed the schedule family by removing only the large
MLP intermediate `_sac` and became the confirmed frontier. This experiment tests
whether lm-head replication interacts favorably with the v169 schedule.

**Mechanism**: Derive a thin image from `v169-no-mlp-sac` and change only the
lm-head parameter sharding plan entry:

```python
"lm_head.weight": ()
```

Keep v169's removal of the MLP intermediate `_sac`. Keep all activation
constraints, RMSNorm parameter sharding, attention settings, batch/sequence
shape, CE path, and XLA flags fixed.

**Predicted signal**: Support requires clean loss, CPU equivalence, a distinct
HLO/profile, and throughput above the confirmed v169/v170 frontier band
(57,935-57,947 tok/s average, 58,051 best observed).

**Falsification criterion**: Equivalence failure, compile/runtime failure,
invalid loss, OOM from larger lm-head replication, HLO no-op without throughput
gain, or a clean run at/below the confirmed frontier.

## Setup

GKE workload `alekseyv-qwen3-v177-lmrep-mlpsac`.

- **Base image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v177-no-mlp-sac-lmhead-repl`
- **Image digest**: `sha256:0daba85f04a6e002bc0862d9bf87a47a158bd1c5b33bbee9c90ecd33076825e0`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v177-no-mlp-sac-lmhead-repl`
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

- Worker 0: 56,449 tok/s average, 7,056 tok/s/chip, 40.5% MFU; final loss `12.0461`.
- Worker 1: 56,423 tok/s average, 7,053 tok/s/chip, 40.5% MFU; final loss `12.0461`.
- Best observed steady step: 56,839 tok/s on worker 1 step 2, 56,833 tok/s on worker 0 step 2.
- Frontier comparison: well below the confirmed v169/v170 frontier band of 57,935-57,947 tok/s average and 58,051 best observed.

## Profile

Profile captured steps 12-14 under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v177-no-mlp-sac-lmhead-repl/plugins/profile/2026_06_04_20_49_44/`

The profile contains both TPU hosts:

- `gke-tpu-32cb1c36-gs53`
- `gke-tpu-32cb1c36-jj8f`

## HLO Dump

Fresh optimized train-step HLO:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v177-no-mlp-sac-lmhead-repl/hlo/module_0107.jit_train_step.cl_854318611.after_optimizations.txt`

- Size: 780,640 bytes / 762.34 KiB
- SHA256: `ac73f41c9703b91efddab337d55cbd99061bf718f14d92ec718904e7006de4ae`
- Text counts: all-gather 136, all-reduce 36, collective-permute 3, convolution 66, custom-call 41, copy 310, fusion 752.

Relative to v169/v170, full lm-head replication changes the HLO and reduces
all-gather text occurrences slightly from 140 to 136, but wall time regresses
substantially.

## Verdict

Refuted. Fully replicating `lm_head.weight` remains harmful even on top of the
v169 no-MLP-intermediate schedule. Do not carry this interaction. Keep the
v169/v170 frontier unchanged.
