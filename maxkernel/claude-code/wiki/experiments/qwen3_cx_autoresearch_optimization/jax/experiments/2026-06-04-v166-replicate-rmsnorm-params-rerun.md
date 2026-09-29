---
title: "Qwen3 JAX v166 - replicate RMSNorm params rerun"
type: experiment
hypothesis: "If v165's replicated-RMSNorm all-gather reduction is useful rather than noise, an exact rerun should preserve the distinct HLO family and meet or exceed the confirmed v153/v154 throughput band."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v165-replicate-rmsnorm
status: refuted
tags: [qwen3-cc, jax, flax-nnx, sharding, rmsnorm, collectives, maxtext-ce, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, activation-sharding, long-seq, v6e-8, rerun]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v166 - replicate RMSNorm params rerun

## Hypothesis under test

**Hypothesis**: v165 replicated RMSNorm parameters, emitted a distinct smaller
train-step HLO, and reduced all-gather occurrences from 211 to 177, but its
average throughput landed just below the confirmed v153/v154 frontier. If the
HLO reduction is useful rather than noise, an exact rerun should preserve the
same HLO family and meet or exceed the confirmed frontier band.

**Mechanism**: Exact rerun of v165:

```python
"model.layers.*.input_layernorm.weight": (),
"model.layers.*.post_attention_layernorm.weight": (),
"model.norm.weight": (),
```

All runtime flags, batch/sequence shape, image, and profile window are held
fixed.

**Predicted signal**: Support requires clean loss, HLO matching v165 or a cache
replay of that program, and average throughput at/above 57,754-57,757 tok/s or
best observed step above v153's 57,888 with comparable average.

**Falsification criterion**: Compile/runtime failure, invalid loss, fallback to
the v153 HLO family, or clean completion below the confirmed v153/v154 band
without a profile win.

## Setup

GKE workload `alekseyv-qwen3-v166-repnormr`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v165-replicate-rmsnorm`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v166-replicate-rmsnorm-params-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: confirmed v154 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss and `EXIT_CODE=0`.

- Worker 0: 57,735 tok/s average, 7,217 tok/s/chip, 41.4% MFU.
- Worker 1: 57,709 tok/s average, 7,214 tok/s/chip, 41.4% MFU.
- Best observed post-warmup step: 57,846 tok/s.
- Final loss: 12.0461 on both workers.

This confirms v165 as a near-frontier tie but not a new best. The best-worker
average remains below the confirmed v153/v154 repeated frontier band
(57,754-57,757 tok/s), and the best observed step is below v153's 57,888.

## Profile

Profile captured successfully:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v166-replicate-rmsnorm-params-rerun/plugins/profile/2026_06_04_19_10_39/`

Artifacts:

- `gke-tpu-964065d9-9vh8.xplane.pb`
- `gke-tpu-964065d9-c180.xplane.pb`
- matching `trace.json.gz` files for both hosts

## HLO Dump

No fresh HLO dump appeared under the v166 prefix, consistent with a persistent
cache replay of v165's compiled train-step program:

- v165 train-step HLO: `module_0105.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 778,215 bytes
- SHA256: `f928230a6d0bf402ad387404cace7c42dd989d5cc630b5f2d190f9cdc1eeff56`
- Quick count: all-gather occurrences dropped from v153's 211 to 177.

## Verdict

Refuted/tie. The replicated-RMSNorm HLO change is real and repeatable, but the
exact rerun did not beat the confirmed v153/v154 throughput band. Do not carry
replicated RMSNorm params.
