---
title: "Qwen3 JAX v165 - replicate RMSNorm params"
type: experiment
hypothesis: "If tiny sharded RMSNorm parameters create avoidable small collective/update barriers, replicating only RMSNorm weights may improve the train-step schedule without changing math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v165-replicate-rmsnorm
status: refuted
tags: [qwen3-cc, jax, flax-nnx, sharding, rmsnorm, collectives, maxtext-ce, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v165 - replicate RMSNorm params

## Hypothesis under test

**Hypothesis**: The current HLO contains small all-reduce tuples carrying
`bf16[4096]` and `bf16[128]` values around norm/reduce-sum regions. QK-norm
weights are already replicated, but decoder RMSNorm and final RMSNorm weights
are FSDP-sharded. If those tiny sharded parameters introduce avoidable update or
collective barriers, replicating only the RMSNorm weights may select a cleaner
schedule.

**Mechanism**: Revert to the confirmed base image family and change only the
RMSNorm parameter sharding entries:

```python
"model.layers.*.input_layernorm.weight": (),
"model.layers.*.post_attention_layernorm.weight": (),
"model.norm.weight": (),
```

All large matmul, embedding, lm-head, CE, attention, optimizer, and runtime
flags are held fixed.

**Predicted signal**: Support requires clean loss, a distinct train-step HLO or
profile with lower small-collective/update pressure, and throughput above the
confirmed v153/v154 frontier band (57,754-57,757 tok/s average, 57,888 best
observed), or at least a clear profile win that motivates a paired follow-up.

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO no-op,
or throughput at/below the confirmed frontier without a clear profile win.

## Setup

GKE workload `alekseyv-qwen3-v165-repnorm`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v165-replicate-rmsnorm`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v165-replicate-rmsnorm-params`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: confirmed v154 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss and `EXIT_CODE=0`.

- Worker 0: 57,730 tok/s average, 7,216 tok/s/chip, 41.4% MFU.
- Worker 1: 57,707 tok/s average, 7,213 tok/s/chip, 41.4% MFU.
- Best observed post-warmup step: 57,889 tok/s.
- Final loss: 12.0461 on both workers.

This is much better than the v163/v164 layout-constraint regressions and very
close to the confirmed v153/v154 frontier, but the best-worker average remains
slightly below v153/v154's 57,754-57,757 tok/s repeated average. The best
observed step is effectively tied with v153's 57,888.

## Profile

Profile captured successfully:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v165-replicate-rmsnorm-params/plugins/profile/2026_06_04_19_04_55/`

Artifacts:

- `gke-tpu-964065d9-9vh8.xplane.pb`
- `gke-tpu-964065d9-c180.xplane.pb`
- matching `trace.json.gz` files for both hosts

## HLO Dump

Fresh optimized train-step HLO was emitted:

- `module_0105.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 778,215 bytes
- SHA256: `f928230a6d0bf402ad387404cace7c42dd989d5cc630b5f2d190f9cdc1eeff56`

Compared with v153's 783,949-byte HLO, this is a real structural change. A
quick opcode count shows all-gather occurrences dropped from 211 to 177 while
all-reduce, convolution, custom-call, and copy counts stayed unchanged.

## Verdict

Refuted/tie after v166 exact rerun. Replicating RMSNorm parameters reduced HLO
size and all-gather count, but repeated throughput stayed below the confirmed
v153/v154 average:

- v165 best-worker average: 57,730 tok/s.
- v166 exact rerun best-worker average: 57,735 tok/s.
- Confirmed frontier: 57,754-57,757 tok/s repeated average.

Do not carry replicated RMSNorm params. The HLO reduction is interesting but
not enough to improve wall time at this frontier.
