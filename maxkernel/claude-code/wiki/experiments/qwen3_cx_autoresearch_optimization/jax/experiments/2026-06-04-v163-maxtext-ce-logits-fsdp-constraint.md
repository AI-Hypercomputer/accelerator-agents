---
title: "Qwen3 JAX v163 - MaxText CE logits FSDP constraint"
type: experiment
hypothesis: "If the residual CE/logits path is paying an implicit layout or collective cost, explicitly constraining the MaxText CE logits to batch-sharded FSDP layout may select a better train-step schedule without changing math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v163-ce-logits-fsdp
status: refuted
tags: [qwen3-cc, jax, flax-nnx, ce, maxtext-ce, logits, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v163 - MaxText CE logits FSDP constraint

## Hypothesis under test

**Hypothesis**: The v162 rerun refuted the lm-head parameter-sharding variant,
but the retrospective still points at the CE/logits boundary as the best
structural target. The current MaxText CE path receives full logits with no
explicit layout constraint. If XLA is choosing an avoidable layout or collective
boundary around the logits/one-hot CE, pinning logits to the same batch-sharded
FSDP layout used for activations may reduce relayout pressure or select a better
schedule.

**Mechanism**: Revert to the confirmed base image family and change only
`_ce_maxtext`:

```python
logits = jax.lax.with_sharding_constraint(
    logits, NamedSharding(mesh, P("fsdp", None, None))
)
```

All runtime flags, batch/sequence shape, MaxText CE math, lm-head sharding, and
optimizer semantics are held fixed.

**Predicted signal**: Support requires clean loss, a distinct train-step HLO or
profile with lower CE/logit/layout pressure, and throughput above the confirmed
v153/v154 frontier band (57,754-57,757 tok/s average, 57,888 best observed) or
a material memory reduction that opens a follow-up batch-5 test.

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO no-op,
or throughput at/below the confirmed frontier without a clear memory/profile
win.

## Setup

GKE workload `alekseyv-qwen3-v163-ce-logits-fsdp`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v163-ce-logits-fsdp`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v163-maxtext-ce-logits-fsdp-constraint`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: confirmed v154 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss and `EXIT_CODE=0`, but
regressed below the confirmed frontier.

- Worker 0: 56,393 tok/s average, 7,049 tok/s/chip, 40.4% MFU.
- Worker 1: 56,408 tok/s average, 7,051 tok/s/chip, 40.4% MFU.
- Best observed post-warmup step: 56,742 tok/s.
- Final loss: 12.0461 on both workers.

The first step took 38.5 s, confirming this was a fresh compile/change rather
than a persistent-cache replay. The steady-state result is below v153/v154's
57,754-57,757 tok/s repeated average and 57,888 best observed step.

## Profile

Profile captured successfully:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v163-maxtext-ce-logits-fsdp-constraint/plugins/profile/2026_06_04_18_50_07/`

Artifacts:

- `gke-tpu-32cb1c36-gs53.xplane.pb`
- `gke-tpu-32cb1c36-jj8f.xplane.pb`
- matching `trace.json.gz` files for both hosts

## HLO Dump

Fresh optimized train-step HLO was emitted:

- `module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 784,080 bytes
- SHA256: `9f7762782921bf93af7dddc93c2c93d04207b6f73ab0ec2ceb205709d93c903f`

This is distinct from the v153/v154/v160 HLO family
(`3734fae066fc35c4b9c6c4cce9b0015413ab4513c34a09ba33a0e1862bff3a3d`)
and v161 (`3648b17f7b1fdb754e9b9330400997ed2e06c3f25543cb549bb90559a8b7a348`).

## Verdict

Refuted. The explicit CE logits FSDP constraint is a real graph change, but it
selects a slower schedule: 56,408 tok/s best-worker average / 40.4% MFU. Do not
carry the logits constraint. Further CE/logits work needs a deeper fused or
streamed mechanism, not a boundary layout constraint on the full logits tensor.
