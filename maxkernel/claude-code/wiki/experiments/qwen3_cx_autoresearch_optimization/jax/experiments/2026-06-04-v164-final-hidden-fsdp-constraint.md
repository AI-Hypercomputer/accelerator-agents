---
title: "Qwen3 JAX v164 - final hidden FSDP constraint"
type: experiment
hypothesis: "If the final RMSNorm to lm-head boundary lacks the activation-sharding constraint used inside decoder layers, constraining the final hidden state to FSDP layout before lm_head may improve the CE/lm-head schedule without changing math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v164-final-hidden-fsdp
status: refuted
tags: [qwen3-cc, jax, flax-nnx, lm-head, activation-sharding, sharding-constraint, ce, maxtext-ce, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v164 - final hidden FSDP constraint

## Hypothesis under test

**Hypothesis**: v163 showed that constraining the full logits tensor is a real
but slower HLO. A smaller layout boundary may be better: the model applies
activation sharding constraints inside decoder layers, but the final
`Qwen3Model.__call__` returns `self.norm(hidden_states)` directly. If the final
RMSNorm output enters `lm_head` without the same FSDP batch-sharded layout, XLA
may insert an avoidable relayout or choose a weaker lm-head/CE schedule.

**Mechanism**: Revert to the confirmed base image family and change only
`Qwen3ForCausalLM.__call__`:

```python
hidden = _sac(hidden)
```

immediately after `hidden = self.model(input_ids, position_ids)` and before the
`return_hidden` check / `lm_head` call. This uses the existing activation
sharding helper and preserves math.

**Predicted signal**: Support requires clean loss, a distinct train-step HLO or
profile with reduced lm-head/CE/layout pressure, and throughput above the
confirmed v153/v154 frontier band (57,754-57,757 tok/s average, 57,888 best
observed) or a material memory reduction that motivates batch-5.

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO no-op,
or throughput at/below the confirmed frontier without a clear memory/profile
win.

## Setup

GKE workload `alekseyv-qwen3-v164-finalhidden`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v164-final-hidden-fsdp`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v164-final-hidden-fsdp-constraint`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: confirmed v154 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss and `EXIT_CODE=0`, but
regressed below the confirmed frontier.

- Worker 0: 56,380 tok/s average, 7,047 tok/s/chip, 40.4% MFU.
- Worker 1: 56,365 tok/s average, 7,046 tok/s/chip, 40.4% MFU.
- Best observed post-warmup step: 56,726 tok/s.
- Final loss: 12.0461 on both workers.

The first step took 37.8 s, confirming a fresh compile/change. The steady-state
result is below v153/v154's 57,754-57,757 tok/s repeated average and 57,888
best observed step.

## Profile

Profile captured successfully:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v164-final-hidden-fsdp-constraint/plugins/profile/2026_06_04_18_57_15/`

Artifacts:

- `gke-tpu-32cb1c36-gs53.xplane.pb`
- `gke-tpu-32cb1c36-jj8f.xplane.pb`
- matching `trace.json.gz` files for both hosts

## HLO Dump

Fresh optimized train-step HLO was emitted:

- `module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 784,126 bytes
- SHA256: `59fb6287422baec57b38443c7e58063e531ba1d98205de3aca1017ce5c37aa01`

This is distinct from the v153/v154/v160 HLO family and from the v163
full-logits constraint HLO.

## Verdict

Refuted. Constraining the final hidden state before `lm_head` is a real graph
change, but it selects a slower schedule: 56,380 tok/s best-worker average /
40.4% MFU. Do not carry the final-hidden constraint. The local full-logits and
lm-head-boundary layout-constraint variants are now closed unless profile
analysis exposes a more specific relayout target.
