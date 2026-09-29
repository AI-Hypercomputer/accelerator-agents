---
title: "Qwen3 JAX v178 - MaxText CE without target cotangent"
type: experiment
hypothesis: "The MaxText custom VJP currently computes an unused cotangent for one-hot CE targets; returning a symbolic zero for that target cotangent may shrink the backward graph/residual without changing loss or parameter gradients."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v178-no-target-grad-maxtext-ce
status: refuted
tags: [qwen3-cc, jax, flax-nnx, ce, maxtext-ce, custom-vjp, activation-sharding, mlp, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v178 - MaxText CE without target cotangent

## Hypothesis under test

**Hypothesis**: The current MaxText/T5X-style
`cross_entropy_with_logits` custom VJP returns a gradient for the one-hot
`targets` argument:

```python
g_targets = -jnp.expand_dims(g, axis=-1) * log_softmax
```

In this trainer `targets` is produced from integer labels inside the loss and is
never differentiated as a model parameter. Returning that cotangent may force
the backward graph to retain `log_softmax` and materialize extra CE work that is
dead for training.

**Mechanism**: Derive a thin image from `v169-no-mlp-sac` and change only the
custom VJP:

- remove `log_softmax` from the saved residual;
- return `None` for the `targets` cotangent.

Keep v169's no-MLP-intermediate `_sac` change, all sharding, attention,
batch/sequence shape, CE primal math, and XLA flags fixed.

**Predicted signal**: Support requires CPU equivalence for loss/logits/parameter
gradients, a distinct HLO/profile with less CE/logit residual work, and
throughput above the confirmed v169/v170 frontier band (57,935-57,947 tok/s
average, 58,051 best observed).

**Falsification criterion**: Equivalence failure, compile/runtime failure,
invalid loss, HLO no-op without throughput gain, or a clean run at/below the
confirmed frontier.

## Setup

GKE workload `alekseyv-qwen3-v178-ce-notarget`.

- **Base image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v178-no-target-grad-maxtext-ce`
- **Image digest**: `sha256:f6557e14c8aa7b9d77fc4bebeb313f04441d7703fd1582d1e905f7b02a825b5f`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v178-no-target-grad-maxtext-ce`
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

- Worker 0: 57,770 tok/s average, 7,221 tok/s/chip, 41.4% MFU; final loss `12.0461`.
- Worker 1: 57,749 tok/s average, 7,219 tok/s/chip, 41.4% MFU; final loss `12.0461`.
- Best observed steady step: 57,941 tok/s on both workers step 2.
- Frontier comparison: below the confirmed v169/v170 frontier band of 57,935-57,947 tok/s repeated average and 58,051 best observed.

## Profile

Profile captured steps 12-14 under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v178-no-target-grad-maxtext-ce/plugins/profile/2026_06_04_20_58_50/`

The profile contains both TPU hosts:

- `gke-tpu-964065d9-9vh8`
- `gke-tpu-964065d9-c180`

## HLO Dump

No HLO objects were emitted under the v178 run prefix:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v178-no-target-grad-maxtext-ce/hlo/`

The run's short step-0/compile behavior and absence of fresh HLO are consistent
with persistent-cache reuse of the v169/v170 train-step program. That implies
the original `g_targets`/`log_softmax` path was already dead-code-eliminated
from the compiled training graph, because parameter gradients do not consume the
one-hot target cotangent.

## Verdict

Refuted / no-op. Returning `None` for the CE target cotangent is correct for the
training objective and passes CPU equivalence, but it does not produce a new
compiled program or improve wall time. Do not carry as a performance mechanism.
Keep the v169/v170 frontier unchanged.
