---
title: "Qwen3 JAX v174 - no pre-MLP norm-output sharding constraint"
type: experiment
hypothesis: "If the `_sac` on `post_attention_layernorm(hidden_states)` before the MLP is an over-eager interior constraint in the v169 frontier, removing only that constraint may improve the MLP input/downstream schedule while preserving residual-stream and attention-boundary constraints."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v174-no-pre-mlp-norm-sac
status: refuted
tags: [qwen3-cc, jax, flax-nnx, activation-sharding, mlp, rmsnorm, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v174 - no pre-MLP norm-output sharding constraint

## Hypothesis under test

**Hypothesis**: v169/v170 support removing only the large MLP intermediate
constraint. v172 and v173 refute relaxing the attention output or pre-attention
norm constraints. The remaining directly comparable MLP-side interior boundary
is the `_sac` around the post-attention RMSNorm output before the MLP input.

**Mechanism**: Derive a thin image from `v169-no-mlp-sac` and change only:

```python
x = self.mlp(_sac(self.post_attention_layernorm(hidden_states)))
```

to:

```python
x = self.mlp(self.post_attention_layernorm(hidden_states))
```

Keep v169's removal of the MLP intermediate `_sac`. Keep the layer-input
`_sac(hidden_states)`, pre-attention norm-output `_sac`, and attention-output
`_sac` intact.

**Predicted signal**: Support requires clean loss, CPU equivalence, a distinct
or meaningfully faster HLO/profile, and throughput above the confirmed v169/v170
frontier band (57,935-57,947 tok/s average, 58,051 best observed).

**Falsification criterion**: Equivalence failure, compile/runtime failure,
invalid loss, HLO no-op without throughput gain, or a clean run at/below the
confirmed frontier.

## Setup

GKE workload `alekseyv-qwen3-v174-nomlpnormsac`.

- **Base image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v174-no-pre-mlp-norm-sac`
- **Image digest**: `sha256:cae2e37808991266c5566a8bb0628abb14fb5606edefa8d24293d0e8671d09cf`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v174-no-pre-mlp-norm-sac`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v170 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

CPU equivalence passed against image
`<your-registry>/torchtitan-images/qwen3-8b-jax:v174-no-pre-mlp-norm-sac`
with `JAX_PLATFORMS=cpu python -u test_equivalence.py`.

Result: `==== EQUIVALENCE PASS ====`.

- Logits max delta: `2.682e-07`
- Loss delta: `0.000e+00`
- Gradients: all 25 HF parameters OK
- Max observed gradient delta: `7.823e-08`

## Results

Completed cleanly on both v6e-8 slices.

| worker | pod | avg tok/s | tok/s/chip | MFU | best step tok/s | final loss | exit |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | `alekseyv-qwen3-v174-nomlpnormsac-slice-job-0-0-ldjv8` | 57,759 | 7,220 | 41.4% | 57,879 | 12.0461 | 0 |
| 1 | `alekseyv-qwen3-v174-nomlpnormsac-slice-job-0-1-k2brw` | 57,737 | 7,217 | 41.4% | 57,883 | 12.0461 | 0 |

This is below the confirmed v169/v170 frontier band:

- v169: 57,947 / 57,919 tok/s average, best observed 58,051, 41.5% MFU.
- v170 rerun: 57,913 / 57,935 tok/s average, best observed 58,050, 41.5% MFU.

## Profile

Profile landed under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v174-no-pre-mlp-norm-sac/plugins/profile/2026_06_04_20_24_26/`

## HLO Dump

Optimized train-step HLO:

- Path: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v174-no-pre-mlp-norm-sac/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 783,590 bytes
- SHA256: `14ca3242bea4cc41c2897f6c5f5f3d7da24597980944a1863c85b98c495f905e`
- Quick counts: all-gather 140, all-reduce 36, collective-permute 3, convolution 66, custom-call 41, copy 318, fusion 753.

The HLO differs from the v169 frontier (`443b5f34...`, 783,769 bytes), but
major op counts are unchanged.

## Verdict

Refuted. The change is numerically equivalent and produces a real HLO variant,
but throughput regresses to 57,759 best-worker average / 41.4% MFU. Keep the
pre-MLP norm-output `_sac` and keep v169/v170 as the confirmed frontier.
