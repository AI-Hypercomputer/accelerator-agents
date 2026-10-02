---
title: "Qwen3 JAX v173 - no pre-attention norm-output sharding constraint"
type: experiment
hypothesis: "If the `_sac` on `input_layernorm(hidden_states)` before attention is an over-eager interior constraint in the v169 frontier, removing only that constraint may improve the QKV/attention schedule while preserving the residual-stream attention-output boundary."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v173-no-pre-attn-norm-sac
status: refuted
tags: [qwen3-cc, jax, flax-nnx, activation-sharding, attention, rmsnorm, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v173 - no pre-attention norm-output sharding constraint

## Hypothesis under test

**Hypothesis**: v172 showed that removing the attention-output `_sac` regresses,
so the residual-stream attention boundary is useful. The next smallest remaining
interior attention constraint is the `_sac` on the pre-attention RMSNorm output.
That tensor feeds only QKV projection/attention and may not need the same
activation layout pinning as the residual stream.

**Mechanism**: Derive a thin image from `v169-no-mlp-sac` and change only:

```python
x = _sac(self.self_attn(_sac(self.input_layernorm(hidden_states)), position_embeddings, attention_mask))
```

to:

```python
x = _sac(self.self_attn(self.input_layernorm(hidden_states), position_embeddings, attention_mask))
```

Keep v169's removal of the MLP intermediate `_sac`. Keep the layer-input
`_sac(hidden_states)`, attention-output `_sac`, and pre-MLP norm-output `_sac`
intact.

**Predicted signal**: Support requires clean loss, CPU equivalence, a distinct
or meaningfully faster HLO/profile, and throughput above the confirmed v169/v170
frontier band (57,935-57,947 tok/s average, 58,051 best observed).

**Falsification criterion**: Equivalence failure, compile/runtime failure,
invalid loss, HLO no-op without throughput gain, or a clean run at/below the
confirmed frontier.

## Setup

GKE workload `alekseyv-qwen3-v173-noinnormsac`.

- **Base image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v173-no-pre-attn-norm-sac`
- **Image digest**: `sha256:319b8515c6fa648faa17e2aae412c7b91d79ee74ae4b3e4201e8afd37c8c5f7c`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v173-no-pre-attn-norm-sac`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v170 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

CPU equivalence passed against image
`<your-registry>/torchtitan-images/qwen3-8b-jax:v173-no-pre-attn-norm-sac`
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
| 0 | `alekseyv-qwen3-v173-noinnormsac-slice-job-0-0-95wfg` | 57,766 | 7,221 | 41.4% | 57,884 | 12.0461 | 0 |
| 1 | `alekseyv-qwen3-v173-noinnormsac-slice-job-0-1-bxwbz` | 57,740 | 7,218 | 41.4% | 57,884 | 12.0461 | 0 |

This is below the confirmed v169/v170 frontier band:

- v169: 57,947 / 57,919 tok/s average, best observed 58,051, 41.5% MFU.
- v170 rerun: 57,913 / 57,935 tok/s average, best observed 58,050, 41.5% MFU.

## Profile

Profile landed under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v173-no-pre-attn-norm-sac/plugins/profile/2026_06_04_20_17_57/`

## HLO Dump

Optimized train-step HLO:

- Path: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v173-no-pre-attn-norm-sac/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 783,587 bytes
- SHA256: `06b6bbab8ca9dc93be1d6984eed9673ab7501b082b3137f439f929f0819ee49f`
- Quick counts: all-gather 140, all-reduce 36, collective-permute 3, convolution 66, custom-call 41, copy 318, fusion 753.

The HLO differs from both v169 (`443b5f34...`, 783,769 bytes) and v172
(`fdcf6528...`, 783,590 bytes), but the major op counts are unchanged.

## Verdict

Refuted. The change is numerically equivalent and produces a real HLO variant,
but throughput regresses to 57,766 best-worker average / 41.4% MFU. Keep the
pre-attention norm-output `_sac` and keep v169/v170 as the confirmed frontier.
