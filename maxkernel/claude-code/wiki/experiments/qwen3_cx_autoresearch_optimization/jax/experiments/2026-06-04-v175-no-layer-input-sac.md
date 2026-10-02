---
title: "Qwen3 JAX v175 - no decoder layer input sharding constraint"
type: experiment
hypothesis: "If the decoder layer input `_sac(hidden_states)` is an over-eager residual-stream boundary constraint in the v169 frontier, removing only that constraint may improve the residual/attention/MLP schedule; if it regresses, v169's win is isolated to the MLP intermediate constraint."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v175-no-layer-input-sac
status: refuted
tags: [qwen3-cc, jax, flax-nnx, activation-sharding, residual, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v175 - no decoder layer input sharding constraint

## Hypothesis under test

**Hypothesis**: v172-v174 refuted removing each remaining local interior
activation constraint around attention and MLP norm outputs. The final directly
comparable boundary stress test is the decoder layer input `_sac(hidden_states)`.
Removing it tests whether v169's improvement generalizes to residual-stream
boundary constraints or is specific to the large MLP intermediate.

**Mechanism**: Derive a thin image from `v169-no-mlp-sac` and remove only:

```python
hidden_states = _sac(hidden_states)
```

Keep v169's removal of the MLP intermediate `_sac`. Keep pre-attention
norm-output `_sac`, attention-output `_sac`, and pre-MLP norm-output `_sac`
intact.

**Predicted signal**: Support requires clean loss, CPU equivalence, a distinct
or meaningfully faster HLO/profile, and throughput above the confirmed v169/v170
frontier band (57,935-57,947 tok/s average, 58,051 best observed).

**Falsification criterion**: Equivalence failure, compile/runtime failure,
invalid loss, HLO no-op without throughput gain, or a clean run at/below the
confirmed frontier.

## Setup

GKE workload `alekseyv-qwen3-v175-nolayersac`.

- **Base image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v175-no-layer-input-sac`
- **Image digest**: `sha256:17388c233e2c9edc3442a3cd8cd3011b7174306d158ca2938c42bdf87d076d27`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v175-no-layer-input-sac`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v170 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

CPU equivalence passed against image
`<your-registry>/torchtitan-images/qwen3-8b-jax:v175-no-layer-input-sac`
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
| 0 | `alekseyv-qwen3-v175-nolayersac-slice-job-0-0-pd5l8` | 57,751 | 7,219 | 41.4% | 57,858 | 12.0461 | 0 |
| 1 | `alekseyv-qwen3-v175-nolayersac-slice-job-0-1-9fm82` | 57,738 | 7,217 | 41.4% | 57,866 | 12.0461 | 0 |

This is below the confirmed v169/v170 frontier band:

- v169: 57,947 / 57,919 tok/s average, best observed 58,051, 41.5% MFU.
- v170 rerun: 57,913 / 57,935 tok/s average, best observed 58,050, 41.5% MFU.

## Profile

Profiles landed under:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v175-no-layer-input-sac/plugins/profile/2026_06_04_20_31_17/`
- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v175-no-layer-input-sac/plugins/profile/2026_06_04_20_31_18/`

## HLO Dump

Optimized train-step HLO:

- Path: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v175-no-layer-input-sac/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 783,587 bytes
- SHA256: `76fc341db3f31dbef527b130e601dce97d3100c2dbd259ebca185f5e2b1d27f1`
- Quick counts: all-gather 140, all-reduce 36, collective-permute 3, convolution 66, custom-call 41, copy 318, fusion 753.

The HLO differs from the v169 frontier (`443b5f34...`, 783,769 bytes), but
major op counts are unchanged.

## Verdict

Refuted. The change is numerically equivalent and produces a real HLO variant,
but throughput regresses to 57,751 best-worker average / 41.4% MFU. Keep the
decoder layer input `_sac`; v169's supported gain remains specific to removing
the large MLP intermediate `_sac`.
