---
title: "Qwen3 JAX v172 - no attention output sharding constraint"
type: experiment
hypothesis: "If the attention-output `_sac` before the residual add over-constrains a residual-sized transient in the v169 no-MLP-sac frontier, removing only that constraint may reduce relayout/copy pressure while preserving layer-boundary activation sharding."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v172-no-attn-output-sac
status: refuted
tags: [qwen3-cc, jax, flax-nnx, activation-sharding, attention, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v172 - no attention output sharding constraint

## Hypothesis under test

**Hypothesis**: v169/v170 showed that one over-eager interior activation
constraint can slow the final HLO family. The remaining `_sac` around the
attention output, immediately before adding the attention residual, may
similarly force a layout that blocks a faster residual-add/output-projection
schedule.

**Mechanism**: Derive a thin image from `v169-no-mlp-sac` and change only:

```python
x = _sac(self.self_attn(_sac(self.input_layernorm(hidden_states)), position_embeddings, attention_mask))
```

to:

```python
x = self.self_attn(_sac(self.input_layernorm(hidden_states)), position_embeddings, attention_mask)
```

Keep the layer-input `_sac(hidden_states)` constraint and the pre-attention and
pre-MLP norm-output constraints intact. The MLP intermediate remains unconstrained
as in v169/v170.

**Predicted signal**: Support requires clean loss, a distinct or meaningfully
faster HLO/profile, and throughput above the confirmed v169/v170 frontier band
(57,935-57,947 tok/s average, 58,051 best observed).

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO no-op,
or a clean run at/below the confirmed frontier without a profile/HLO win.

## Setup

GKE workload `alekseyv-qwen3-v172-noattnsac`.

- **Base image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v172-no-attn-output-sac`
- **Image digest**: `sha256:17d89a609f3608cf9a3e4f9f00d92877275bdab765cc063e8c68909934593339`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v172-no-attn-output-sac`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v170 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

CPU equivalence was rerun against image
`<your-registry>/torchtitan-images/qwen3-8b-jax:v172-no-attn-output-sac`
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
| 0 | `alekseyv-qwen3-v172-noattnsac-slice-job-0-0-hrmsm` | 57,770 | 7,221 | 41.4% | 57,915 | 12.0461 | 0 |
| 1 | `alekseyv-qwen3-v172-noattnsac-slice-job-0-1-dnmsv` | 57,749 | 7,219 | 41.4% | 57,922 | 12.0461 | 0 |

This is below the confirmed v169/v170 frontier band:

- v169: 57,947 / 57,919 tok/s average, best observed 58,051, 41.5% MFU.
- v170 rerun: 57,913 / 57,935 tok/s average, best observed 58,050, 41.5% MFU.

## Profile

Profile landed under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v172-no-attn-output-sac/plugins/profile/2026_06_04_20_08_58/`

## HLO Dump

Optimized train-step HLO:

- Path: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v172-no-attn-output-sac/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 783,590 bytes
- SHA256: `fdcf65285f8c251968d5fe4279d8914e6b8e50978e3f43f236aa5d52c54afcb5`
- Quick counts: all-gather 140, all-reduce 36, collective-permute 3, convolution 66, custom-call 41, copy 318, fusion 753.

The op counts match v169/v170, but the HLO size/hash changed from the frontier
program (`443b5f34...`, 783,769 bytes). Removing the attention-output
constraint therefore produces a real but slower schedule/layout variant rather
than a no-op.

## Verdict

Refuted. The change is numerically equivalent and compiles/runs cleanly, but it
regresses throughput to 57,770 best-worker average / 41.4% MFU. Keep the
attention-output `_sac` and retain v169/v170 as the confirmed frontier.
