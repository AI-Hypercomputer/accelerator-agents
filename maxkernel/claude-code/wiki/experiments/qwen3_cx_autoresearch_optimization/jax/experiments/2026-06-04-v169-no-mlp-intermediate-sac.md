---
title: "Qwen3 JAX v169 - no MLP intermediate sharding constraint"
type: experiment
hypothesis: "If the activation sharding constraint on the large MLP gate/up product forces an unfavorable intermediate layout, removing only that constraint may reduce copy/fusion pressure without weakening layer-boundary activation sharding."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac
status: supported
tags: [qwen3-cc, jax, flax-nnx, activation-sharding, mlp, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v169 - no MLP intermediate sharding constraint

## Hypothesis under test

**Hypothesis**: The current frontier keeps activation sharding constraints at
layer boundaries and also constrains the large MLP intermediate
`silu(gate_proj(x)) * up_proj(x)` before `down_proj`. That intermediate has
shape `[B, L, intermediate]`; pinning it to `P(fsdp, None, None)` may force
extra copies or block a better down-projection layout. Removing only this
intermediate constraint may reduce copy/fusion pressure while preserving all
residual-stream layer-boundary constraints.

**Mechanism**: Derive a thin image from `v041-shard-acts` and change only:

```python
return self.down_proj(_sac(jax.nn.silu(g) * u))
```

to:

```python
return self.down_proj(jax.nn.silu(g) * u)
```

All model math, params, batch/sequence shape, CE, attention, remat, scan, and XLA
runtime flags remain fixed.

**Predicted signal**: Support requires clean loss, a distinct HLO with lower
copy/fusion or MLP/down-projection layout pressure, and throughput above the
confirmed v153/v154 frontier band (57,754-57,757 tok/s average, 57,888 best
observed).

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO no-op,
or a clean run at/below the confirmed frontier without a clear profile/HLO win.

## Setup

GKE workload `alekseyv-qwen3-v169-mlpsac`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Image digest**: `sha256:21a390217fc38d05a8638d2ae681f00ece3d255d57ed88a0651e4054304f6854`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v169-no-mlp-intermediate-sac`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: confirmed v154 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

CPU equivalence was rerun against image
`<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
with `JAX_PLATFORMS=cpu python -u test_equivalence.py`.

Result: `==== EQUIVALENCE PASS ====`.

- Logits max delta: `2.682e-07`
- Loss delta: `0.000e+00`
- Gradients: all 25 HF parameters OK
- Max observed gradient delta: `7.823e-08`

## Results

Completed cleanly on both workers with normal loss and `EXIT_CODE=0`.

- Worker 0: 57,947 tok/s average, 7,243 tok/s/chip, 41.5% MFU.
- Worker 1: 57,919 tok/s average, 7,240 tok/s/chip, 41.5% MFU.
- Best observed post-warmup step: 58,051 tok/s.
- Final loss: 12.0461 on both workers.

This exceeds the confirmed v153/v154 frontier band (57,754-57,757 tok/s
average, 57,888 best observed) by about +0.28-0.33% on average. Treat as
provisional supported pending exact rerun before carrying.

## Profile

Profile captured successfully:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v169-no-mlp-intermediate-sac/plugins/profile/2026_06_04_19_35_11/`

Artifacts:

- `gke-tpu-46dd3e54-g30s.xplane.pb`
- `gke-tpu-46dd3e54-rkr2.xplane.pb`
- matching `trace.json.gz` files for both hosts

## HLO Dump

Fresh optimized train-step HLO was emitted:

- `module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 783,769 bytes
- SHA256: `443b5f34d11a802519fca1f96187c541566bdb9d532c86177249d46b11e3d644`
- Compiled memory: 26.86 GiB

Quick textual opcode counts:

| Opcode text | v153 | v169 |
|---|---:|---:|
| `all-gather` | 211 | 140 |
| `all-reduce` | 45 | 36 |
| `collective-permute` | 6 | 3 |
| `convolution` | 30 | 66 |
| `custom-call` | 36 | 41 |
| `copy` | 14 | 420 |
| `fusion` | 225 | 753 |

The count deltas are not sufficient on their own because v168 had a similar
high-copy/high-fusion shape and was slower. The wall-time signal is the key
difference: removing only the MLP intermediate `_sac` constraint keeps memory at
the frontier level while selecting faster steady steps.

## Verdict

Supported after v170 exact rerun. Removing the MLP-intermediate activation
sharding constraint is a clean, one-line, semantics-preserving layout change and
improves the best-worker average from the confirmed 57,754-57,757 tok/s band to
57,947 tok/s.

The v170 rerun replayed the same compiled program family and reached
57,935 tok/s best-worker average, confirming this as the new frontier stack.
