---
title: "Qwen3 JAX v252 - fused MLP gate/up projection"
type: experiment
hypothesis: "Combining the MLP gate and up input projections into one dot may reduce repeated projection overhead while preserving Qwen3 SwiGLU math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v252-fused-mlp-gate-up
status: invalid
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, mlp, swiglu, fusion, layout, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v252 - fused MLP gate/up projection

## Hypothesis under test

**Hypothesis**: The current frontier's top device-time surface is still dominated
by repeated MLP/SwiGLU convolution-fusion work. Qwen3's MLP computes two input
projections from the same activation:

```python
gate = gate_proj(x)
up = up_proj(x)
out = down_proj(silu(gate) * up)
```

The two projection weights have identical sharding and input shape. Concatenating
the existing `gate_proj.weight` and `up_proj.weight` at runtime and issuing one
larger `dot_general`, then splitting the result, may reduce launch/scheduling
and repeated input-read overhead without changing parameter values or SwiGLU
math.

**Mechanism**: Patch only `Qwen3MLP.__call__` in a thin image derived from
`qwen3-8b-jax:v169-no-mlp-sac`.

- Existing parameter leaves remain `gate_proj.weight` and `up_proj.weight`.
- The model computes `w = concat([gate_w, up_w], axis=0)`, runs one
  `dot_general(x, w)`, then splits the last dimension into `gate` and `up`.
- `checkpoint_name(..., "mlpwi")` remains on the split gate/up outputs, matching
  the carried no-MLP-intermediate-`_sac` frontier.
- Keep MaxText CE, no-scan, remat, Tokamax Splash, activation sharding, and the
  v230 RS2+AR2 runtime stack fixed.

**No-fallback assumption**: There is no alternate MLP path in the image. If the
combined projection shape compiles poorly or introduces copy overhead, the run
will refute directly.

**Predicted signal**: Support requires a clean finite-loss run above the current
v230/v236 frontier band (**59,749-59,750 tok/s / 42.9% MFU**) or a clear XProf
reduction in MLP convolution-fusion time at in-band throughput.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed
semantics signal, or clean completion below the frontier.

## Setup

- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v252-fused-mlp-gate-up`
- Digest:
  `sha256:a25d23bef428af724e29b708ecc6f1c547337a1354f2298535ffcffbb50466fb`
- Thin image patch: `Qwen3MLP.__call__` only, extracted from base image
  `qwen3-8b-jax:v169-no-mlp-sac`.
- Syntax check: `python -m py_compile /app/trainer/model/modeling_qwen3.py`
  passed in the base image.
- Workload: pending launch as `alekseyv-q3-v252-fusedmlp` on
  `<your-cluster>`.
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v252-no-scan-sparsecore-rs-ar-rs2-ar2-fused-mlp-gate-up`
- `XLA_FLAGS` uses a concrete GCS HLO dump path.
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Result

Both workers failed while loading the compiled `jit_train_step` program, before
step 0 completed:

```text
RESOURCE_EXHAUSTED: Error loading program 'jit_train_step':
Attempting to allocate 1.82G. That was not possible. There are 1.54G free.
```

Worker0 reported the error on `0x1x0_HBM0`; worker1 reported the same allocation
failure on partition 4. The JobSet failure policy then marked the workload
failed. No throughput/MFU result exists.

## Profile

No XProf trace was captured because the run failed before the profile window.

## HLO Dump

Train-step HLO was emitted despite the load-time failure:

- GCS:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v252-no-scan-sparsecore-rs-ar-rs2-ar2-fused-mlp-gate-up/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Local analysis copy:
  `/tmp/qwen3-v252-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `efe69c0f710972418836d958ed55c446c61a43691461c9fa8e85bf5a532e299b`
- Size: **17,454,915 bytes**
- Lines: **103,987**

Compiled memory report:

- Total bytes used: **28.57 GiB**

This is materially higher than the v230/v236 frontier family's compiled memory
and leaves too little runtime allocation headroom at VMEM 100352.

## Verdict

**Invalid/refuted at load time.** The combined gate/up projection compiles, but
the concatenated projection raises compiled/live memory enough that the program
cannot load on v6e-8 with the carried VMEM 100352 runtime. The intended compute
benefit cannot be measured at this setting.

A single bounded follow-up is warranted: rerun the same image with a lower
scoped VMEM limit to see whether freeing custom-call scratch lets the fused MLP
program load. If that runs below frontier, close this direction.
