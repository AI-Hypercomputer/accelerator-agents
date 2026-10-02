---
title: "Qwen3 JAX v253 - fused MLP gate/up VMEM98304"
type: experiment
hypothesis: "v252's fused MLP gate/up projection failed only at program load due runtime HBM headroom; lowering scoped VMEM may let the same HLO family run."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v252-fused-mlp-gate-up
status: invalid
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, mlp, swiglu, fusion, layout, vmem, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v253 - fused MLP gate/up VMEM98304

## Hypothesis under test

**Hypothesis**: v252 compiled the combined MLP gate/up projection but failed to
load at VMEM 100352:

```text
Attempting to allocate 1.82G. That was not possible. There are 1.54G free.
```

The compiled memory report was **28.57 GiB**, so the experiment may be a runtime
headroom failure rather than a fundamental compile impossibility. Lowering
`--xla_tpu_scoped_vmem_limit_kib` from 100352 to **98304** may free enough
custom-call scratch for the same fused-MLP image to load.

**Mechanism**: Reuse the v252 image unchanged and alter only scoped VMEM.

- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v252-fused-mlp-gate-up`
- Digest:
  `sha256:a25d23bef428af724e29b708ecc6f1c547337a1354f2298535ffcffbb50466fb`
- Runtime delta from v252: `--xla_tpu_scoped_vmem_limit_kib=98304`
- Keep MaxText CE, no-scan, remat, Tokamax Splash, activation sharding, and the
  rest of the v230 RS2+AR2 runtime stack fixed.

**Predicted signal**: Support requires clean program load and throughput above
the v230/v236 frontier (**59,749-59,750 tok/s / 42.9% MFU**). A clean run below
frontier closes the fused gate/up direction; another load-time OOM closes it
without further memory sweeps.

## Setup

- Workload: `alekseyv-q3-v253-fusedmlp-vmem98`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v253-no-scan-sparsecore-rs-ar-rs2-ar2-fused-mlp-gate-up-vmem98304`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Result

Both workers failed while loading the compiled `jit_train_step` program, before
step 0 completed:

```text
RESOURCE_EXHAUSTED: Error loading program 'jit_train_step':
Attempting to allocate 1.81G. That was not possible. There are 1.54G free.
```

This reproduces v252's load-time failure despite lowering scoped VMEM from
100352 to 98304. No throughput/MFU result exists.

## Profile

No XProf trace was captured because the run failed before the profile window.

## HLO Dump

Train-step HLO was emitted:

- GCS:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v253-no-scan-sparsecore-rs-ar-rs2-ar2-fused-mlp-gate-up-vmem98304/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Local analysis copy:
  `/tmp/qwen3-v253-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `61dd6f8b385914d01a62d398a8d1e9378091ecf6a856b5c3d7de490422a9eb5d`
- Size: **17,406,954 bytes**
- Lines: **103,597**

Compiled memory report:

- Total bytes used: **28.56 GiB**

Lowering scoped VMEM changed the optimized HLO hash and slightly reduced text
size, but did not materially reduce compiled/live memory versus v252
(28.57 GiB).

## Verdict

**Invalid/refuted.** The fused MLP gate/up projection remains unable to load on
v6e-8 even with lower scoped VMEM. Close this direction: the runtime headroom
problem is not fixed by the bounded VMEM relief probe, and further VMEM sweeps
would be a low-signal search around a structurally memory-heavier HLO.
