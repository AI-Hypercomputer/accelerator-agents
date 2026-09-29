---
title: "Qwen3 JAX v255 - explicit SwiGLU sigmoid form"
type: experiment
hypothesis: "The hot MLP fusions in the v230/v236 frontier contain jvp_jit_silu__ subfusions; rewriting SiLU as its explicit x * sigmoid(x) definition may expose a simpler fusion pattern without changing model math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v255-explicit-swiglu
status: invalid
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, silu, sparsecore, collectives, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v255 - explicit SwiGLU sigmoid form

## Hypothesis under test

**Hypothesis**: XProf on the v230/v236 frontier shows the largest train-step
bucket is fused MLP matmul/all-gather/SwiGLU work, and the hot nested fusions
include `jvp_jit_silu__` nodes. Replacing `jax.nn.silu(g)` with the equivalent
source expression `g * jax.nn.sigmoid(g)` may expose a simpler HLO fusion or
backward expression while preserving Qwen3 math.

**Mechanism**: Derive a thin image from
`qwen3-8b-jax:v169-no-mlp-sac` and change only `Qwen3MLP.__call__`:

```python
return self.down_proj((g * jax.nn.sigmoid(g)) * u)
```

Keep the v230 RS2+AR2 runtime stack fixed: no-scan, remat, Tokamax Splash,
MaxText CE, activation sharding, no MLP intermediate `_sac`, scoped VMEM
100352 KiB, AG on async collective fusion, SparseCore RS/AR offload with
aggregator, RS latency multiplier 2, AR latency multiplier 2.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v230/v236 band of **59,749-59,750 tok/s / 42.9% MFU**, or a same-band
throughput with materially lower convolution-fusion/SwiGLU HLO cost.

**Falsification criterion**: Compile/runtime failure, invalid loss, unchanged
HLO, or clean completion below the v230/v236 frontier.

## Setup

GKE workload: `alekseyv-q3-v255-expl-swiglu`.

- **Base image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v255-explicit-swiglu`
- **Image digest**: `sha256:ddeb45016ae9cb45926d097c35461533042c68beaf160ab08d19e65326049d3d`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v255-no-scan-sparsecore-rs-ar-rs2-ar2-explicit-swiglu`
- **Mesh**: `fsdp=8,tp=1`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

The edit is the mathematical definition of SiLU (`x * sigmoid(x)`) written
explicitly. A cluster loss check still gates validity because HLO lowering and
bf16 execution may choose a different expression schedule.

## Results

Invalid launch. The workload completed cleanly, but the XPK launch omitted the
frontier-required `USE_TOKAMAX_SPLASH=1` / `TOKAMAX_MAX_LOGIT_CONST=30.0`
environment variables. The optimized HLO confirms the wrong attention path:
`tokamax: 0`.

Observed throughput on the invalid run:

- Worker 0: **56,347 tok/s**, **7,043 tok/s/chip**, **40.4% MFU**.
- Worker 1: **56,414 tok/s**, **7,052 tok/s/chip**, **40.4% MFU**.
- Loss stayed finite and ended at **12.0460** on both workers.

Because the runtime stack did not match v230/v236, this does not test the
explicit-SwiGLU hypothesis. Rerun as v256 with the same image and Tokamax
Splash env restored.

## Profile

- XProf run:
  `2026-06-05-qwen3-jax-v255-no-scan-sparsecore-rs-ar-rs2-ar2-explicit-swiglu/2026_06_05_12_38_52`
- Step time average: **4659.5 ms**
- MXU utilization: **63.8%**
- Peak HBM: **30.87 GiB**
- Op profile: convolution fusion **56,877.5 ms / 50.8%**; custom fusion
  **2,956.5 ms / 2.6%**; visible async-done all-gather tail about **1.0%**.

## HLO Dump

- Optimized train-step HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v255-no-scan-sparsecore-rs-ar-rs2-ar2-explicit-swiglu/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Local copy:
  `/tmp/qwen3-v255-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `caa50cd355c30a2748ba62d2507741cda4c3aab750081bef1b9dcc5980d99186`
- Size: **18,956,701 bytes / 113,580 lines**
- Key text counts: `all-gather` 6,927; `all-reduce` 165;
  `reduce-scatter` 2,774; `async-start` 364; `async-done` 364;
  `convolution` 3,071; `custom-call` 3,539; `fusion` 30,150;
  `copy` 18,204; `tokamax` 0.

## Verdict

**Invalid.** The missing Tokamax Splash env changed the tested runtime/HLO, so
v255 is not evidence for or against the explicit-SwiGLU source rewrite. Do not
compare this throughput to v230/v236. Rerun as v256 with the correct env.
