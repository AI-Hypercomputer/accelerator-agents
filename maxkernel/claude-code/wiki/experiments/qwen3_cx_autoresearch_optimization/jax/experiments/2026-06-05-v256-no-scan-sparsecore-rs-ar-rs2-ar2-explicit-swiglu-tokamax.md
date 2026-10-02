---
title: "Qwen3 JAX v256 - explicit SwiGLU with Tokamax Splash env"
type: experiment
hypothesis: "With the correct v230/v236 Tokamax Splash runtime restored, rewriting SiLU as its explicit x * sigmoid(x) definition may expose a better MLP fusion without changing model math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v255-explicit-swiglu-runtime-tokamax
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, silu, sparsecore, collectives, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v256 - explicit SwiGLU with Tokamax Splash env

## Hypothesis under test

**Hypothesis**: The v230/v236 frontier's largest bucket is fused MLP
matmul/all-gather/SwiGLU work with `jvp_jit_silu__` nested fusions. Replacing
`jax.nn.silu(g)` with the equivalent source expression `g * jax.nn.sigmoid(g)`
may expose a simpler HLO fusion or backward expression while preserving Qwen3
math.

**Mechanism**: Rerun the v255 image after correcting the invalid v255 launch.
The only source change is:

```python
return self.down_proj((g * jax.nn.sigmoid(g)) * u)
```

Keep the v230 RS2+AR2 runtime stack fixed, including the previously omitted
Tokamax Splash env:

```text
USE_TOKAMAX_SPLASH=1
TOKAMAX_MAX_LOGIT_CONST=30.0
```

**Predicted signal**: Support requires clean compile/loss and throughput above
the v230/v236 band of **59,749-59,750 tok/s / 42.9% MFU**, or a same-band
throughput with materially lower convolution-fusion/SwiGLU HLO cost.

**Falsification criterion**: Compile/runtime failure, invalid loss, unchanged
HLO, or clean completion below the v230/v236 frontier.

## Setup

GKE workload: `alekseyv-q3-v256-expl-swiglu`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v255-explicit-swiglu`
- **Image digest**: `sha256:ddeb45016ae9cb45926d097c35461533042c68beaf160ab08d19e65326049d3d`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v256-no-scan-sparsecore-rs-ar-rs2-ar2-explicit-swiglu-tokamax`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

The edit is the mathematical definition of SiLU (`x * sigmoid(x)`) written
explicitly. Cluster loss and HLO/profile comparison gate validity.

## Results

The corrected run completed cleanly on both workers with `EXIT_CODE=0`. The
Tokamax Splash path was active:

```text
[tokamax-splash] config: bq=2048 bkv=1024 bq_dkv=2048 bkv_dkv=2048 base2=True fuse_recip=True exp_sched=False
```

Throughput:

- Worker 0: **59,328 tok/s**, **7,416 tok/s/chip**, **42.5% MFU**.
- Worker 1: **59,301 tok/s**, **7,413 tok/s/chip**, **42.5% MFU**.
- Loss stayed finite and ended at **12.0458** on both workers.

This is valid, but below the v230/v236 frontier by about **421-449 tok/s**.
The explicit `g * sigmoid(g)` source form does not improve the current MLP
fusion surface.

## Profile

- XProf run:
  `2026-06-05-qwen3-jax-v256-no-scan-sparsecore-rs-ar-rs2-ar2-explicit-swiglu-tokamax/2026_06_05_12_52_05`
- Step time average: **4431.8 ms**
- MXU utilization: **67.1%**
- Peak HBM: **30.94 GiB**
- Op profile top buckets: convolution fusion **57,047.8 ms / 53.6%**; custom
  fusion **2,994.7 ms / 2.8%**; visible async all-gather done/start tails
  remain about **1.1% / 0.6%** for the largest all-gather group.

## HLO Dump

- Optimized train-step HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v256-no-scan-sparsecore-rs-ar-rs2-ar2-explicit-swiglu-tokamax/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Local copy:
  `/tmp/qwen3-v256-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `d1ded2a43caada7f7ba021d557240abf94a0da749cc6b09f44382a8d2dfd798c`
- Size: **19,518,444 bytes / 116,497 lines**
- Key text counts: `all-gather` 6,901; `all-reduce` 165;
  `reduce-scatter` 2,774; `async-start` 361; `async-done` 361;
  `convolution` 3,087; `custom-call` 3,536; `fusion` 30,655;
  `copy` 21,320; `tokamax` 2.

## Verdict

**Refuted.** The corrected v256 run preserved loss and used the intended
Tokamax Splash stack, but the explicit SwiGLU source rewrite regressed
throughput versus v230/v236. Keep the original `jax.nn.silu(g)` expression on
the native-JAX frontier.
