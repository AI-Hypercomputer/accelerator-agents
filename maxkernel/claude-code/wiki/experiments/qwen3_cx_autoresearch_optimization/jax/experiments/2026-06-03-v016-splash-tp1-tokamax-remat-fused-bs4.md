---
title: "Qwen3 JAX v016 - splash TP1 tokamax remat fused bs4"
type: experiment
hypothesis: "Dropping tensor parallelism from TP2 to TP1 removes TP communication and layout overhead while tokamax/remat/fused kernels keep global batch 32 fitting in HBM."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8
verdict: invalid
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax, remat, fused-rope, tp1, invalid, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v016 - splash TP1 tokamax remat fused bs4

## Hypothesis under test

**Hypothesis**: Dropping tensor parallelism from TP2 to TP1 removes TP communication and layout overhead while tokamax/remat/fused kernels keep global batch 32 fitting in HBM.

**Mechanism**: Reuse the v015 image, but run `--tp_parallelism=1` and `--batch_size=4`. The mesh becomes `fsdp=8,tp=1`, removing all TP all-gather/all-to-all paths while preserving the same Splash/tokamax/remat/fused code path.

**Predicted signal**: If TP communication was the limiter, XProf should show lower all-gather/all-to-all/data-formatting share than v015 and throughput should exceed the v015 frontier of 47,846 tok/s / 30.8% MFU without increasing peak HBM beyond the 31.25 GiB device limit.

**Falsification criterion**: Compile/runtime failure, HBM OOM, or a clean run with lower throughput/MFU than v015 despite lower communication share.

## Setup

GKE workload `alekseyv-qwen3-v016`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v016-splash-tp1-tokamax-remat-fused-bs4`
- **Mesh**: `fsdp=8,tp=1`
- **Command shape**: `--batch_size=4 --seqlen=2048 --tp_parallelism=1 --use_splash=True`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

No training step completed. Worker 1 exited with `EXIT_CODE=1`; the JobSet failed.

```text
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error.
Ran out of memory in memory space hbm. Used 31.96G of 31.25G hbm.
Exceeded hbm capacity by 736.97M.
```

The failing operator reported in the runtime log:

```text
jit(train_step)/forward_backward/transpose(jvp(forward_backward))/jvp()/
checkpoint/checkpoint/rematted_computation/mul
```

## Profile

No XProf trace was captured because the program failed during compile before the profiled window.

## HLO Dump

HLO dump path:

```text
gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v016-splash-tp1-tokamax-remat-fused-bs4/hlo/
```

Artifact count: 24 GCS objects, 686.07 MiB total.

The compile produced OOM debug artifacts:

```text
module_0089.jit_train_step.cl_854318611.oom_intermediate_module.txt
module_0089.jit_train_step.cl_854318611.oom_top_memory_consuming_instructions
```

OOM memory summary:

```text
Program hbm requirement 26.24G:
  HLO temp 26.23G
  HLO temp fragmentation 56.6% (14.84G)

Largest allocation:
  2.32G bf16[8192,151936] custom-call AllocateBuffer

Repeated large temps:
  192.00M bf16[4,2048,12288] rematted_computation/mul
```

## Verdict

`invalid`.

TP1 removes TP communication, but this shape does not fit. The failure is informative: unlike the earlier CE/log-softmax OOMs, this one is dominated by HLO temp fragmentation and rematerialized hidden-state-sized temps after removing TP sharding. The TP2 v015 configuration remains the frontier; retrying TP1 needs a lower batch or a memory-layout/remat change before it is worth profiling.
