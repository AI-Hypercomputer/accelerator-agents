---
title: "Qwen3 JAX v198 - no scan, all-gather collective-matmul auto"
type: experiment
hypothesis: "v197 showed that restoring both collective-matmul modes to auto exceeds HBM; flipping only all-gather to auto isolates whether all-gather collective-matmul lowering is the memory trigger under no-scan."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-no-scan-ag-collective-matmul-auto
status: invalid
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, collective-matmul, all-gather, collectives, xla, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v198 - no scan, all-gather collective-matmul auto

## Hypothesis under test

**Hypothesis**: v197 failed with both collective-matmul modes restored to
`auto`, using 35.56 GiB HBM. This isolate tests whether the all-gather
collective-matmul lowering alone is enough to trigger the memory blowup, while
keeping reduce-scatter lowering at the frontier value `none`.

**Mechanism**: Reuse the confirmed v183 no-scan runtime and v169 image,
changing only:

```text
--xla_tpu_all_gather_collective_matmul_mode=auto
--xla_tpu_reduce_scatter_collective_matmul_mode=none
```

**Predicted signal**: If compile OOM recurs, all-gather collective-matmul
lowering is the dominant HBM trigger. If it compiles, compare throughput,
memory, and HLO against v183 to decide whether this one-sided lowering is useful
or merely slower.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion below the no-scan frontier without a profile/HLO advantage.

## Setup

GKE workload `alekseyv-qwen3-v198-noscan-agauto`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v198-no-scan-ag-collective-matmul-auto`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack except all-gather collective-matmul mode restored to `auto`.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code or math change. This uses the already-equivalent v169 image and
changes only XLA runtime lowering flags.

## Results

Invalid. The job failed before step 0 with a TPU compile-time HBM OOM.

Error signature:

```text
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error.
Ran out of memory in memory space hbm. Used 32.67G of 31.25G hbm.
Exceeded hbm capacity by 1.42G.
Operator: jit(train_step)/forward_backward/jvp()/dot_general
XLA label: fusion.9455.remat = fusion(async-collective-done, ...)
XLA label: fusion = fusion(all-gather.6169.remat, copy-done.2039)
```

Both pods ended with `EXIT_CODE=1` on 2026-06-04 23:55:53 UTC.

## Profile

No profile was produced because the train step never ran.

## HLO Dump

Optimized train-step HLO was emitted before the compile failure:

- File: `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 28,098,167 bytes
- SHA256:
  `52ec4c05240fefa69a03e817b4050015f57edb0ef24df9a9a6020df151ab418f`
- Memory report: 35,092,736,456 bytes / 32.68 GiB.
- Largest allocation: 26.93 GiB preallocated temp.
- Text counts: all-gather 5,544, all-reduce 1,148, collective-permute 9,221,
  convolution 3,956, custom-call 2,285, copy 30,383, fusion 36,308.

Compared with the valid v183 frontier, all-gather collective-matmul `auto`
lowers the all-gather text count but explodes collective-permute/copy/fusion
pressure and exceeds HBM.

## Verdict

Invalid/refuted. All-gather collective-matmul `auto` alone is enough to exceed
v6e HBM under the no-scan target shape. Keep
`--xla_tpu_all_gather_collective_matmul_mode=none`. The remaining isolate is
reduce-scatter `auto` with all-gather kept at `none`.
