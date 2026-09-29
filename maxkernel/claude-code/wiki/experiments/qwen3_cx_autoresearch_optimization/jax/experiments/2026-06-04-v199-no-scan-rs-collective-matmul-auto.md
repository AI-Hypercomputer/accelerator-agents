---
title: "Qwen3 JAX v199 - no scan, reduce-scatter collective-matmul auto"
type: experiment
hypothesis: "v198 proved all-gather collective-matmul auto alone exceeds HBM; flipping only reduce-scatter to auto tests whether reduce-scatter collective-matmul lowering is independently invalid or possibly usable under no-scan."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-no-scan-rs-collective-matmul-auto
status: invalid
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, collective-matmul, reduce-scatter, collectives, xla, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v199 - no scan, reduce-scatter collective-matmul auto

## Hypothesis under test

**Hypothesis**: v197 failed with both collective-matmul modes restored to
`auto`, and v198 showed all-gather `auto` alone is enough to exceed HBM. This
remaining isolate tests whether reduce-scatter `auto` is independently invalid
or whether it can compile and produce a useful schedule while all-gather remains
at the frontier value `none`.

**Mechanism**: Reuse the confirmed v183 no-scan runtime and v169 image,
changing only:

```text
--xla_tpu_all_gather_collective_matmul_mode=none
--xla_tpu_reduce_scatter_collective_matmul_mode=auto
```

**Predicted signal**: If compile OOM recurs, both collective-matmul modes must
remain `none`. If it compiles, compare throughput, memory, and HLO against v183
to decide whether reduce-scatter `auto` is useful or merely slower.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion below the no-scan frontier without a profile/HLO advantage.

## Setup

GKE workload `alekseyv-qwen3-v199-noscan-rsauto`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v199-no-scan-rs-collective-matmul-auto`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack except reduce-scatter collective-matmul mode restored to `auto`.
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
Ran out of memory in memory space hbm. Used 35.87G of 31.25G hbm.
Exceeded hbm capacity by 4.62G.
Operator: jit(train_step)/forward_backward/jvp()/dot_general
XLA label: fusion.10969 = fusion(fusion.7165, async-collective-done.337)
```

Both pods ended with `EXIT_CODE=1` on 2026-06-05 00:06:41 UTC.

## Profile

No profile was produced because the train step never ran.

## HLO Dump

Optimized train-step HLO was emitted before the compile failure:

- File: `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 26,420,592 bytes
- SHA256:
  `66a1b7df4c332ae916784705aa6e902bde041c4204c2fddc24a3cd987a856ce9`
- Memory report: 38,524,424,144 bytes / 35.88 GiB.
- Largest allocation: 30.12 GiB preallocated temp.
- Text counts: all-gather 5,664, all-reduce 153, collective-permute 7,596,
  convolution 3,676, custom-call 3,381, copy 34,627, fusion 33,024.

The reduce-scatter `auto` HLO also exceeds HBM and substantially increases
copy/collective-permute pressure versus the valid no-scan frontier.

## Verdict

Invalid/refuted. Reduce-scatter collective-matmul `auto` alone exceeds HBM
under the no-scan target shape. Together with v197/v198, this closes the
collective-matmul isolate sweep: both all-gather and reduce-scatter
collective-matmul modes must remain `none` for the current no-scan
batch-4/seq8192 frontier.
