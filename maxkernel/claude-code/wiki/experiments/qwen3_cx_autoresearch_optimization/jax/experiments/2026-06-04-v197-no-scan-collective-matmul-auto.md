---
title: "Qwen3 JAX v197 - no scan, collective-matmul auto"
type: experiment
hypothesis: "The confirmed no-scan frontier may interact differently with XLA collective-matmul lowering than the scan frontier; restoring auto lowering tests whether the `none` pair is still required or whether no-scan can recover a faster collective layout."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-no-scan-collective-matmul-auto
status: invalid
tags: [qwen3-cc, jax, flax-nnx, scan, no-scan, remat, collective-matmul, collectives, xla, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v197 - no scan, collective-matmul auto

## Hypothesis under test

**Hypothesis**: The retrospective ranks code-level collective-matmul-none
investigation as a structural direction. v116/v117 made a large jump by setting
both XLA collective-matmul modes to `none`, and that pair has been carried
through v183. The no-scan unrolled graph changes HLO structure substantially;
restoring `auto` may either recover a faster collective-matmul lowering or
prove that the `none` pair remains required in the no-scan frontier.

**Mechanism**: Reuse the confirmed v183 no-scan runtime and v169 image, changing
only:

```text
--xla_tpu_all_gather_collective_matmul_mode=auto
--xla_tpu_reduce_scatter_collective_matmul_mode=auto
```

instead of `none` for both flags.

**Predicted signal**: Support requires clean loss and steady throughput above
the confirmed no-scan band: v182/v183 at 58,288-58,330 tok/s confirmed average
and 58,443 best observed. Even a regression is useful if it confirms the
collective-matmul-none dependency in the no-scan HLO family.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion below the no-scan frontier without a profile/HLO advantage.

## Setup

GKE workload `alekseyv-qwen3-v197-noscan-cmauto`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v197-no-scan-collective-matmul-auto`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack except collective-matmul modes restored to `auto`.
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
Ran out of memory in memory space hbm. Used 35.56G of 31.25G hbm.
Exceeded hbm capacity by 4.32G.
Operator: jit(train_step)/forward_backward/jvp()/dot_general
XLA label: fusion.14097 = fusion(... async-collective-done)
```

The workload ended with `EXIT_CODE=1` on 2026-06-04 23:45:50 UTC.

## Profile

No profile was produced because the train step never ran.

## HLO Dump

Train-step compilation failed before a usable optimized train-step HLO/profile
could be collected. Smaller helper HLO dumps were emitted, and the runtime logs
confirm the intended flags:

```text
--xla_tpu_all_gather_collective_matmul_mode=auto
--xla_tpu_reduce_scatter_collective_matmul_mode=auto
```

## Verdict

Invalid/refuted for the target shape. Restoring both collective-matmul modes to
`auto` under the no-scan frontier exceeds v6e-8 HBM by 4.32 GiB. The
collective-matmul `none` pair remains required for the current no-scan
batch-4/seq8192 stack. The next isolate should flip only one of the two modes
to identify whether all-gather or reduce-scatter collective-matmul lowering is
the dominant memory trigger.
