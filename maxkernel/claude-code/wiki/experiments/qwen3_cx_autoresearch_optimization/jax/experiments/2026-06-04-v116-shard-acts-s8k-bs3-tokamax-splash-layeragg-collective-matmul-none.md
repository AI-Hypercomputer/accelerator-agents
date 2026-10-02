---
title: "Qwen3 JAX v116 - shard acts s8k bs3 tokamax splash layeragg collective matmul none"
type: experiment
hypothesis: "Disabling TPU collective-matmul rewrites for all-gather and reduce-scatter may improve the v090 layeragg frontier by avoiding a suboptimal auto-selected collective-matmul schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, all-gather, reduce-scatter, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v116 - shard acts s8k bs3 tokamax splash layeragg collective matmul none

## Hypothesis under test

**Hypothesis**: The v090 frontier flagfile leaves
`--xla_tpu_all_gather_collective_matmul_mode=auto` and
`--xla_tpu_reduce_scatter_collective_matmul_mode=auto`. The project-wide
libtpu flag matrix lists the MaxText `DISABLE_COLLECTIVE_MATMUL` bundle as
forcing both modes to `none`. On this qseq1 layeragg frontier, `auto` may be
selecting a collective-matmul path that is neutral or slightly harmful around
the FSDP all-gather/reduce-scatter regions.

**Mechanism**: Keep the valid v090 frontier fixed and add only:

```text
--xla_tpu_all_gather_collective_matmul_mode=none
--xla_tpu_reduce_scatter_collective_matmul_mode=none
```

**Predicted signal**: Loss stays finite and throughput exceeds the v090
51,033 tok/s best-observed / 51,026+ repeated frontier, or XProf shows lower
train-step wall time with no custom-call/convolution-fusion regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v090 throughput band without a profile win.

## Setup

GKE workload `alekseyv-qwen3-v116-cmnone`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v116-shard-acts-s8k-bs3-tokamax-splash-layeragg-collective-matmul-none`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v090 scheduler frontier plus collective-matmul modes set to `none`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with finite loss and a large throughput
frontier shift.

- **Workload**: `alekseyv-qwen3-v116-cmnone`
- **Worker 0**: avg 56,009 tok/s (7,001 tok/s/chip), approx 40.2% MFU; best step 56,172 tok/s
- **Worker 1**: avg 56,009 tok/s (7,001 tok/s/chip), approx 40.2% MFU; best step 56,171 tok/s
- **Final loss**: 12.0512 on both workers
- **Profile window**: steps 12-14
- **XProf run**: `2026-06-04-qwen3-jax-v116-shard-acts-s8k-bs3-tokamax-splash-layeragg-collective-matmul-none/2026_06_04_10_52_19`

## Profile

XProf confirms the speedup is device-side, not only a host logging artifact.

- **Total time**: 84,576.4 ms
- **Idle time**: 278.3 ms (0.3%)
- **Compute time**: 84,298.1 ms
- **MXU utilization**: 63.1%
- **Peak HBM**: 22.21 GiB
- **Top op buckets**:
  - convolution fusion: 44,281.0 ms (52.4%)
  - custom-call: 22,060.9 ms (26.1%)
  - DKV custom-call: 11,650.0 ms (13.8%)
  - forward residuals: 5,209.0 ms + 5,201.9 ms (12.4% combined)
  - loop fusion: 8,515.8 ms (10.1%)
  - all-reduce-scatter fusion: 3,756.9 ms (4.4%)

Compared with v090/v114-style frontier profiles, total train-step device time
drops from about 92.8 s to 84.6 s, MXU rises from about 56.8% to 63.1%, and
peak HBM drops from about 28.32 GiB to 22.21 GiB. The main profile change is a
large loop-fusion reduction, with a smaller explicit all-reduce-scatter bucket
appearing in exchange.

## HLO Dump

Train-step optimized HLO was regenerated under the run prefix:

- **Train-step HLO**: `hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **Size**: 795,784 bytes
- **Object count under HLO prefix**: 1,289
- **Approx instruction-line count**: 3,848
- **Content hash**: `ff53351f8c3f828d212665c2505f4f865079a5f42232a66c05bf345a35044a8b`
- **Train-step program hash**: `16402808057163658277`
- **Flagfile confirmation**:
  - `--xla_tpu_all_gather_collective_matmul_mode=NONE`
  - `--xla_tpu_reduce_scatter_collective_matmul_mode=NONE`

## Verdict

**Supported / provisional frontier.** Forcing the all-gather and reduce-scatter
collective-matmul modes to `none` is valid and produces the first large qseq1
frontier move after v090: 56,009 avg tok/s / 40.2% MFU versus the prior 51,033
best-observed / 36.6% MFU frontier. Adopt the `DISABLE_COLLECTIVE_MATMUL` pair
provisionally and immediately rerun the exact configuration before banking it
as durable.
