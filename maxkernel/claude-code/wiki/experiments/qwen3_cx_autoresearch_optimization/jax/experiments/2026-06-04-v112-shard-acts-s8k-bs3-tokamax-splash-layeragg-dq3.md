---
title: "Qwen3 JAX v112 - shard acts s8k bs3 tokamax splash layeragg dq3"
type: experiment
hypothesis: "Tokamax DQ reduction may compose with the layeragg scheduler stack and reduce valid attention backward time."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, dq-reduction, attention-backward, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v112 - shard acts s8k bs3 tokamax splash layeragg dq3

## Hypothesis under test

**Hypothesis**: v049/v054 showed `TOKAMAX_DQ_REDUCTION_STEPS=3` reduces the
attention backward custom-call bucket but loses wall-clock on the older
frontiers. The current v090 stack adds selective resources, custom-call scoped
VMEM adjustment, layer scheduler, and sparse-core collective aggregation. That
scheduler stack may recover the wall-clock loss while retaining the DQ/DKV
custom-call reduction.

**Mechanism**: Keep the valid v090 qseq1 frontier fixed and add only
`TOKAMAX_DQ_REDUCTION_STEPS=3` at runtime.

**Predicted signal**: Loss stays finite and throughput exceeds the v090
51,033 tok/s best-observed / 51,026+ repeated frontier, or the profile shows a
custom-call/DKV reduction large enough to justify a follow-up despite a small
in-band throughput result.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the v090 throughput band without a net profile win.

## Setup

Planned GKE workload `alekseyv-qwen3-v112-dq3`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v112-shard-acts-s8k-bs3-tokamax-splash-layeragg-dq3`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0 TOKAMAX_DQ_REDUCTION_STEPS=3`
- **XLA flags**: v090 scheduler frontier unchanged.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v112-dq3` completed cleanly on
`<your-cluster>`.

- Worker 0: 50,853 tok/s average, 36.5% MFU.
- Worker 1: 50,825 tok/s average, 36.4% MFU.
- Final loss: 12.0508 on both workers.
- XProf run:
  `2026-06-04-qwen3-jax-v112-shard-acts-s8k-bs3-tokamax-splash-layeragg-dq3/2026_06_04_09_56_36`.

This is valid but below both the v090 best-observed 51,033 tok/s and the
51,026+ repeated v089/v090 frontier band.

## Profile

XProf shows a real but unfavorable tradeoff.

- Total profiled time: 93,096.6 ms.
- MXU utilization: 56.6%.
- Peak HBM: 28.32 GiB, unchanged from the frontier.
- Custom-call: 21,653.4 ms / 23.3%.
- DKV custom-call: 11,235.8 ms / 12.1%.
- Forward residual custom-calls: 5,214.3 ms / 5.6% and 5,203.3 ms / 5.6%.
- Convolution fusion: 45,693.8 ms / 49.1%.
- Loop fusion: 16,833.3 ms / 18.1%.
- Data formatting: 3,734.7 ms / 4.0%.

`TOKAMAX_DQ_REDUCTION_STEPS=3` reduced the DKV custom-call bucket relative to
v110, but increased convolution-fusion time enough to lose wall-clock.

## HLO Dump

Train-step HLO artifacts were written under the run's `hlo/` prefix.

- Optimized train-step text:
  `module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 1,394,534 bytes.
- HLO object count under the run prefix: 32.
- Content hash: `ebcd22220b014d8e255b05311311e539b3dd9923ffbdea299732c26249d978dd`.
- XProf train-step program hash: `12596932080449751219`.

This is a real HLO/schedule change versus the normal layeragg frontier
family, not a no-op runtime flag.

## Verdict

Refuted. The DQ3 path remains correct but does not compose profitably with the
v090 layer-scheduler/sparse-core-aggregator frontier. Do not carry
`TOKAMAX_DQ_REDUCTION_STEPS=3` forward for the qseq1 frontier.
