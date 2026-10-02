---
title: "Qwen3 JAX v067 - shard acts s8k bs3 tokamax splash no ag backward pipeline"
type: experiment
hypothesis: "Disabling AG backward pipelining may reduce valid-frontier scheduling overhead without hurting compute overlap."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v067 - shard acts s8k bs3 tokamax splash no ag backward pipeline

## Hypothesis under test

**Hypothesis**: The valid v056 frontier carries
`--xla_tpu_enable_ag_backward_pipelining=true` as part of the scheduler bundle.
On this qseq1/FSDP8 shape, that flag may add data-formatting or schedule
pressure without contributing enough overlap. Disabling it may improve wall
clock or confirm the bundle is needed.

**Mechanism**: Keep v056 fixed and change only
`--xla_tpu_enable_ag_backward_pipelining=true` to `false`.

**Predicted signal**: Loss stays near the v056 trajectory, ending around 12.05,
and throughput exceeds v056's 50,383 tok/s / 36.1% MFU or reduces data
formatting/collective overhead without increasing custom-call time.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v056's 50,383 tok/s / 36.1% MFU.

## Setup

GKE workload `alekseyv-qwen3-v067`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v067-shard-acts-s8k-bs3-tokamax-splash-no-ag-bwd-pipeline`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: `--xla_tpu_enable_ag_backward_pipelining=false`
  instead of v056's `true`, keeping scheduler rerun3.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run but below v056:

- Worker 0: average 50,367 tok/s total, 6,296 tok/s/chip, 36.1% MFU.
- Worker 1: average 50,343 tok/s total, 6,293 tok/s/chip, 36.1% MFU.
- Step time stabilized around 3.90 s after warmup.
- Loss trajectory stayed normal, ending at 12.0508.

Compared with v056's 50,383 tok/s / 36.1% MFU, disabling AG backward
pipelining is a small raw-throughput regression.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v067-shard-acts-s8k-bs3-tokamax-splash-no-ag-bwd-pipeline/2026_06_03_21_40_57`
- **MXU utilization**: 55.9%.
- **Peak HBM**: 28.33 GiB, 2.914 GiB free.
- **Device idle**: 0.3%.

Top device buckets:

- Convolution fusion: 48.1%.
- Custom-call: 23.4%.
- Loop fusion: 18.6%.
- Data formatting: 4.0%.

Custom-call is unchanged from v056: DKV remains 12.3% and forward residuals
remain about 11.1%. The collective-permute-start share is 1.8%, but that does
not translate into a wall-clock gain.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v067-shard-acts-s8k-bs3-tokamax-splash-no-ag-bwd-pipeline/hlo/`
- **Size**: 28.69 MiB
- **Object count**: 1179

## Verdict

**Refuted.** `--xla_tpu_enable_ag_backward_pipelining=false` is valid but does
not beat the v056 frontier or move the dominant custom-call buckets. Keep AG
backward pipelining enabled in the current scheduler bundle.
