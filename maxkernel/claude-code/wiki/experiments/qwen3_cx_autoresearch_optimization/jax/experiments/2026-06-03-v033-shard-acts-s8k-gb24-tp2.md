---
title: "Qwen3 JAX v033 - shard acts s8k gb24 TP2"
type: experiment
hypothesis: "At the v041 global batch 24 target shape, TP2 may trade some communication for better vocab/model sharding and improve the MaxText-CE memory/perf balance."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, scan, remat, maxtext-ce, activation-sharding, tp2, long-seq, sharding, invalid, oom, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v033 - shard acts s8k gb24 TP2

## Hypothesis under test

**Hypothesis**: At the v041 global batch 24 target shape, TP2 may trade some
communication for better vocab/model sharding and improve the MaxText-CE
memory/perf balance.

**Mechanism**: Keep v041's total target-sequence work fixed at global batch 24,
seq8192, scan, remat, Splash, MaxText CE, and activation sharding. Change only
the mesh from `fsdp=8,tp=1` to `fsdp=4,tp=2` by setting
`--tp_parallelism=2 --batch_size=6` (global batch remains 24).

**Predicted signal**: If TP2 reduces vocab/logit/model-memory pressure or
improves matmul tiling enough to offset communication, throughput and MFU should
beat v041's 48,312 tok/s / 34.6% MFU. If TP collectives/layout overhead
dominates, it will regress or OOM.

**Falsification criterion**: Compile/runtime failure, or clean completion at or
below v041's 48,312 tok/s / 34.6% MFU.

## Setup

Planned GKE workload `alekseyv-qwen3-v033`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v033-shard-acts-s8k-gb24-tp2`
- **Mesh**: `fsdp=4,tp=2`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=6 --seqlen=8192 --tp_parallelism=2`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Invalid. Workload `alekseyv-qwen3-v033` failed during train-step compilation on
both workers with HBM OOM.

- **Mesh/shape confirmed**: `fsdp=4,tp=2`, global batch 24, seq8192
- **Exit**: `EXIT_CODE=1` at `Wed Jun 3 03:04:09 UTC 2026` on worker 0 and
  `Wed Jun 3 03:04:10 UTC 2026` on worker 1
- **OOM**: used 33.78 GiB of 31.25 GiB HBM; exceeded capacity by 2.53 GiB
- **Memory accounting**: total HBM usage at least 34.03 GiB, reserved 260.00
  MiB, program 28.05 GiB, arguments 5.72 GiB, HLO temp 28.04 GiB
- **Fragmentation**: HLO temp padded 28.04 GiB vs unpadded 14.29 GiB; 49.0%
  fragmentation / 13.75 GiB
- **Largest allocation**: 13.50 GiB `bf16[36,6,8192,4096]` stacked-layer
  broadcast in the scan path
- **Other large allocations**: 593.50 MiB `bf16[151936,2048]` all-gather and
  192.00 MiB `bf16[49152,2048]` fusion

## Profile

No runtime profile; the workload failed before executing training steps.

## HLO Dump

OOM artifacts were dumped under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v033-shard-acts-s8k-gb24-tp2/hlo/`

The prefix contains OOM dump artifacts totaling 17.63 MiB.

## Verdict

Invalid/refuted. TP2 does not improve the v041 same-global-batch target shape
under the scan layout. The failure is not the v026 full-logit tensor; it is a
large stacked-layer broadcast introduced by combining scan, per-FSDP batch 6,
and TP2. Do not pursue TP2 with scan at global batch 24 without first changing
the stacked-layer layout.
