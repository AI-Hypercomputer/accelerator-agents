---
title: "Qwen3 JAX v032 - shard acts s8k bs4 MaxText offload"
type: experiment
hypothesis: "Remat offload may provide enough HBM relief for the v026 MaxText-CE batch-4 shape to fit without tokamax CE's custom-call slowdown."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, scan, remat, offload, maxtext-ce, activation-sharding, tp1, long-seq, batch-scaling, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v032 - shard acts s8k bs4 MaxText offload

## Hypothesis under test

**Hypothesis**: Remat offload may provide enough HBM relief for the v026
MaxText-CE batch-4 shape to fit without tokamax CE's custom-call slowdown.

**Mechanism**: Retry the v026 shape (`fsdp=8,tp=1`, seq8192, per-chip batch 4
/ global batch 32, scan, remat, Splash, MaxText CE, activation sharding), adding
only `--offload_remat=True`. The offload policy saves/offloads named decoder
layer inputs and attention projections to pinned host memory while keeping the
fast MaxText CE path.

**Predicted signal**: v026 exceeded HBM by 2.29 GiB. If remat offload removes
enough activation pressure and host-transfer overhead is modest, the run should
fit and beat v041's 48,312 tok/s / 34.6% MFU. If the OOM is dominated by logits
or offload overhead, it will fail or regress.

**Falsification criterion**: Compile/runtime failure, or clean completion at or
below v041's 48,312 tok/s / 34.6% MFU.

## Setup

Planned GKE workload `alekseyv-qwen3-v032`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v032-shard-acts-s8k-bs4-maxtext-offload`
- **Mesh**: `fsdp=8,tp=1`
- **Command shape**: `--use_scan=True --use_remat=True --offload_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Refuted. Workload `alekseyv-qwen3-v032` completed cleanly and fit the v026
batch-4 MaxText-CE shape, but throughput and MFU stayed below v041.

- **Worker 0 summary**: 45,953 tok/s, 5,744 tok/s/chip, 32.9% MFU
- **Worker 1 summary**: 45,940 tok/s, 5,742 tok/s/chip, 32.9% MFU
- **v041 comparison**: 48,312 tok/s, 6,039 tok/s/chip, 34.6% MFU
- **Delta**: -4.9% throughput and -1.7 pp MFU
- **Offload confirmation**: startup printed
  `per-layer jax.checkpoint (save_and_offload(proj+mlpwi -> pinned_host)) ON`

## Profile

- **XProf run**: `2026-06-03-qwen3-jax-v032-shard-acts-s8k-bs4-maxtext-offload/2026_06_03_02_56_53`
- **MXU utilization**: 49.6% vs v041's 53.7%
- **Idle time**: 0.2% from op profile
- **Peak HBM**: 28.30 GiB of 31.25 GiB; 2.951 GiB free
- **Stack / heap**: 22.4943 GiB stack reservation, 5.8008 GiB heap allocation

Top HLO op self-time from XProf:

| Op family | Self-time |
|-----------|-----------|
| convolution fusion | 41.7% |
| custom-call | 25.6% |
| loop fusion | 17.4% |
| data formatting | 8.2% |
| collective-permute-start | 1.8% |
| non-fusion elementwise | 1.6% |
| broadcast | 1.1% |
| all-gather | 0.7% |
| reduce | 0.7% |

Offload covers the v026 HBM miss while keeping MaxText CE, but it shifts too
much time into data formatting and lowers MXU utilization. The attention
custom-call bucket itself is not worse than v041: forward residuals are 14.5%
and DKV is 11.1%, but the extra formatting/offload overhead dominates.

## HLO Dump

HLO artifacts were dumped under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v032-shard-acts-s8k-bs4-maxtext-offload/hlo/`

The prefix contains 32 objects totaling 31.53 MiB.

## Verdict

Refuted. Remat offload makes the MaxText-CE global batch 32 shape fit, but the
host/offload-induced data formatting cost leaves it slower than the v041
batch-3 frontier. Do not pursue MaxText batch scaling via remat offload on v6e-8.
