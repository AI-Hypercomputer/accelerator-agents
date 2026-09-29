---
title: "Qwen3 JAX v041 - shard acts s8k bs3"
type: experiment
hypothesis: "Activation sharding with scan/remat and MaxText CE can fit and accelerate target seq8192 by avoiding the TP2 tokamax HBM wall."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported
tags: [qwen3-cc, jax, flax-nnx, splash, scan, remat, maxtext-ce, activation-sharding, tp1, long-seq, supported, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v041 - shard acts s8k bs3

## Hypothesis under test

**Hypothesis**: Activation sharding with scan/remat and MaxText CE can fit and
accelerate target seq8192 by avoiding the TP2 tokamax HBM wall.

**Mechanism**: Use the `v041-shard-acts` image with `--use_scan=True`,
`--use_remat=True`, `--use_maxtext_ce=True`, `--shard_acts=True`, Splash
attention, TP1/FSDP8, seq8192, and per-chip batch 3 for global batch 24.

**Predicted signal**: A clean target-sequence run with higher MFU than v023
and enough HBM margin to probe batch 4.

**Falsification criterion**: Compile/runtime OOM, or supported completion below
v023's 26.6% MFU.

## Setup

Adopted completed GKE workload `alekseyv-qwen3-cc-jax-v041-shardact`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v041-shard-acts-s8k-bs3`
- **Mesh**: `fsdp=8,tp=1`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Supported. Both workers completed 20 steps cleanly.

- **Worker 0 summary**: 48,312 tok/s, 6,039 tok/s/chip, 34.6% MFU
- **Worker 1 summary**: 48,292 tok/s, 6,036 tok/s/chip, 34.6% MFU
- **Steady step time**: about 4.07 s after compile/warmup
- **Prior target-sequence frontier**: v023 at 37,148 tok/s and 26.6% MFU
- **Prior overall frontier**: v015 at 47,846 tok/s and 30.8% MFU

## Profile

- **XProf run**: `2026-06-03-qwen3-jax-v041-shard-acts-s8k-bs3/2026_06_03_00_28_22`
- **MXU utilization**: 53.7%
- **Idle time**: 0.3% from op profile
- **Peak HBM**: 28.33 GiB of 31.25 GiB
- **Free HBM at peak**: 2.916 GiB
- **Stack / heap**: 22.51 GiB stack reservation, 5.82 GiB heap allocation

Top HLO op self-time from XProf:

| Op family | Self-time |
|-----------|-----------|
| convolution fusion | 46.1% |
| custom-call | 26.7% |
| loop fusion | 17.9% |
| data formatting | 3.8% |
| collective-permute-start | 1.6% |
| all-gather | 0.9% |

The custom-call block is Splash attention: forward residuals account for 15.0%
and DKV backward accounts for 11.7%.

## HLO Dump

No HLO dump was captured for the adopted v041 workload; the command did not set
`XLA_FLAGS=--xla_dump_to=...`.

## Verdict

Supported and promoted. v041 is the current best overall and target-sequence
frontier: 48,312 tok/s / 34.6% MFU at seq8192, global batch 24. It also shows
2.916 GiB free HBM, so the next direct frontier test is the same mechanism at
per-chip batch 4 / global batch 32.
