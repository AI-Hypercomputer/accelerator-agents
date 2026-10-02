---
title: "Qwen3 JAX v029 - shard acts s8k bs3 splash unfused bwd"
type: experiment
hypothesis: "Disabling fused Splash backward may reduce DKV custom-call time at the v041 target-sequence frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, scan, remat, maxtext-ce, activation-sharding, tp1, long-seq, block-tuning, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v029 - shard acts s8k bs3 splash unfused bwd

## Hypothesis under test

**Hypothesis**: Disabling fused Splash backward may reduce DKV custom-call time
at the v041 target-sequence frontier.

**Mechanism**: Reuse the v041 stack exactly at the known-fitting shape
(`fsdp=8,tp=1`, seq8192, per-chip batch 3/global batch 24) and set only
`SPLASH_FUSED_BWD=0`. The prior block-size probes left the custom-call bucket
flat or worse, so this tests a different Splash backward implementation path
without changing model semantics or batch shape.

**Predicted signal**: If fused backward is the source of the DKV custom-call
stall, MFU should beat v041's 34.6% and custom-call share should fall below
26.7%. If fusion is required for locality or launch efficiency, throughput will
be flat or lower.

**Falsification criterion**: Compile/runtime failure, or clean completion at or
below v041's 48,312 tok/s / 34.6% MFU.

## Setup

Planned GKE workload `alekseyv-qwen3-v029`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v029-shard-acts-s8k-bs3-splash-unfused-bwd`
- **Mesh**: `fsdp=8,tp=1`
- **Command shape**: `SPLASH_FUSED_BWD=0 --use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Refuted. Workload `alekseyv-qwen3-v029` completed cleanly, but throughput and
MFU regressed versus v041.

- **Worker 0 summary**: 46,086 tok/s, 5,761 tok/s/chip, 33.0% MFU
- **Worker 1 summary**: 46,103 tok/s, 5,763 tok/s/chip, 33.1% MFU
- **v041 comparison**: 48,312 tok/s, 6,039 tok/s/chip, 34.6% MFU
- **Delta**: -4.6% throughput and -1.5 to -1.6 pp MFU
- **Kernel confirmation**: startup printed
  `bq=2048 bkv=1024 bq_dkv=2048 bkv_dkv=2048 fused_bwd=False bq_dq=2048 bkv_dq=1024`

## Profile

- **XProf run**: `2026-06-03-qwen3-jax-v029-shard-acts-s8k-bs3-splash-unfused-bwd/2026_06_03_02_34_49`
- **MXU utilization**: 55.0% vs v041's 53.7%
- **Idle time**: 0.3% from op profile
- **Peak HBM**: 28.33 GiB of 31.25 GiB; 2.9151 GiB free
- **Stack / heap**: 22.5089 GiB stack reservation, 5.8221 GiB heap allocation

Top HLO op self-time from XProf:

| Op family | Self-time |
|-----------|-----------|
| convolution fusion | 43.9% |
| custom-call | 30.9% |
| loop fusion | 16.8% |
| data formatting | 4.1% |
| collective-permute-start | 1.6% |
| all-gather | 0.6% |
| broadcast | 0.5% |

Disabling fused backward splits the attention backward into separate DKV and DQ
custom calls and increases total attention self-time. The custom-call bucket is
14.3% forward residuals, 8.8% DKV backward, and 7.8% DQ backward. v041's fused
path kept the whole custom-call bucket at 26.7%, with 15.0% forward residuals
and 11.7% fused backward.

## HLO Dump

HLO artifacts were dumped under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v029-shard-acts-s8k-bs3-splash-unfused-bwd/hlo/`

The prefix contains 32 objects totaling 18.07 MiB.

## Verdict

Refuted. The fused Splash backward path is necessary at this shape: disabling it
does not save HBM and makes attention custom-call self-time materially worse.
Keep `SPLASH_FUSED_BWD=True` for the v041 frontier.
