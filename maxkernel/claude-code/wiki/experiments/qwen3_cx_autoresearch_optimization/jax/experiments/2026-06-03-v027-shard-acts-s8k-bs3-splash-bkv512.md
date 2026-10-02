---
title: "Qwen3 JAX v027 - shard acts s8k bs3 splash bkv512"
type: experiment
hypothesis: "Reducing Splash forward KV block size from 1024 to 512 may reduce custom-call time at the v041 target-sequence frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, scan, remat, maxtext-ce, activation-sharding, tp1, long-seq, block-tuning, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v027 - shard acts s8k bs3 splash bkv512

## Hypothesis under test

**Hypothesis**: Reducing Splash forward KV block size from 1024 to 512 may
reduce custom-call time at the v041 target-sequence frontier.

**Mechanism**: Reuse the v041 stack exactly at the known-fitting shape
(`fsdp=8,tp=1`, seq8192, per-chip batch 3/global batch 24) and set only
`SPLASH_BKV=512 SPLASH_BKV_COMPUTE=512`. Keep the DKV backward block sizes at
the v041 defaults.

**Predicted signal**: If smaller forward KV blocks improve Splash occupancy or
reduce custom-call stalls, MFU should beat v041's 34.6% while HBM stays near
28.33 GiB. If the extra tiling overhead dominates, throughput will be flat or
lower.

**Falsification criterion**: Compile/runtime failure, or clean completion at or
below v041's 48,312 tok/s / 34.6% MFU.

## Setup

Planned GKE workload `alekseyv-qwen3-v027`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v027-shard-acts-s8k-bs3-splash-bkv512`
- **Mesh**: `fsdp=8,tp=1`
- **Command shape**: `SPLASH_BKV=512 SPLASH_BKV_COMPUTE=512 --use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Refuted. Workload `alekseyv-qwen3-v027` completed cleanly, but throughput and
MFU regressed versus v041.

- **Worker 0 summary**: 47,128 tok/s, 5,891 tok/s/chip, 33.8% MFU
- **Worker 1 summary**: 47,110 tok/s, 5,889 tok/s/chip, 33.8% MFU
- **v041 comparison**: 48,312 tok/s, 6,039 tok/s/chip, 34.6% MFU
- **Delta**: -2.5% throughput and -0.8 pp MFU
- **Kernel confirmation**: startup printed
  `bq=2048 bkv=512 bq_dkv=2048 bkv_dkv=2048 fused_bwd=True`

## Profile

- **XProf run**: `2026-06-03-qwen3-jax-v027-shard-acts-s8k-bs3-splash-bkv512/2026_06_03_02_19_19`
- **MXU utilization**: 52.4% vs v041's 53.7%
- **Peak HBM**: 28.33 GiB of 31.25 GiB; 2.9165 GiB free

Top HLO op self-time from XProf:

| Op family | Self-time |
|-----------|-----------|
| convolution fusion | 44.9% |
| custom-call | 28.5% |
| loop fusion | 17.4% |
| data formatting | 3.7% |
| collective-permute-start | 1.6% |
| all-gather | 0.8% |

Compared with v041, the smaller forward KV block increased Splash custom-call
time from 26.7% to 28.5%. The forward residual custom calls rose to 17.1%
while DKV stayed near 11.4%.

## HLO Dump

HLO artifacts were dumped under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v027-shard-acts-s8k-bs3-splash-bkv512/hlo/`

The prefix contains 24 objects totaling 17.37 MiB.

## Verdict

Refuted. `SPLASH_BKV=512` worsens the attention custom-call bucket and does not
save HBM. Keep the v041/v040 forward KV defaults; if continuing block tuning,
target the DKV backward block separately.
