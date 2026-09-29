---
title: "Qwen3 JAX v028 - shard acts s8k bs3 splash bkv dkv1024"
type: experiment
hypothesis: "Reducing Splash backward KV block size from 2048 to 1024 may reduce DKV custom-call time at the v041 frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, scan, remat, maxtext-ce, activation-sharding, tp1, long-seq, block-tuning, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v028 - shard acts s8k bs3 splash bkv dkv1024

## Hypothesis under test

**Hypothesis**: Reducing Splash backward KV block size from 2048 to 1024 may
reduce DKV custom-call time at the v041 frontier.

**Mechanism**: Reuse the v041 stack at the known-fitting target shape and set
only `SPLASH_BKV_DKV=1024 SPLASH_BKV_DKV_COMPUTE=1024`. Keep forward block
sizes at v041 defaults (`bq=2048,bkv=1024`) because v027 showed smaller forward
KV blocks are worse.

**Predicted signal**: If DKV block size is the limiting custom-call component,
MFU should improve over v041 and custom-call share should fall below 26.7%. If
extra backward tiling overhead dominates, throughput will be flat or lower.

**Falsification criterion**: Compile/runtime failure, or clean completion at or
below v041's 48,312 tok/s / 34.6% MFU.

## Setup

Planned GKE workload `alekseyv-qwen3-v028`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v028-shard-acts-s8k-bs3-splash-bkv-dkv1024`
- **Mesh**: `fsdp=8,tp=1`
- **Command shape**: `SPLASH_BKV_DKV=1024 SPLASH_BKV_DKV_COMPUTE=1024 --use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Refuted. Workload `alekseyv-qwen3-v028` completed cleanly, but throughput and
MFU regressed versus v041.

- **Worker 0 summary**: 47,563 tok/s, 5,945 tok/s/chip, 34.1% MFU
- **Worker 1 summary**: 47,547 tok/s, 5,943 tok/s/chip, 34.1% MFU
- **v041 comparison**: 48,312 tok/s, 6,039 tok/s/chip, 34.6% MFU
- **Delta**: -1.55% throughput and -0.5 pp MFU
- **Kernel confirmation**: startup printed
  `bq=2048 bkv=1024 bq_dkv=2048 bkv_dkv=1024 fused_bwd=True`

## Profile

- **XProf run**: `2026-06-03-qwen3-jax-v028-shard-acts-s8k-bs3-splash-bkv-dkv1024/2026_06_03_02_25_57`
- **MXU utilization**: 52.9% vs v041's 53.7%
- **Peak HBM**: 28.33 GiB of 31.25 GiB; 2.9168 GiB free
- **Stack / heap**: 22.5088 GiB stack reservation, 5.8205 GiB heap allocation

Top HLO op self-time from XProf:

| Op family | Self-time |
|-----------|-----------|
| convolution fusion | 45.5% |
| custom-call | 27.0% |
| loop fusion | 17.6% |
| data formatting | 3.8% |
| collective-permute-start | 1.7% |
| reduce | 1.4% |
| all-gather | 0.8% |

Compared with v041, the smaller DKV block did not reduce the attention bucket:
custom-call stayed roughly flat-to-worse at 27.0% vs 26.7%. The forward
residual custom calls stayed near 14.8%, while DKV backward rose to 12.2% from
v041's 11.7%.

## HLO Dump

HLO artifacts were dumped under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v028-shard-acts-s8k-bs3-splash-bkv-dkv1024/hlo/`

The prefix contains 24 objects totaling 17.36 MiB.

## Verdict

Refuted. `SPLASH_BKV_DKV=1024` makes the DKV custom-call component slightly
worse and does not save HBM. Keep the v041/v040 DKV defaults; remaining
attention probes should target a different Splash path rather than smaller KV
tiling.
