---
title: "Qwen3 JAX v031 - shard acts s8k bs4 tokamax CE"
type: experiment
hypothesis: "The v030 tokamax-CE memory relief may enable seq8192 global batch 32 and recover enough throughput to beat the v041 frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, scan, remat, tokamax-ce, activation-sharding, tp1, long-seq, ce, batch-scaling, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v031 - shard acts s8k bs4 tokamax CE

## Hypothesis under test

**Hypothesis**: The v030 tokamax-CE memory relief may enable seq8192 global
batch 32 and recover enough throughput to beat the v041 frontier.

**Mechanism**: Reuse the v030 CE stack (`--use_tokamax_ce=True
--tokamax_ce_impl=mosaic_tpu`) and increase only `--batch_size` from 3 to 4.
Keep `fsdp=8,tp=1`, seq8192, scan, remat, Splash, and activation sharding fixed.

**Predicted signal**: v030 left 10.47 GiB free HBM at batch 3. If tokamax CE
scales to batch 4, the larger global batch 32 should lift total tokens/sec and
MFU above v041's 48,312 tok/s / 34.6% despite tokamax's per-token overhead.

**Falsification criterion**: Compile/runtime failure, or clean completion at or
below v041's 48,312 tok/s / 34.6% MFU.

## Setup

Planned GKE workload `alekseyv-qwen3-v031`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v031-shard-acts-s8k-bs4-tokamax-ce`
- **Mesh**: `fsdp=8,tp=1`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Refuted. Workload `alekseyv-qwen3-v031` completed cleanly and fit batch 4
/ global batch 32, but throughput and MFU stayed below v041.

- **Worker 0 summary**: 45,449 tok/s, 5,681 tok/s/chip, 32.6% MFU
- **Worker 1 summary**: 45,461 tok/s, 5,683 tok/s/chip, 32.6% MFU
- **v041 comparison**: 48,312 tok/s, 6,039 tok/s/chip, 34.6% MFU
- **Delta**: -5.9% throughput and -2.0 pp MFU
- **Tokamax confirmation**: startup emitted tokamax autotuning for
  `x=(32768,4096)`, `labels=(32768,)`, `w=(4096,151936)`, all fp32.

## Profile

- **XProf run**: `2026-06-03-qwen3-jax-v031-shard-acts-s8k-bs4-tokamax-ce/2026_06_03_02_49_46`
- **MXU utilization**: 51.8% vs v041's 53.7%
- **Idle time**: 0.2% from op profile
- **Peak HBM**: 24.75 GiB of 31.25 GiB; 6.5005 GiB free
- **Stack / heap**: 18.9261 GiB stack reservation, 5.8195 GiB heap allocation

Top HLO op self-time from XProf:

| Op family | Self-time |
|-----------|-----------|
| convolution fusion | 41.0% |
| custom-call | 34.3% |
| loop fusion | 15.1% |
| data formatting | 3.7% |
| collective-permute-start | 1.6% |
| all-gather | 0.8% |
| all-reduce | 0.8% |
| reduce | 0.7% |

The larger batch consumes the tokamax memory headroom but not profitably. The
custom-call bucket remains the limiter: 14.1% forward Splash residuals, 11.0%
Splash DKV, and 7.3% tokamax CE backward.

## HLO Dump

HLO artifacts were dumped under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v031-shard-acts-s8k-bs4-tokamax-ce/hlo/`

The prefix contains 63 objects totaling 35.86 MiB.

## Verdict

Refuted. Tokamax CE solves the v026 batch-4 memory failure but is too slow at
seq8192 on v6e-8: batch 4 remains below v041's MaxText-CE batch-3 frontier.
Further CE work needs a faster streamed/fused CE implementation, not simply
larger tokamax-CE batch scaling.
