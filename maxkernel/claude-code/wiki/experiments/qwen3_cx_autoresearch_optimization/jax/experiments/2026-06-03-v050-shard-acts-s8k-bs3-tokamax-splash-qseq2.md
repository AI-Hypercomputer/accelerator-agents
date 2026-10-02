---
title: "Qwen3 JAX v050 - shard acts s8k bs3 tokamax splash qseq2"
type: experiment
hypothesis: "Splitting tokamax Splash query sequence work into two shards may improve attention custom-call scheduling versus v045."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, remat, maxtext-ce, activation-sharding, long-seq, invalid, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v050 - shard acts s8k bs3 tokamax splash qseq2

## Hypothesis under test

**Hypothesis**: Setting `TOKAMAX_Q_SEQ_SHARDS=2` may improve tokamax Splash
custom-call scheduling at the v045 target shape.

**Mechanism**: Keep v045 fixed (`USE_TOKAMAX_SPLASH=1`, base2 exp enabled,
fused reciprocal enabled, experimental scheduler disabled, default
`dq_reduction_steps=None`) and set only `TOKAMAX_Q_SEQ_SHARDS=2` when building
the tokamax Splash kernel.

**Predicted signal**: Clean run above v045's 49,069 tok/s / 35.2% MFU, or an
XProf custom-call split showing lower forward residual or DKV time.

**Falsification criterion**: Compile/runtime failure, invalid loss trajectory,
or clean completion at or below v045's 49,069 tok/s / 35.2% MFU.

## Setup

GKE workload `alekseyv-qwen3-v050`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v050-shard-acts-s8k-bs3-tokamax-splash-qseq2`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_Q_SEQ_SHARDS=2`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Invalid run:

- Worker 0: 54,513 tok/s total, 6,814 tok/s/chip, 39.1% nominal MFU.
- Worker 1: 54,537 tok/s total, 6,817 tok/s/chip, 39.1% nominal MFU.
- Loss became NaN at step 1 on both workers and stayed NaN through step 19.

The throughput is therefore not a valid frontier result. The knob changes the
attention kernel behavior enough to break the training loss at this shape.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v050-shard-acts-s8k-bs3-tokamax-splash-qseq2/2026_06_03_04_10_35`
- **MXU utilization**: 54.0%.
- **Peak HBM**: 28.33 GiB, 2.914 GiB free.
- **Device idle**: 0.3%.

Top device buckets:

- Convolution fusion: 51.4%.
- Loop fusion: 20.2%.
- Custom-call: 18.0%.

The invalid profile is still diagnostic: `TOKAMAX_Q_SEQ_SHARDS=2` cuts the
forward residual custom-call path sharply, but DKV remains 13.2% and the loss
is NaN from the first update.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v050-shard-acts-s8k-bs3-tokamax-splash-qseq2/hlo/`
- **Size**: 19.04 MiB
- **Object count**: 26

## Verdict

**Invalid.** Do not use `TOKAMAX_Q_SEQ_SHARDS=2` for this Qwen3 target shape
without a correctness fix. The apparent 39.1% MFU is unusable because the loss
becomes NaN at step 1.
