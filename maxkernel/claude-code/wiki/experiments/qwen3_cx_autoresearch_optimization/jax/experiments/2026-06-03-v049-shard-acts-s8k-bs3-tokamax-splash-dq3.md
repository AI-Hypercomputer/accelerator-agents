---
title: "Qwen3 JAX v049 - shard acts s8k bs3 tokamax splash dq3"
type: experiment
hypothesis: "Tokamax Splash DQ reduction steps may reduce backward attention custom-call time versus v045."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v049 - shard acts s8k bs3 tokamax splash dq3

## Hypothesis under test

**Hypothesis**: Setting `TOKAMAX_DQ_REDUCTION_STEPS=3` may reduce backward
attention custom-call time versus the v045 tokamax Splash frontier.

**Mechanism**: Keep v045 fixed (`USE_TOKAMAX_SPLASH=1`, base2 exp enabled,
fused reciprocal enabled, experimental scheduler disabled) and set only
`TOKAMAX_DQ_REDUCTION_STEPS=3`.

**Predicted signal**: Clean run above v045's 49,069 tok/s / 35.2% MFU, or an
XProf custom-call split showing lower DQ/DKV time.

**Falsification criterion**: Compile/runtime failure, invalid loss trajectory,
or clean completion at or below v045's 49,069 tok/s / 35.2% MFU.

## Setup

GKE workload `alekseyv-qwen3-v049`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v049-shard-acts-s8k-bs3-tokamax-splash-dq3`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_DQ_REDUCTION_STEPS=3`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run, but below v045:

- Worker 0: 48,855 tok/s total, 6,107 tok/s/chip, 35.0% MFU.
- Worker 1: 48,834 tok/s total, 6,104 tok/s/chip, 35.0% MFU.
- Step time stabilized around 4.02-4.03 s after warmup.
- Loss trajectory matched v045/v041, ending at 12.0509.

Compared with v045's 49,069 tok/s / 35.2% MFU, `DQ_REDUCTION_STEPS=3`
costs about 214-235 tok/s and 0.2 MFU points.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v049-shard-acts-s8k-bs3-tokamax-splash-dq3/2026_06_03_04_05_09`
- **MXU utilization**: 54.3%.
- **Peak HBM**: 28.33 GiB, 2.914 GiB free.
- **Device idle**: 0.3%.

Top device buckets:

- Convolution fusion: 46.7%.
- Custom-call: 25.0%.
- Loop fusion: 18.1%.
- Data formatting: 3.9%.

The custom-call bucket did move in the desired direction versus v045's 25.5%:
forward residuals were 13.4% and DKV was 11.6%. The wall-clock step time did
not improve, so the lower custom-call share was offset elsewhere.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v049-shard-acts-s8k-bs3-tokamax-splash-dq3/hlo/`
- **Size**: 19.32 MiB
- **Object count**: 26

## Verdict

**Refuted.** `TOKAMAX_DQ_REDUCTION_STEPS=3` slightly reduces the XProf
custom-call share but does not improve end-to-end throughput. Keep the v045
default `dq_reduction_steps=None`.
