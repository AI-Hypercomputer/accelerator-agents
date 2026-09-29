---
title: "Qwen3 JAX v048 - shard acts s8k bs3 tokamax splash no fuse reciprocal"
type: experiment
hypothesis: "Disabling tokamax Splash's fused reciprocal path may improve custom-call scheduling versus v045."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v048 - shard acts s8k bs3 tokamax splash no fuse reciprocal

## Hypothesis under test

**Hypothesis**: Disabling tokamax Splash's fused reciprocal path may improve
custom-call scheduling versus v045.

**Mechanism**: Keep v045 fixed (`USE_TOKAMAX_SPLASH=1`, base2 exp enabled,
experimental scheduler disabled) and set only `TOKAMAX_FUSE_RECIPROCAL=0`.

**Predicted signal**: Clean run above v045's 49,069 tok/s / 35.2% MFU, or an
XProf custom-call split showing lower forward residual or DKV time.

**Falsification criterion**: Compile/runtime failure, invalid loss trajectory,
or clean completion at or below v045's 49,069 tok/s / 35.2% MFU.

## Setup

GKE workload `alekseyv-qwen3-v048`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v048-shard-acts-s8k-bs3-tokamax-splash-no-fuse-recip`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_FUSE_RECIPROCAL=0`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run, but below v045:

- Worker 0: 48,626 tok/s total, 6,078 tok/s/chip, 34.9% MFU.
- Worker 1: 48,606 tok/s total, 6,076 tok/s/chip, 34.9% MFU.
- Step time stabilized around 4.04-4.05 s after warmup.
- Loss trajectory matched v045/v041, ending at 12.0511.

The `tokamax-splash` config confirmed `fuse_recip=False`, so the knob was
active. Compared with v045's 49,069 tok/s / 35.2% MFU, disabling fused
reciprocal costs about 443-463 tok/s and 0.3 MFU points.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v048-shard-acts-s8k-bs3-tokamax-splash-no-fuse-recip`
- Detailed XProf analysis skipped because wall-clock logs already refute the
  hypothesis against v045.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v048-shard-acts-s8k-bs3-tokamax-splash-no-fuse-recip/hlo/`
- **Size**: 18.3 MiB
- **Object count**: 32

## Verdict

**Refuted.** Keep tokamax Splash fused reciprocal enabled. The default v045
tokamax Splash configuration remains the frontier.
