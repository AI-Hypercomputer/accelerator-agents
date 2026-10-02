---
title: "Qwen3 JAX v045 - shard acts s8k bs3 tokamax splash"
type: experiment
hypothesis: "Switching only the Splash attention implementation to tokamax Splash may reduce v041's custom-call bucket and improve throughput."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, remat, maxtext-ce, activation-sharding, long-seq, supported, frontier, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v045 - shard acts s8k bs3 tokamax splash

## Hypothesis under test

**Hypothesis**: Switching only the Splash attention implementation to tokamax
Splash may reduce v041's custom-call bucket and improve throughput.

**Mechanism**: Reuse v041's image and target shape (`fsdp=8,tp=1`, seq8192,
per-chip batch 3, global batch 24) with scan, remat, MaxText CE, and activation
sharding unchanged. Set only `USE_TOKAMAX_SPLASH=1`, using the default tokamax
Splash options already wired in `splash_attn.py`.

**Predicted signal**: Clean run with the same loss trajectory and throughput
above v041's 48,312 tok/s / 34.6% MFU. XProf should show lower Splash
custom-call self-time or a better custom-call split.

**Falsification criterion**: Compile/runtime failure, invalid loss trajectory,
or clean completion at or below v041's 48,312 tok/s / 34.6% MFU.

## Setup

GKE workload `alekseyv-qwen3-v045`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v045-shard-acts-s8k-bs3-tokamax-splash`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Clean run on both workers. This is a new target-sequence frontier.

- Worker 0: 49,069 tok/s total, 6,134 tok/s/chip, 35.2% MFU.
- Worker 1: 49,048 tok/s total, 6,131 tok/s/chip, 35.2% MFU.
- Step time stabilized around 4.00-4.01 s after warmup.
- Loss trajectory matches the v041 synthetic-data pattern, ending at 12.0511.

Relative to v041's 48,312 tok/s / 34.6% MFU, v045 improves throughput by
757 tok/s (+1.57%) and MFU by 0.6 percentage points.

## Profile

- **XProf run**:
  `2026-06-03-qwen3-jax-v045-shard-acts-s8k-bs3-tokamax-splash/2026_06_03_03_38_35`
- **Average step time reported by XProf**: 1,725.2 ms over the trace window.
- **MXU utilization**: 54.5%.
- **Idle**: 0.3%.
- **Peak HBM**: 28.33 GiB / 31.25 GiB, with 2.914 GiB free.

Top device-time buckets:

| Bucket | Time % | Notes |
|--------|--------|-------|
| convolution fusion | 46.9% | Main matmul/MLP work. |
| custom-call | 25.5% | Down from v041's 26.7%. |
| loop fusion | 18.0% | Flat vs v041's 17.9%. |
| data formatting | 3.9% | Similar to v041's 3.8%. |
| collective-permute-start | 1.7% | Similar low collective overhead. |

Tokamax Splash custom-call split:

- `splash_mha_fwd_residuals.*`: 13.5%.
- `splash_mha_dkv_no_residuals.1`: 12.0%.

The win comes primarily from reducing forward Splash residual custom-call time;
DKV backward remains the main attention residual.

## HLO Dump

- **Path**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v045-shard-acts-s8k-bs3-tokamax-splash/hlo/`
- **Size**: 18.16 MiB
- **Object count**: 32

## Verdict

**Supported.** `USE_TOKAMAX_SPLASH=1` is a clean runtime win at the v041 target
shape with no extra HBM cost and no loss-trajectory issue. Promote this as the
new Qwen3 native-JAX target-sequence frontier and continue with tokamax Splash
sub-knobs, especially DKV/backward-related options, because forward residuals
improved but DKV remains about 12% of device time.
