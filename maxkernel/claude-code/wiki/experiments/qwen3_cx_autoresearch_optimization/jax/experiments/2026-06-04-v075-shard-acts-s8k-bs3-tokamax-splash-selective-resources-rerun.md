---
title: "Qwen3 JAX v075 - shard acts s8k bs3 tokamax splash selective resources rerun"
type: experiment
hypothesis: "The v074 selective-resources scheduler gain is repeatable rather than measurement noise."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, scheduler, selective-resources, custom-call, rerun, remat, maxtext-ce, activation-sharding, long-seq, supported, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v075 - shard acts s8k bs3 tokamax splash selective resources rerun

## Hypothesis under test

**Hypothesis**: v074's 51,021 tok/s / 36.6% MFU result is repeatable. A second
run of the identical configuration should remain above the v072/v073 band and
near v074.

**Mechanism**: Exact rerun of v074: keep VMEM 81920, custom-call scoped-VMEM
adjustments, selective resources, scheduler rerun3, and the existing
Tokamax-Splash/MaxText-CE/activation-sharding stack.

**Predicted signal**: Valid loss ending around 12.05 and throughput near or
above 51k tok/s / 36.6% MFU.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion back in the 50.78k tok/s v072/v073 band.

## Setup

Planned GKE workload `alekseyv-qwen3-v075-selective-rerun`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v075-shard-acts-s8k-bs3-tokamax-splash-selective-resources-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: exact v074 bundle, including
  `--xla_tpu_scoped_vmem_limit_kib=81920`,
  `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true`, and
  `--xla_latency_hiding_scheduler_enable_selective_resources=true`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v075-selective-rerun` completed cleanly on
`<your-cluster>`.

| Worker | Avg step | Avg throughput | Tok/s/chip | MFU | Loss end |
| --- | ---: | ---: | ---: | ---: | ---: |
| worker0 | 3.85 s | 50,991 tok/s | 6,374 | 36.6% | 12.0508 |
| worker1 | 3.85 s | 51,021 tok/s | 6,378 | 36.6% | 12.0508 |

This reproduces the v074 selective-resources band with valid loss.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v075-shard-acts-s8k-bs3-tokamax-splash-selective-resources-rerun/2026_06_04_03_33_18`.

- Profile summary: MXU 56.7% on TPU v6 Lite.
- Op profile: 92,797.9 ms total, 0.3% idle, `jit_train_step` 99.7%.
- Top train-step buckets: convolution fusion 49.3%, custom-call 23.8%, loop
  fusion 18.0%, data formatting 4.1%, collective-permute-start 1.0%, reduce
  0.8%.
- Splash split: `splash_mha_dkv_no_residuals.1` 12.5%;
  `splash_mha_fwd_residuals.3` 5.6%;
  `splash_mha_fwd_residuals.2` 5.6%.
- Memory profile: peak HBM 28.32 GiB, stack reservation 22.508 GiB, heap
  5.8103 GiB, free 2.9278 GiB, utilization 90.62%.

The profile is effectively the same as v074, confirming the selective-resources
gain.

## HLO Dump

No new HLO dump objects were written for this exact rerun, consistent with a
persistent compilation-cache hit. Use the v074 HLO dump for the compiled
configuration:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v074-shard-acts-s8k-bs3-tokamax-splash-selective-resources/hlo/`.

## Verdict

**Supported.** v075 confirms the v074 selective-resources result. Carry forward
VMEM 81920, custom-call scoped-VMEM adjustments, and selective resources as the
current scheduler frontier.
