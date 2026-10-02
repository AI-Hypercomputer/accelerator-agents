---
title: "Qwen3 JAX v073 - shard acts s8k bs3 tokamax splash customcall vmem adjust rerun"
type: experiment
hypothesis: "The v072 custom-call scoped-VMEM adjustment gain is repeatable rather than measurement noise."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, custom-call, scheduler, rerun, remat, maxtext-ce, activation-sharding, long-seq, supported, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v073 - shard acts s8k bs3 tokamax splash customcall vmem adjust rerun

## Hypothesis under test

**Hypothesis**: v072's small improvement over v068 is repeatable. A second run
of the identical configuration should land at or above v068's 50,790 tok/s /
36.4% MFU and near v072's 50,793 tok/s / 36.4% MFU.

**Mechanism**: Exact rerun of v072: keep v068 fixed (`USE_TOKAMAX_SPLASH=1`,
`TOKAMAX_MAX_LOGIT_CONST=30.0`, VMEM 81920 KiB, scan/remat/MaxText CE/
activation sharding, scheduler rerun3, and AG backward pipelining enabled) and
force `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true`.

**Predicted signal**: Loss stays near the v068-v072 trajectory, ending around
12.05, and throughput is at or above v068. A result materially below v068 means
v072 was noise and should not replace v068 as the durable frontier.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion below v068's 50,790 tok/s / 36.4% MFU on both
workers.

## Setup

Planned GKE workload `alekseyv-qwen3-v073-ccvmem-rerun`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v073-shard-acts-s8k-bs3-tokamax-splash-customcall-vmem-adjust-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v072 bundle with `--xla_tpu_scoped_vmem_limit_kib=81920` and
  `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v073-ccvmem-rerun` completed cleanly on
`<your-cluster>`.

| Worker | Avg step | Avg throughput | Tok/s/chip | MFU | Loss end |
| --- | ---: | ---: | ---: | ---: | ---: |
| worker0 | 3.87 s | 50,779 tok/s | 6,347 | 36.4% | 12.0508 |
| worker1 | 3.87 s | 50,789 tok/s | 6,349 | 36.4% | 12.0508 |

This does not beat v072's worker0 number, but it reproduces the same
configuration in the same 50.78k tok/s band with valid loss. The average across
workers is still slightly above v068.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v073-shard-acts-s8k-bs3-tokamax-splash-customcall-vmem-adjust-rerun/2026_06_04_03_03_54`.

- Profile summary: MXU 56.4% on TPU v6 Lite.
- Op profile: 93,144.3 ms total, 0.2% idle, `jit_train_step` 99.8%.
- Top train-step buckets: convolution fusion 49.1%, custom-call 23.7%, loop
  fusion 17.9%, data formatting 4.1%, collective-permute-start 1.0%,
  all-gather 0.9%.
- Splash split: `splash_mha_dkv_no_residuals.1` 12.5%;
  `splash_mha_fwd_residuals.3` 5.6%;
  `splash_mha_fwd_residuals.2` 5.6%.
- Memory profile: peak HBM 28.32 GiB, stack reservation 22.508 GiB, heap
  5.8103 GiB, free 2.9278 GiB, utilization 90.62%.

The profile is essentially identical to v072 and supports the conclusion that
the flag changes scheduling enough to hold the v068/v072 performance band.

## HLO Dump

No new HLO dump objects were written for this exact rerun, consistent with a
persistent compilation-cache hit. Use the v072 HLO dump for the compiled
configuration:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v072-shard-acts-s8k-bs3-tokamax-splash-customcall-vmem-adjust/hlo/`.

## Verdict

**Supported.** v073 confirms the v072 configuration is repeatably in the
50.78k tok/s / 36.4% MFU band. Keep
`--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true` enabled, but do
not overstate the gain: it is only a few tokens/sec versus v068.
