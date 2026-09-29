---
title: "Qwen3 JAX v072 - shard acts s8k bs3 tokamax splash customcall vmem adjust"
type: experiment
hypothesis: "Forcing TPU custom-call scoped-VMEM adjustments at the v068 frontier may improve Splash custom-call scheduling."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, custom-call, scheduler, remat, maxtext-ce, activation-sharding, long-seq, supported, frontier, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v072 - shard acts s8k bs3 tokamax splash customcall vmem adjust

## Hypothesis under test

**Hypothesis**: v068 is the current best VMEM point, but the profile still
spends about 23% of device time in Splash custom-calls. The flagfile leaves
`--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=auto`; forcing it to
`true` may let the scoped VMEM adjuster make stronger custom-call-specific
scheduling choices and reduce the Splash/custom-call or adjacent fusion cost.

**Mechanism**: Keep v068 fixed (`USE_TOKAMAX_SPLASH=1`,
`TOKAMAX_MAX_LOGIT_CONST=30.0`, VMEM 81920 KiB, scan/remat/MaxText CE/
activation sharding, scheduler rerun3, and AG backward pipelining enabled) and
add only `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true`.

**Predicted signal**: Loss stays near the v068 trajectory, ending around 12.05,
and throughput exceeds v068's 50,790 tok/s / 36.4% MFU, ideally with lower
custom-call share or lower train-step total time.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v068's 50,790 tok/s / 36.4% MFU.

## Setup

Planned GKE workload `alekseyv-qwen3-v072-ccvmem`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v072-shard-acts-s8k-bs3-tokamax-splash-customcall-vmem-adjust`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: add
  `--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true`, keeping
  `--xla_tpu_scoped_vmem_limit_kib=81920` and the rest of the v068 scheduler
  bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v072-ccvmem` completed cleanly on
`<your-cluster>`.

| Worker | Avg step | Avg throughput | Tok/s/chip | MFU | Loss end |
| --- | ---: | ---: | ---: | ---: | ---: |
| worker0 | 3.87 s | 50,793 tok/s | 6,349 | 36.4% | 12.0508 |
| worker1 | 3.88 s | 50,767 tok/s | 6,346 | 36.4% | 12.0508 |

This is a tiny improvement over v068's 50,790 / 50,763 tok/s, so it should be
treated as a provisional frontier until a direct rerun confirms the signal.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v072-shard-acts-s8k-bs3-tokamax-splash-customcall-vmem-adjust/2026_06_04_02_57_24`.

- Profile summary: MXU 56.6% on TPU v6 Lite.
- Op profile: 93,197.2 ms total, 0.3% idle, `jit_train_step` 99.7%.
- Top train-step buckets: convolution fusion 49.1%, custom-call 23.7%, loop
  fusion 18.0%, data formatting 4.1%, collective-permute-start 1.0%,
  all-gather 0.9%.
- Splash split: `splash_mha_dkv_no_residuals.1` 12.5%;
  `splash_mha_fwd_residuals.3` 5.6%;
  `splash_mha_fwd_residuals.2` 5.6%.
- Memory profile: peak HBM 28.32 GiB, stack reservation 22.508 GiB, heap
  5.8103 GiB, free 2.9278 GiB, utilization 90.62%.

The profile does not reduce the custom-call bucket itself, but improves the
reported train-step window and MXU slightly. The likely mechanism is scheduler
placement/overlap around the custom-call and fusion regions rather than a
faster Splash kernel.

## HLO Dump

HLO dump:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v072-shard-acts-s8k-bs3-tokamax-splash-customcall-vmem-adjust/hlo/`.

The train-step dump is `module_0267.jit_train_step.cl_854318611` with a
1,375,090-byte `after_optimizations.txt`; the dump prefix contains 1,619
objects. The flagfile confirms `--xla_tpu_scoped_vmem_limit_kib=81920`,
`--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='true'`,
`--xla_latency_hiding_scheduler_rerun=3`, and
`--xla_tpu_enable_ag_backward_pipelining=true`.

The run also printed per-Pallas custom-call HLO snippets to stdout during
compile, consistent with the custom-call scoped-VMEM path exercising additional
custom-call compilation/debug plumbing.

## Verdict

**Supported.** Forcing TPU custom-call scoped-VMEM adjustments at the v068 VMEM
point gives the best observed single-worker throughput so far, but only by a
few tokens/sec. v073 reran the same configuration and landed in the same
performance band, so keep the flag enabled while recognizing the gain is very
small.
