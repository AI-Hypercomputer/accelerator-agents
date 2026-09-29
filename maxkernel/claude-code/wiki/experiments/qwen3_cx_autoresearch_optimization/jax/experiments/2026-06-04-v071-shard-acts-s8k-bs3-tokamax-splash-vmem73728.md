---
title: "Qwen3 JAX v071 - shard acts s8k bs3 tokamax splash vmem73728"
type: experiment
hypothesis: "A lower-side VMEM midpoint at 73728 KiB may improve on 81920 KiB while avoiding the 65536 KiB regression."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, vmem, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v071 - shard acts s8k bs3 tokamax splash vmem73728

## Hypothesis under test

**Hypothesis**: VMEM 81920 KiB is the current best point, 65536 KiB is too low,
and 90112 KiB is too high. A lower-side midpoint at 73728 KiB may retain the
beneficial tighter scheduling pressure from v068 while avoiding the stronger
constraint that hurt v069.

**Mechanism**: Keep v068 fixed (`USE_TOKAMAX_SPLASH=1`,
`TOKAMAX_MAX_LOGIT_CONST=30.0`, scan/remat/MaxText CE/activation sharding,
scheduler rerun3, and AG backward pipelining enabled) and change only
`--xla_tpu_scoped_vmem_limit_kib=81920` to `73728`.

**Predicted signal**: Loss stays near the v068-v070 trajectory, ending around
12.05, and throughput exceeds v068's 50,790 tok/s / 36.4% MFU. If it does not,
the VMEM flag axis is bracketed and should be retired.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v068's 50,790 tok/s / 36.4% MFU.

## Setup

Planned GKE workload `alekseyv-qwen3-v071-vmem73728`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v071-shard-acts-s8k-bs3-tokamax-splash-vmem73728`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: `--xla_tpu_scoped_vmem_limit_kib=73728` instead of
  v068's `81920`, keeping the rest of the v068 scheduler bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v071-vmem73728` completed cleanly on
`<your-cluster>`.

| Worker | Avg step | Avg throughput | Tok/s/chip | MFU | Loss end |
| --- | ---: | ---: | ---: | ---: | ---: |
| worker0 | 3.92 s | 50,174 tok/s | 6,272 | 36.0% | 12.0510 |
| worker1 | 3.92 s | 50,155 tok/s | 6,269 | 36.0% | 12.0510 |

Loss stayed valid and this lower midpoint beat v069/v070, but it did not beat
v068's 50,790 tok/s / 36.4% MFU.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v071-shard-acts-s8k-bs3-tokamax-splash-vmem73728/2026_06_04_02_47_45`.

- Profile summary: MXU 56.0% on TPU v6 Lite.
- Op profile: 94,316.3 ms total, 0.3% idle, `jit_train_step` 99.7%.
- Top train-step buckets: convolution fusion 49.6%, custom-call 23.3%, loop
  fusion 17.4%, data formatting 3.9%, collective-permute-start 1.2%,
  all-gather 1.0%.
- Splash split: `splash_mha_dkv_no_residuals.1` 12.3%;
  `splash_mha_fwd_residuals.2` 5.5%;
  `splash_mha_fwd_residuals.3` 5.4%.
- Memory profile: peak HBM 28.31 GiB, stack reservation 22.5092 GiB, heap
  5.8056 GiB, free 2.9312 GiB, utilization 90.59%.

The midpoint is better than the too-low 65536 cap, but the wall-clock and MFU
remain below v068.

## HLO Dump

HLO dump:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v071-shard-acts-s8k-bs3-tokamax-splash-vmem73728/hlo/`.

The train-step dumps include `module_0264.jit_train_step.cl_854318611` and
`module_0267.jit_train_step.cl_854318611`, each with a 1,417,462-byte
`after_optimizations.txt`; the dump prefix contains 2,226 objects. The
`module_0267` flagfile confirms `--xla_tpu_scoped_vmem_limit_kib=73728`,
`--xla_latency_hiding_scheduler_rerun=3`, and
`--xla_tpu_enable_ag_backward_pipelining=true`. It also leaves
`--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=auto`.

## Verdict

**Refuted.** VMEM 73728 KiB is valid but slower than v068. The VMEM sweep is
now bracketed: 65536, 73728, 90112, and 98304 all lose to 81920. Keep VMEM
81920 as the frontier and move to custom-call scheduler behavior.
