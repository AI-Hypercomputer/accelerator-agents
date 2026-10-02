---
title: "Qwen3 JAX v070 - shard acts s8k bs3 tokamax splash vmem90112"
type: experiment
hypothesis: "An upper-side VMEM bracket at 90112 KiB may keep v068's scheduling gain while avoiding the v069 low-cap regression."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, vmem, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v070 - shard acts s8k bs3 tokamax splash vmem90112

## Hypothesis under test

**Hypothesis**: v068 improved by lowering the VMEM cap from 98304 KiB to
81920 KiB, while v069 regressed at 65536 KiB. A midpoint on the upper side,
90112 KiB, may preserve the scheduler freedom that helped v068 while avoiding
the too-low cap behavior from v069.

**Mechanism**: Keep v068 fixed (`USE_TOKAMAX_SPLASH=1`,
`TOKAMAX_MAX_LOGIT_CONST=30.0`, scan/remat/MaxText CE/activation sharding,
scheduler rerun3, and AG backward pipelining enabled) and change only
`--xla_tpu_scoped_vmem_limit_kib=81920` to `90112`.

**Predicted signal**: Loss stays near the v068/v069 trajectory, ending around
12.05, and throughput exceeds v068's 50,790 tok/s / 36.4% MFU or narrows the
VMEM optimum enough to justify leaving the flag axis.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v068's 50,790 tok/s / 36.4% MFU.

## Setup

Planned GKE workload `alekseyv-qwen3-v070-vmem90112`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v070-shard-acts-s8k-bs3-tokamax-splash-vmem90112`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: `--xla_tpu_scoped_vmem_limit_kib=90112` instead of
  v068's `81920`, keeping the rest of the v068 scheduler bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v070-vmem90112` completed cleanly on
`<your-cluster>`.

| Worker | Avg step | Avg throughput | Tok/s/chip | MFU | Loss end |
| --- | ---: | ---: | ---: | ---: | ---: |
| worker0 | 3.93 s | 50,015 tok/s | 6,252 | 35.9% | 12.0510 |
| worker1 | 3.93 s | 50,034 tok/s | 6,254 | 35.9% | 12.0510 |

Loss stayed valid, but throughput landed below v068's 50,790 tok/s / 36.4% MFU
and below the original v056/v068 VMEM endpoints.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v070-shard-acts-s8k-bs3-tokamax-splash-vmem90112/2026_06_04_02_37_40`.

- Profile summary: MXU 55.7% on TPU v6 Lite.
- Op profile: 94,531.7 ms total, 0.3% idle, `jit_train_step` 99.7%.
- Top train-step buckets: convolution fusion 47.9%, custom-call 23.3%, loop
  fusion 18.6%, data formatting 4.0%, collective-permute-start 1.7%,
  all-gather 1.1%.
- Splash split: `splash_mha_dkv_no_residuals.1` 12.3%;
  `splash_mha_fwd_residuals.3` + `.2` 11.0%.
- Memory profile: peak HBM 28.32 GiB, stack reservation 22.5085 GiB, heap
  5.8155 GiB, free 2.9221 GiB, utilization 90.62%.

The upper-side VMEM point keeps memory pressure similar to v068 but gives lower
MXU and higher loop-fusion share, so the scheduling effect is worse.

## HLO Dump

HLO dump:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v070-shard-acts-s8k-bs3-tokamax-splash-vmem90112/hlo/`.

The train-step dump is `module_0267.jit_train_step.cl_854318611` with a
1,373,455-byte `after_optimizations.txt`; the dump prefix contains 1,619
objects. The flagfile confirms `--xla_tpu_scoped_vmem_limit_kib=90112`,
`--xla_latency_hiding_scheduler_rerun=3`, and
`--xla_tpu_enable_ag_backward_pipelining=true`.

## Verdict

**Refuted.** VMEM 90112 KiB is valid but slower than the v068 frontier. The
best observed VMEM point remains 81920 KiB; run one lower-side midpoint at
73728 KiB to finish the bracket, then pivot away from VMEM flag sweeps.
