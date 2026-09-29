---
title: "Qwen3 JAX v069 - shard acts s8k bs3 tokamax splash vmem65536"
type: experiment
hypothesis: "Lowering the v068 VMEM limit further to 65536 KiB may improve or bracket the VMEM schedule optimum."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, vmem, remat, maxtext-ce, activation-sharding, long-seq, refuted, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v069 - shard acts s8k bs3 tokamax splash vmem65536

## Hypothesis under test

**Hypothesis**: v068 showed that reducing the v056 VMEM cap from 98304 KiB to
81920 KiB improves throughput. Reducing the limit further to 65536 KiB may
either improve scheduling again or identify 81920 as the better v6e op-point.

**Mechanism**: Keep v068 fixed (`USE_TOKAMAX_SPLASH=1`,
`TOKAMAX_MAX_LOGIT_CONST=30.0`, scan/remat/MaxText CE/activation sharding,
scheduler rerun3, and AG backward pipelining enabled) and change only
`--xla_tpu_scoped_vmem_limit_kib=81920` to `65536`.

**Predicted signal**: Loss stays near the v068 trajectory, ending around 12.05,
and throughput exceeds v068's 50,790 tok/s / 36.4% MFU or shows lower
loop-fusion/custom-call scheduling overhead without increasing HBM pressure.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss
trajectory, or clean completion at or below v068's 50,790 tok/s / 36.4% MFU.

## Setup

Planned GKE workload `alekseyv-qwen3-v069-vmem65536`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v069-shard-acts-s8k-bs3-tokamax-splash-vmem65536`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag change**: `--xla_tpu_scoped_vmem_limit_kib=65536` instead of
  v068's `81920`, keeping the rest of the v068 scheduler bundle.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v069-vmem65536` completed cleanly on
`<your-cluster>`.

| Worker | Avg step | Avg throughput | Tok/s/chip | MFU | Loss end |
| --- | ---: | ---: | ---: | ---: | ---: |
| worker0 | 3.95 s | 49,785 tok/s | 6,223 | 35.7% | 12.0509 |
| worker1 | 3.95 s | 49,761 tok/s | 6,220 | 35.7% | 12.0509 |

Loss stayed valid, but throughput regressed versus v068's 50,790 tok/s /
36.4% MFU. The lower VMEM cap is therefore below the useful scheduling range
for this shape.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v069-shard-acts-s8k-bs3-tokamax-splash-vmem65536/2026_06_04_02_25_54`.

- Profile summary: MXU 55.6% on TPU v6 Lite.
- Op profile: 95,100.9 ms total, 0.3% idle, `jit_train_step` 99.7%.
- Top train-step buckets: convolution fusion 48.8%, custom-call 23.0%, loop
  fusion 17.2%, data formatting 4.1%, collective-permute-start 2.3%, all-gather
  1.0%.
- Splash split: `splash_mha_dkv_no_residuals.1` 12.2%;
  `splash_mha_fwd_residuals.3` + `.2` 10.8%.
- Memory profile: peak HBM 28.30 GiB, stack reservation 22.5079 GiB, heap
  5.795 GiB, free 2.9432 GiB, utilization 90.56%.

Compared with v068, the lower VMEM cap reduced custom-call share slightly but
increased total train-step time and lowered MXU, so the wall-clock signal is
negative.

## HLO Dump

HLO dump:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v069-shard-acts-s8k-bs3-tokamax-splash-vmem65536/hlo/`.

The train-step dumps include `module_0264.jit_train_step.cl_854318611` and
`module_0267.jit_train_step.cl_854318611`, each with a 1,422,873-byte
`after_optimizations.txt`; the dump prefix contains 2,226 objects. The
`module_0267` flagfile confirms `--xla_tpu_scoped_vmem_limit_kib=65536`,
`--xla_latency_hiding_scheduler_rerun=3`, and
`--xla_tpu_enable_ag_backward_pipelining=true`.

## Verdict

**Refuted.** VMEM 65536 KiB is valid but slower than both v068 and v056. Keep
VMEM 81920 KiB as the current frontier and run one upper-side bracket point
between 81920 and 98304 before pivoting back to structural attention work.
