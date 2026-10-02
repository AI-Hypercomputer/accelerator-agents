---
title: "Qwen3 JAX v241 - RS2+AR2 scoped VMEM 100864"
type: experiment
hypothesis: "The material v230 RS2+AR2 frontier may respond differently to a mild high-side scoped-VMEM retune than the RS3+AR2 same-band alternative; testing 100864 KiB closes VMEM tuning for the actual frontier HLO family."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs2-ar2-vmem100864
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, scoped-vmem, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v241 - RS2+AR2 scoped VMEM 100864

## Hypothesis under test

**Hypothesis**: v230's RS2+AR2 HLO family remains the material frontier by
profile quality, while v236/v238 are same-band RS3+AR2 alternatives. v239/v240
closed the immediate low/high VMEM bracket for RS3+AR2; the actual RS2+AR2 HLO
family may still benefit from the high-side 100864 KiB point.

**Mechanism**: Keep v230 fixed except change
`--xla_tpu_scoped_vmem_limit_kib=100352` to `100864`.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v230/v236 band (**59,749-59,750 tok/s**) or a same-band run with lower
profile step time/peak HBM.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed
HLO with worse profile, higher memory pressure, or clean completion at or below
the v230/v236 band.

## Setup

GKE workload `alekseyv-q3-v241-rs2ar2-vmem100864`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v241-no-scan-sparsecore-rs-ar-rs2-ar2-vmem100864`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an XLA scoped-VMEM scheduler flag isolate
and preserves model math.

## Result

Completed cleanly on GKE as `alekseyv-q3-v241-rs2ar2-vmem100864`.

- Worker0 pod: `alekseyv-q3-v241-rs2ar2-vmem100864-slice-job-0-0-hsj2s`
- Worker1 pod: `alekseyv-q3-v241-rs2ar2-vmem100864-slice-job-0-1-9gmzn`
- Exit: both pods reported `EXIT_CODE=0`
- Loss: normal, ending at `12.0459`

Throughput:

| Worker | avg_2_19 tok/s | best_2_19 tok/s | avg_15_19 tok/s | avg_2_19 excl trace tok/s |
|--------|----------------|-----------------|-----------------|---------------------------|
| 0 | 59,679 | 59,874 | 59,744 | 59,715 |
| 1 | 59,702 | 59,879 | 59,745 | 59,714 |

Best full-window worker throughput was **59,702 tok/s**, below the confirmed
v230/v236 band (**59,749-59,750 tok/s**). Nominal MFU is about **42.8%**.

## Profile

XProf split hosts into timestamped runs:

- `2026-06-05-qwen3-jax-v241-no-scan-sparsecore-rs-ar-rs2-ar2-vmem100864/2026_06_05_09_14_12`
- `2026-06-05-qwen3-jax-v241-no-scan-sparsecore-rs-ar-rs2-ar2-vmem100864/2026_06_05_09_14_11`

Representative profile summary:

- Step time: **4404.0 ms**
- MXU utilization: **68.0%** on host `g30s`, **66.9%** on host `rkr2`
- Peak HBM: **30.93 GiB** / 31.25 GiB
- Stack reservation: **23.0798 GiB**
- Heap allocation: **7.8488 GiB**
- Free memory at peak: **0.3175 GiB**

Representative one-host train-step op profile:

| Bucket | Time | Share |
|--------|------|-------|
| convolution fusion | 28,804.9 ms | 54.5% |
| custom-call | 14,712.0 ms | 27.9% |
| loop fusion | 5,388.8 ms | 10.2% |

## HLO Dump

Optimized train-step HLO:

- Local copy: `/tmp/qwen3-v241-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256: `1932dd46ea5b84d5be8f0a627d6ace6b056438114bd1b21586750fb9dddd65e3`
- Size: **19,925,483 bytes**
- Lines: **119,435**

The HLO is distinct from the v230 RS2+AR2 frontier
(`8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`,
19,925,715 bytes), and it is also distinct from the RS3+AR2 v240 high-side
VMEM HLO.

## Verdict

**Refuted.** Raising scoped VMEM from 100352 KiB to 100864 KiB on the material
RS2+AR2 frontier changed the HLO but regressed throughput to **59,702 tok/s**
and worsened profiled step time to **4404.0 ms** versus v230's **4400.7 ms**.
Together with v239/v240 on the RS3+AR2 alternative, close immediate
post-SparseCore VMEM retuning and keep
`--xla_tpu_scoped_vmem_limit_kib=100352`.
