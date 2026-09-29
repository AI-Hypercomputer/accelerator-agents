---
title: "Qwen3 JAX v239 - RS3+AR2 scoped VMEM 98304"
type: experiment
hypothesis: "The RS3+AR2 SparseCore frontier inherited scoped VMEM 100352 KiB from the pre-SparseCore no-scan frontier, but SparseCore offload raises peak HBM to 30.93 GiB; lowering scoped VMEM to 98304 may select a lower-pressure schedule without losing throughput."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs3-ar2-vmem98304
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, scoped-vmem, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v239 - RS3+AR2 scoped VMEM 98304

## Hypothesis under test

**Hypothesis**: The confirmed RS3+AR2 stack uses scoped VMEM 100352 KiB, a
setting tuned before the SparseCore collective-offload family. Because the
current SC frontier runs at **30.93 GiB** peak HBM with little free memory,
lowering scoped VMEM to 98304 KiB may reduce memory pressure or select a better
post-SC schedule.

**Mechanism**: Keep v236 fixed except change
`--xla_tpu_scoped_vmem_limit_kib=100352` to `98304`.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v230/v236 band (**59,749-59,750 tok/s**) or an equal throughput band with
lower peak HBM/profile step time.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed
HLO with worse profile, higher memory pressure, or clean completion below the
v230/v236 band.

## Setup

GKE workload `alekseyv-q3-v239-vmem98304`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v239-no-scan-sparsecore-rs-ar-rs3-ar2-vmem98304`
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

Completed cleanly on GKE as `alekseyv-q3-v239-vmem98304`.

- Worker0 pod: `alekseyv-q3-v239-vmem98304-slice-job-0-0-ljq27`
- Worker1 pod: `alekseyv-q3-v239-vmem98304-slice-job-0-1-zbdl5`
- Exit: both pods reported `EXIT_CODE=0`
- Loss: normal, ending at `12.0462`

Throughput:

| Worker | avg_2_19 tok/s | best_2_19 tok/s | avg_15_19 tok/s | avg_2_19 excl trace tok/s |
|--------|----------------|-----------------|-----------------|---------------------------|
| 0 | 59,440 | 59,724 | 59,237 | 59,437 |
| 1 | 59,507 | 59,728 | 59,577 | 59,553 |

Best full-window worker throughput was **59,507 tok/s**, below the confirmed
v230/v236 band (**59,749-59,750 tok/s**). Nominal MFU is about **42.7%**.

## Profile

XProf split the two hosts into timestamped one-host runs:

- `2026-06-05-qwen3-jax-v239-no-scan-sparsecore-rs-ar-rs3-ar2-vmem98304/2026_06_05_08_46_19`
- `2026-06-05-qwen3-jax-v239-no-scan-sparsecore-rs-ar-rs3-ar2-vmem98304/2026_06_05_08_46_18`

Representative profile summary:

- Step time: **4418.9 ms**
- MXU utilization: **67.7%** on host `g30s`, **66.7%** on host `rkr2`
- Peak HBM: **30.89 GiB** / 31.25 GiB
- Stack reservation: **23.077 GiB**
- Heap allocation: **7.8166 GiB**
- Free memory at peak: **0.3525 GiB**

Representative one-host train-step op profile:

| Bucket | Time | Share |
|--------|------|-------|
| convolution fusion | 29,072.2 ms | 54.9% |
| custom-call | 14,721.4 ms | 27.8% |
| loop fusion | 5,403.0 ms | 10.2% |

## HLO Dump

Optimized train-step HLO:

- Local copy: `/tmp/qwen3-v239-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256: `6ca611a629704d2fbc70da3a9b34244b60a11c19d4418dade3ee3d86474734ef`
- Size: **19,838,938 bytes**
- Lines: **118,766**

This is a distinct, smaller HLO than the v236/v238 RS3+AR2 schedule
(`d4d46797cc4508667e0aad8b0a75bf6485f046806db6ced027a159f89ee80b5c`,
19,909,611 bytes), but the smaller HLO did not translate into a faster
schedule.

## Verdict

**Refuted.** Lowering scoped VMEM from 100352 KiB to 98304 KiB on the
post-SparseCore RS3+AR2 stack selected a distinct lower-HLO-size schedule, but
regressed throughput to **59,507 tok/s** and worsened profiled step time to
**4418.9 ms**. Keep `--xla_tpu_scoped_vmem_limit_kib=100352` for the current
frontier; the low-side post-SparseCore VMEM retune is closed.
