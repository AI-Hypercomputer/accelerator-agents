---
title: "Qwen3 JAX v240 - RS3+AR2 scoped VMEM 100864"
type: experiment
hypothesis: "The post-SparseCore RS3+AR2 stack may have a narrow scoped-VMEM schedule optimum around the pre-SparseCore 100352 KiB frontier; after v239 refuted the lower side at 98304, testing 100864 closes the immediate high-side bracket."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs3-ar2-vmem100864
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, scoped-vmem, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v240 - RS3+AR2 scoped VMEM 100864

## Hypothesis under test

**Hypothesis**: v239 showed that lowering the post-SparseCore RS3+AR2 scoped
VMEM point to 98304 KiB selects a distinct but slower schedule. A mild
high-side retune to 100864 KiB may either preserve the v236 HLO as a no-op or
select a better schedule around the current 100352 KiB frontier.

**Mechanism**: Keep v236 fixed except change
`--xla_tpu_scoped_vmem_limit_kib=100352` to `100864`.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v230/v236 band (**59,749-59,750 tok/s**) or a same-band run with lower
profile step time/peak HBM.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed
HLO with worse profile, higher memory pressure, or clean completion at or below
the v230/v236 band.

## Setup

GKE workload `alekseyv-q3-v240-vmem100864`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v240-no-scan-sparsecore-rs-ar-rs3-ar2-vmem100864`
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

Completed cleanly on GKE as `alekseyv-q3-v240-vmem100864`.

An initial launch failed immediately because the xpk command body was wrapped
twice (`((set ...))`). That invalid JobSet was deleted with `xpk workload
delete`, and the same workload name/run directory was resubmitted with the
command body unwrapped. The resubmitted job completed normally.

- Worker0 pod: `alekseyv-q3-v240-vmem100864-slice-job-0-0-bgzq5`
- Worker1 pod: `alekseyv-q3-v240-vmem100864-slice-job-0-1-jtplb`
- Exit: both pods reported `EXIT_CODE=0`
- Loss: normal, ending at `12.0459`

Throughput:

| Worker | avg_2_19 tok/s | best_2_19 tok/s | avg_15_19 tok/s | avg_2_19 excl trace tok/s |
|--------|----------------|-----------------|-----------------|---------------------------|
| 0 | 59,689 | 59,901 | 59,758 | 59,728 |
| 1 | 59,714 | 59,890 | 59,759 | 59,730 |

Best full-window worker throughput was **59,714 tok/s**, below the confirmed
v230/v236 band (**59,749-59,750 tok/s**). Nominal MFU is about **42.8%**.

## Profile

XProf run:
`2026-06-05-qwen3-jax-v240-no-scan-sparsecore-rs-ar-rs3-ar2-vmem100864/2026_06_05_09_01_58`.

- Step time: **4404.1 ms**
- MXU utilization: **67.4%**
- Peak HBM: **30.93 GiB** / 31.25 GiB
- Stack reservation: **23.0797 GiB**
- Heap allocation: **7.8483 GiB**
- Free memory at peak: **0.3181 GiB**

Top train-step op buckets across all hosts:

| Bucket | Time | Share |
|--------|------|-------|
| convolution fusion | 56,784.0 ms | 53.7% |
| custom-call | 29,416.5 ms | 27.8% |
| loop fusion | 10,694.8 ms | 10.1% |

## HLO Dump

Optimized train-step HLO:

- Local copy: `/tmp/qwen3-v240-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256: `774cf325442f03b866bd63d2955e5f2709bfa292642d319b54e8a2887066121b`
- Size: **19,909,379 bytes**
- Lines: **119,347**

The HLO is distinct from v236/v238
(`d4d46797cc4508667e0aad8b0a75bf6485f046806db6ced027a159f89ee80b5c`,
19,909,611 bytes) and from v239
(`6ca611a629704d2fbc70da3a9b34244b60a11c19d4418dade3ee3d86474734ef`,
19,838,938 bytes).

## Verdict

**Refuted / near-tie.** Raising scoped VMEM from 100352 KiB to 100864 KiB on
the post-SparseCore RS3+AR2 stack selected a distinct HLO and produced a strong
late window, but full-window throughput (**59,714 tok/s**) and profile step
time (**4404.1 ms**) remained below the v230/v236 frontier band. Combined with
v239's low-side regression, keep `--xla_tpu_scoped_vmem_limit_kib=100352` for
the RS3+AR2 schedule.
