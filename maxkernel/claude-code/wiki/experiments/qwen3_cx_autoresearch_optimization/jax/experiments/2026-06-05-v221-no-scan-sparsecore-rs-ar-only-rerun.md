---
title: "Qwen3 JAX v221 - no-scan SparseCore RS/AR-only offload rerun"
type: experiment
hypothesis: "v220 produced a small provisional gain by disabling all-gather SparseCore offload; an exact rerun must confirm the gain before carrying it."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs-ar-only-rerun
status: supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, all-gather, reduce-scatter, all-reduce, ablation, collective-aggregator, rerun, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v221 - no-scan SparseCore RS/AR-only offload rerun

## Hypothesis under test

**Hypothesis**: [v220](2026-06-05-v220-no-scan-sparsecore-rs-ar-only.md)
is a real runtime-level improvement over v208/v210, not run noise. It should
rerun at or above the confirmed 59,370-59,372 tok/s band and near v220's
59,441 tok/s average.

**Mechanism**: Exact rerun of v220: keep reduce-scatter/all-reduce SparseCore
offload plus SparseCore collective aggregator, keep all-gather on async
collective fusion, and preserve all trainer shape/runtime flags.

**Predicted signal**: Support requires clean compile/loss and best-worker
throughput above the confirmed v210 band, preferably within noise of v220's
59,441 tok/s average. XProf should preserve the lowered async-done bucket
relative to v210.

**Falsification criterion**: Compile/runtime failure, invalid loss, throughput
falling back to or below the v208/v210 band, or profile evidence that the v220
gain was a one-run artifact.

## Setup

GKE workload `alekseyv-qwen3-v221-sc-rsar-r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v221-no-scan-sparsecore-rs-ar-only-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an exact runtime rerun of v220 and
preserves model math.

## Result

Completed cleanly as workload `alekseyv-qwen3-v221-sc-rsar-r` with `EXIT_CODE=0`
on both workers. Loss followed the v220/v210 trace and ended at 12.0459.

- Worker0 pod: `alekseyv-qwen3-v221-sc-rsar-r-slice-job-0-0-b9s58` on
  `gke-tpu-964065d9-c180`
  - Avg steps 2-19: **59,435 tok/s**
  - Best step 2-19: **59,651 tok/s**
  - Avg steps 15-19: **59,478 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,463 tok/s**
  - Tok/s/chip from full average: **7,429**
  - Nominal MFU: **42.7%**
- Worker1 pod: `alekseyv-qwen3-v221-sc-rsar-r-slice-job-0-1-2gzpj` on
  `gke-tpu-964065d9-9vh8`
  - Avg steps 2-19: **59,406 tok/s**
  - Best step 2-19: **59,650 tok/s**
  - Avg steps 15-19: **59,478 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,462 tok/s**
  - Tok/s/chip from full average: **7,426**
  - Nominal MFU: **42.6%**

The full-window average is slightly below v220 because worker1 step 12 slowed
during trace capture (`58,772 tok/s`), but the non-profiled and late-step
windows reproduce v220's operating band. This confirms v220's runtime
configuration as the current native-JAX frontier, while v220 remains the single
best measured full-window run.

## Profile

- XProf run:
  `2026-06-05-qwen3-jax-v221-no-scan-sparsecore-rs-ar-only-rerun/2026_06_05_05_19_51`
- Profile files:
  - `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v221-no-scan-sparsecore-rs-ar-only-rerun/plugins/profile/2026_06_05_05_19_51/gke-tpu-964065d9-c180.{trace.json.gz,xplane.pb}`
  - `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v221-no-scan-sparsecore-rs-ar-only-rerun/plugins/profile/2026_06_05_05_19_51/gke-tpu-964065d9-9vh8.{trace.json.gz,xplane.pb}`
- XProf summary:
  - Average step time: **4429.0 ms**
  - MXU utilization: **67.4%**
  - Train-step program time: **106,069.7 ms**
  - Total profile time: **106,385.2 ms**
  - Idle: **314.6 ms / 0.3%**
  - Peak memory: **30.93 GiB / 31.25 GiB** (98.98%)

Top op buckets:

| Bucket | Time | Share | Notes |
|---|---:|---:|---|
| `convolution fusion` | 57,380.8 ms | 53.9% | matmul-dominant path |
| `custom-call` | 29,419.5 ms | 27.7% | Splash attention |
| `loop fusion` | 10,748.2 ms | 10.1% | bandwidth-heavy elementwise/reductions |
| `data formatting` | 2,939.0 ms | 2.8% | copies/formatting |
| `custom fusion` | 2,813.3 ms | 2.6% | non-matmul fusions |
| `async-done` | 1,093.0 ms | 1.0% | remains far below v210's all-SC-offload async-done bucket |
| `all-reduce-scatter fusion` | 310.6 ms | 0.3% | aggregate collective bucket |
| `all-gather` | 225.5 ms | 0.2% | direct all-gather after disabling AG SC offload |

Profile matches v220 within noise: v220 step time was 4428.4 ms, MXU 67.5%,
peak HBM 30.93 GiB, and train-step program time 106,044.4 ms.

## HLO Dump

- Optimized train-step HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v221-no-scan-sparsecore-rs-ar-only-rerun/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: **19,954,794 bytes**
- SHA256:
  `a33c7940efea974ff9a7dcf2704fa4d08f79e5d82e1821a59a553805ceca81d5`

The optimized HLO hash is byte-identical to v220, as expected for an exact
runtime rerun.

## Verdict

**Supported.** v221 confirms that v220 was not a one-off compile/runtime
artifact: the same HLO, same profile shape, and non-profiled/late-step
throughput reproduce the 59.46-59.48k tok/s band. Carry the v220/v221
configuration as the native-JAX frontier:

- reduce-scatter SparseCore offload enabled
- all-reduce SparseCore offload enabled
- all-gather SparseCore offload disabled
- all-gather async collective fusion enabled
- SparseCore collective aggregator enabled

The best observed full-window result remains v220 at **59,441 tok/s / 42.7%
MFU** because v221's traced step depressed worker1's full-window average.
Next isolate: keep this collective split and disable only
`--xla_tpu_enable_sparse_core_collective_aggregator` to measure whether the
aggregator is still beneficial after returning all-gather to async fusion.
