---
title: "Qwen3 JAX v225 - no-scan SparseCore RS/AR with RS latency multiplier 3"
type: experiment
hypothesis: "The confirmed v220/v221 frontier is sensitive to reduce-scatter SparseCore scheduling; using the TPU recipe reduce-scatter latency multiplier may improve runtime scheduling without changing HLO or enabling invalid RS variants."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs-ar-rslat3
status: provisional-supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, all-gather, latency-model, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v225 - no-scan SparseCore RS/AR with RS latency multiplier 3

## Hypothesis under test

**Hypothesis**: TPU recipe stacks often set
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3` with SparseCore
reduce-scatter offload. The current v220/v221 frontier depends on RS offload but
uses the default latency model. Raising the RS latency multiplier may improve
latency-hiding decisions without enabling v223's invalid ND path or v224's
regressing RS v2 path.

**Mechanism**: Keep v220/v221 fixed and add only
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3`.

**Predicted signal**: Support requires clean compile/loss and throughput above
v220/v221, preferably with lower `async-done`, lower `custom fusion`, or higher
MXU in XProf.

**Falsification criterion**: Compile/runtime failure, invalid loss, unchanged
throughput, or regression below v220/v221.

## Setup

GKE workload `alekseyv-qwen3-v225-sc-rsar-rslat3`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v225-no-scan-sparsecore-rs-ar-rs-latency3`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an XLA runtime scheduler-cost flag isolate
and preserves model math.

## Result

Completed cleanly as workload `alekseyv-qwen3-v225-sc-rsar-rslat3` with
`EXIT_CODE=0` on both workers. Loss stayed normal and ended at 12.0459.

- Worker0 pod: `alekseyv-qwen3-v225-sc-rsar-rslat3-slice-job-0-0-9jxvg` on
  `gke-tpu-46dd3e54-rkr2`
  - Avg steps 2-19: **59,713 tok/s**
  - Best step 2-19: **59,864 tok/s**
  - Avg steps 15-19: **59,757 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,747 tok/s**
  - Tok/s/chip from full average: **7,464**
  - Nominal MFU: **42.9%**
- Worker1 pod: `alekseyv-qwen3-v225-sc-rsar-rslat3-slice-job-0-1-q26vz` on
  `gke-tpu-46dd3e54-g30s`
  - Avg steps 2-19: **59,732 tok/s**
  - Best step 2-19: **59,863 tok/s**
  - Avg steps 15-19: **59,756 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,746 tok/s**
  - Tok/s/chip from full average: **7,466**
  - Nominal MFU: **42.9%**

This is above the confirmed v220/v221 band by roughly **291 tok/s** over the
best prior full-window value, or about **+0.49%**. Because the margin is small
and the HLO changed, require an exact rerun before carrying the flag.

## Profile

Profile capture completed:

- `2026-06-05-qwen3-jax-v225-no-scan-sparsecore-rs-ar-rs-latency3/2026_06_05_06_02_53`

XProf summary:

- Average step time: **4402.9 ms**
- MXU utilization: **67.5%**
- Peak memory: **30.93 GiB / 31.25 GiB**

Top buckets:

| Bucket | Time | Share | Note |
|---|---:|---:|---|
| `convolution fusion` | 56,780.7 ms | 53.7% | dominant matmul bucket |
| `custom-call` | 29,410.7 ms | 27.8% | Splash attention |
| `loop fusion` | 10,683.8 ms | 10.1% | bandwidth-heavy fusions |
| `custom fusion` | 3,347.8 ms | 3.2% | higher than v221, but wall time improves |
| `data formatting` | 2,900.9 ms | 2.7% | copies/formatting |
| `reduce` | 992.9 ms | 0.9% | reduction bucket |
| `async-done` | 620.8 ms | 0.6% | much lower than v221's 1,093.0 ms / 1.0% |
| `all-reduce-scatter fusion` | 311.8 ms | 0.3% | flat share |
| `all-gather` | 230.6 ms | 0.2% | flat share |

The main supporting signal is the lower step time and much lower `async-done`
bucket, while MXU remains comparable to v220.

## HLO Dump

- Optimized train-step HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v225-no-scan-sparsecore-rs-ar-rs-latency3/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: **19,909,611 bytes**
- SHA256:
  `50dae614cf9c7f4cb48f50ed3bee40310a972e141d58476da21b951462d66b88`
- Memory report total bytes used: **29,704,637,908 bytes** / **27.66 GiB**

The optimized train-step HLO changed from the v220/v221/v222/v224 hash
`a33c7940efea974ff9a7dcf2704fa4d08f79e5d82e1821a59a553805ceca81d5`.

## Verdict

**Provisional supported.** The reduce-scatter SparseCore latency multiplier
selects a different optimized HLO and improves the measured full-window
throughput to **59,732 tok/s / 42.9% MFU** with a lower profiled step time
(**4402.9 ms**) and a much smaller `async-done` bucket. Exact rerun is required
before carrying `--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3` as
the confirmed frontier.
