---
title: "Qwen3 JAX v222 - no-scan SparseCore RS/AR-only without aggregator"
type: experiment
hypothesis: "After v220/v221 returned all-gather to async collective fusion, the SparseCore collective aggregator may no longer be needed; disabling it could reduce scheduling overhead or confirm that it still helps RS/AR offload."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs-ar-noaggregator
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, all-gather, ablation, collective-aggregator, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v222 - no-scan SparseCore RS/AR-only without aggregator

## Hypothesis under test

**Hypothesis**: The v220/v221 frontier keeps reduce-scatter/all-reduce on
SparseCore, returns all-gather to async collective fusion, and still carries
`--xla_tpu_enable_sparse_core_collective_aggregator=true`. Once all-gather is no
longer offloaded, the aggregator may be neutral or harmful; disabling only this
flag should either preserve the frontier or prove that the aggregator still
benefits the RS/AR-only offload stack.

**Mechanism**: Keep v220/v221 fixed except set
`--xla_tpu_enable_sparse_core_collective_aggregator=false`.

**Predicted signal**: Support requires clean compile/loss and throughput at or
above the v220/v221 confirmed band, with profile evidence that async-done and
collective buckets do not regress.

**Falsification criterion**: Compile/runtime failure, invalid loss, or
throughput below the v220/v221 band. A regression means keep the aggregator
enabled for the RS/AR-only frontier.

## Setup

GKE workload `alekseyv-qwen3-v222-sc-rsar-noagg`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v222-no-scan-sparsecore-rs-ar-no-aggregator`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is a runtime flag isolate; model math should
remain byte-identical from the JAX program's perspective.

## Result

Completed cleanly as workload `alekseyv-qwen3-v222-sc-rsar-noagg` with
`EXIT_CODE=0` on both workers. Loss matched prior runs and ended at 12.0459.

- Worker0 pod: `alekseyv-qwen3-v222-sc-rsar-noagg-slice-job-0-0-gbkn5` on
  `gke-tpu-0d0013c1-9rkl`
  - Avg steps 2-19: **59,246 tok/s**
  - Best step 2-19: **59,355 tok/s**
  - Avg steps 15-19: **59,229 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,261 tok/s**
  - Tok/s/chip from full average: **7,406**
  - Nominal MFU: **42.5%**
- Worker1 pod: `alekseyv-qwen3-v222-sc-rsar-noagg-slice-job-0-1-bttlj` on
  `gke-tpu-0d0013c1-l5jf`
  - Avg steps 2-19: **59,213 tok/s**
  - Best step 2-19: **59,354 tok/s**
  - Avg steps 15-19: **59,231 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,260 tok/s**
  - Tok/s/chip from full average: **7,402**
  - Nominal MFU: **42.5%**

This is below v220/v221. The no-trace window is roughly **200 tok/s** behind
v221's no-trace window, and late steps are roughly **247 tok/s** behind
v221's late-step window.

## Profile

- XProf run:
  `2026-06-05-qwen3-jax-v222-no-scan-sparsecore-rs-ar-no-aggregator/2026_06_05_05_34_27`
- Average step time: **4439.1 ms**
- MXU utilization: **66.8%**
- Peak memory: **30.93 GiB / 31.25 GiB** (98.98%)

Top op buckets:

| Bucket | Time | Share | v221 comparison |
|---|---:|---:|---|
| `convolution fusion` | 56,837.2 ms | 53.3% | lower share/time, but lower MXU overall |
| `custom-call` | 29,431.2 ms | 27.6% | essentially flat |
| `loop fusion` | 10,741.2 ms | 10.1% | essentially flat |
| `custom fusion` | 3,355.9 ms | 3.1% | worse than v221's 2,813.3 ms |
| `data formatting` | 2,926.4 ms | 2.7% | essentially flat |
| `async-done` | 1,340.1 ms | 1.3% | worse than v221's 1,093.0 ms |
| `all-reduce-scatter fusion` | 310.7 ms | 0.3% | flat |
| `all-gather` | 214.5 ms | 0.2% | flat/slightly lower |

The profile supports the stdout regression: disabling the aggregator raises
step time by about **10.1 ms** versus v221 and drops MXU by **0.6 pp**.

## HLO Dump

- Optimized train-step HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v222-no-scan-sparsecore-rs-ar-no-aggregator/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: **19,954,794 bytes**
- SHA256:
  `a33c7940efea974ff9a7dcf2704fa4d08f79e5d82e1821a59a553805ceca81d5`

The HLO text is byte-identical to v220/v221. The regression is therefore a
runtime scheduling effect from the aggregator flag, not a graph rewrite.

## Verdict

**Refuted.** Disabling the SparseCore collective aggregator on top of the
v220/v221 RS/AR-only SparseCore offload stack regresses throughput and profile
step time while preserving identical HLO. Keep
`--xla_tpu_enable_sparse_core_collective_aggregator=true` in the current
native-JAX frontier.
