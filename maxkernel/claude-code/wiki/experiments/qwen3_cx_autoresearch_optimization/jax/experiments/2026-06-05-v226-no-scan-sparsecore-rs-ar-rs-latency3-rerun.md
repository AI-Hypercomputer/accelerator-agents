---
title: "Qwen3 JAX v226 - rerun no-scan SparseCore RS/AR with RS latency multiplier 3"
type: experiment
hypothesis: "Exact rerun of v225 will reproduce the RS latency multiplier 3 gain, confirming it as a durable frontier rather than noise."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs-ar-rslat3-rerun
status: supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, all-gather, latency-model, rerun, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v226 - rerun no-scan SparseCore RS/AR with RS latency multiplier 3

## Hypothesis under test

**Hypothesis**: v225's
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3` result is a real
runtime/HLO scheduling improvement over v220/v221. An exact rerun should
reproduce the **59.7k tok/s / 42.9% MFU** band with normal loss, comparable
profile, and the same optimized train-step HLO hash.

**Mechanism**: Exactly rerun v225: keep v220/v221 fixed and add only
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3`.

**Predicted signal**: Support requires clean compile/loss, full-window
throughput in the v225 band, profile step time near **4402.9 ms**, and optimized
train-step HLO hash
`50dae614cf9c7f4cb48f50ed3bee40310a972e141d58476da21b951462d66b88`.

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO drift
without a clear reason, or throughput returning to the confirmed v220/v221
**59.4k tok/s / 42.7% MFU** band.

## Setup

GKE workload `alekseyv-qwen3-v226-sc-rsar-rslat3-r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v226-no-scan-sparsecore-rs-ar-rs-latency3-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an exact XLA runtime flag rerun of v225 and
preserves model math.

## Result

Completed cleanly as workload `alekseyv-qwen3-v226-sc-rsar-rslat3-r` with
`EXIT_CODE=0` on both workers. Loss stayed normal and ended at 12.0459.

- Worker0 pod: `alekseyv-qwen3-v226-sc-rsar-rslat3-r-slice-job-0-0-klkzq` on
  `gke-tpu-964065d9-c180`
  - Avg steps 2-19: **59,519 tok/s**
  - Best step 2-19: **59,699 tok/s**
  - Avg steps 15-19: **59,578 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,544 tok/s**
  - Tok/s/chip from full average: **7,440**
  - Nominal MFU: **42.7%**
- Worker1 pod: `alekseyv-qwen3-v226-sc-rsar-rslat3-r-slice-job-0-1-wlslm` on
  `gke-tpu-964065d9-9vh8`
  - Avg steps 2-19: **59,495 tok/s**
  - Best step 2-19: **59,702 tok/s**
  - Avg steps 15-19: **59,579 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,542 tok/s**
  - Tok/s/chip from full average: **7,437**
  - Nominal MFU: **42.7%**

This confirms the RS-latency3 HLO and a throughput band above the prior
confirmed v220/v221 full-window measurements, but it does **not** reproduce
v225's high **59,732 tok/s** full-window result. Treat v225 as the best
observed single run and v226 as the conservative confirmed carry signal.

## Profile

Profile capture completed:

- `2026-06-05-qwen3-jax-v226-no-scan-sparsecore-rs-ar-rs-latency3-rerun/2026_06_05_06_15_45`

XProf summary:

- Average step time: **4422.3 ms**
- MXU utilization: **67.3%**
- Peak memory: **30.93 GiB / 31.25 GiB**

Top buckets:

| Bucket | Time | Share | Note |
|---|---:|---:|---|
| `convolution fusion` | 57,405.2 ms | 54.1% | higher than v225 |
| `custom-call` | 29,412.8 ms | 27.7% | Splash attention, unchanged |
| `loop fusion` | 10,722.6 ms | 10.1% | near v225 |
| `custom fusion` | 3,207.7 ms | 3.0% | slightly lower than v225 |
| `data formatting` | 2,907.8 ms | 2.7% | flat |
| `reduce` | 992.4 ms | 0.9% | flat |
| `async-done` | 577.3 ms | 0.5% | lower than v225 and much lower than v221 |
| `all-reduce-scatter fusion` | 312.0 ms | 0.3% | flat |
| `all-gather` | 222.7 ms | 0.2% | flat |

The profile supports the scheduling mechanism: `async-done` remains much lower
than the confirmed v220/v221 frontier. The weaker stdout result appears to come
from higher convolution-fusion time and ordinary run-to-run schedule/noise, not
from HLO drift.

## HLO Dump

- Optimized train-step HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v226-no-scan-sparsecore-rs-ar-rs-latency3-rerun/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: **19,909,611 bytes**
- Lines: **119,349**
- SHA256:
  `50dae614cf9c7f4cb48f50ed3bee40310a972e141d58476da21b951462d66b88`

The optimized train-step HLO is byte-identical to v225.

## Verdict

**Supported, but with a conservative carry value.** v226 confirms that
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3` selects the same HLO
as v225 and preserves the lower `async-done` profile. It does not confirm
v225's high-water throughput, so carry the flag as a marginal supported
frontier improvement while recording **59,732 tok/s / 42.9% MFU** as best
observed and **59,519 tok/s / 42.7% MFU** as the rerun-confirmed full-window
value.
