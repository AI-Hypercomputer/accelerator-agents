---
title: "Qwen3 JAX v230 - rerun RS latency2 plus all-reduce latency multiplier 2"
type: experiment
hypothesis: "Exact rerun of v229 will reproduce the all-reduce latency multiplier 2 gain and confirm the RS2+AR2 latency-model stack."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs-ar-rslat2-arlat2-rerun
status: supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, all-gather, latency-model, rerun, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v230 - rerun RS latency2 plus all-reduce latency multiplier 2

## Hypothesis under test

**Hypothesis**: v229's RS2+AR2 latency-model stack is a durable improvement over
confirmed v227/v228 RS2-only. An exact rerun should reproduce the **59.7k tok/s
/ 42.9% MFU** band with normal loss, the same HLO hash, and profile step time
near **4402.4 ms**.

**Mechanism**: Exactly rerun v229: keep confirmed RS latency multiplier `2` and
add only `--xla_tpu_sparse_core_all_reduce_latency_multiplier=2`.

**Predicted signal**: Support requires clean compile/loss, full-window
throughput above v228's **59,681 tok/s** confirmed value, and optimized
train-step HLO hash
`8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`.

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO drift
without a clear reason, or throughput returning to the v227/v228 RS2-only band.

## Setup

GKE workload `alekseyv-q3-v230-rs2-ar2-r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v230-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency2-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an exact XLA runtime flag rerun of v229 and
preserves model math.

## Result

Completed cleanly as workload `alekseyv-q3-v230-rs2-ar2-r` with `EXIT_CODE=0`
on both workers. Loss stayed normal and ended at 12.0459.

- Worker0 pod: `alekseyv-q3-v230-rs2-ar2-r-slice-job-0-0-xrhnq` on
  `gke-tpu-46dd3e54-g30s`
  - Avg steps 2-19: **59,749 tok/s**
  - Best step 2-19: **59,917 tok/s**
  - Avg steps 15-19: **59,778 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,762 tok/s**
  - Tok/s/chip from full average: **7,469**
  - Nominal MFU: **42.9%**
- Worker1 pod: `alekseyv-q3-v230-rs2-ar2-r-slice-job-0-1-6p9st` on
  `gke-tpu-46dd3e54-rkr2`
  - Avg steps 2-19: **59,725 tok/s**
  - Best step 2-19: **59,917 tok/s**
  - Avg steps 15-19: **59,779 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,764 tok/s**
  - Tok/s/chip from full average: **7,466**
  - Nominal MFU: **42.9%**

This exactly confirms v229's HLO and improves the best-worker full-window
measurement slightly. RS2+AR2 is now the strongest confirmed stack.

## Profile

Profile capture completed:

- `2026-06-05-qwen3-jax-v230-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency2-rerun/2026_06_05_06_59_59`

XProf summary:

- Average step time: **4400.7 ms**
- MXU utilization: **67.5%**
- Peak memory: **30.93 GiB / 31.25 GiB**

Top buckets:

| Bucket | Time | Share | Note |
|---|---:|---:|---|
| `convolution fusion` | 56,766.9 ms | 53.7% | comparable to v229 |
| `custom-call` | 29,411.7 ms | 27.8% | Splash attention |
| `loop fusion` | 10,701.4 ms | 10.1% | flat |
| `custom fusion` | 3,296.4 ms | 3.1% | slightly lower than v229 |
| `data formatting` | 2,910.1 ms | 2.8% | flat |
| `reduce` | 991.3 ms | 0.9% | flat |
| `async-done` | 626.4 ms | 0.6% | matches v229, below v227/v228 |
| `all-reduce-scatter fusion` | 310.4 ms | 0.3% | flat |
| `all-gather` | 226.9 ms | 0.2% | flat |

## HLO Dump

- Optimized train-step HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v230-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency2-rerun/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: **19,925,715 bytes**
- Lines: **119,437**
- SHA256:
  `8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`

The optimized train-step HLO is byte-identical to v229.

## Verdict

**Supported.** v230 confirms v229's RS2+AR2 latency-model stack with the same
HLO, stronger full-window throughput (**59,749 tok/s / 42.9% MFU**), and lower
profile step time (**4400.7 ms**) than v228. Carry
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=2` and
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=2` as the confirmed
native-JAX frontier.
