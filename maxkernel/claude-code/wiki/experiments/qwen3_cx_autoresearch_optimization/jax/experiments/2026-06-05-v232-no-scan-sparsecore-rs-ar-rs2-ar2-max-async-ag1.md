---
title: "Qwen3 JAX v232 - RS2+AR2 with max concurrent async all-gathers 1"
type: experiment
hypothesis: "The confirmed frontier keeps all-gather on async collective fusion; limiting concurrent async all-gathers may reduce residual all-gather/async-done scheduling pressure without changing the SparseCore RS/AR offload split."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs2-ar2-maxag1
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, all-gather, async-collective-fusion, concurrency, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v232 - RS2+AR2 with max concurrent async all-gathers 1

## Hypothesis under test

**Hypothesis**: v230's confirmed frontier returns all-gather to async collective
fusion while offloading reduce-scatter/all-reduce to SparseCore. TPU recipe
stacks sometimes constrain async all-gather concurrency. Setting
`--xla_max_concurrent_async_all_gathers=1` may reduce residual all-gather or
async-done scheduling pressure without changing the confirmed RS2+AR2
SparseCore latency-model stack.

**Mechanism**: Keep v230 fixed and add only
`--xla_max_concurrent_async_all_gathers=1`.

**Predicted signal**: Support requires clean compile/loss and throughput above
v230's **59,749 tok/s** confirmed value, or a lower profile step time/all-gather
bucket with the same throughput band and no HBM regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO/memory
regression, or clean completion at or below v230's confirmed band.

## Setup

GKE workload `alekseyv-q3-v232-rs2-ar2-ag1`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v232-no-scan-sparsecore-rs-ar-rs2-ar2-max-async-ag1`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an XLA runtime scheduling flag isolate and
preserves model math.

## Result

Completed cleanly as workload `alekseyv-q3-v232-rs2-ar2-ag1` with
`EXIT_CODE=0` on both workers and normal loss ending at 12.0459.

- Worker0 pod: `alekseyv-q3-v232-rs2-ar2-ag1-slice-job-0-0-76qb4`
  averaged **59,730 tok/s** over steps 2-19, **59,766 tok/s** over steps
  15-19, and **59,745 tok/s** excluding profiled steps. Best post-warmup step:
  **59,894 tok/s**.
- Worker1 pod: `alekseyv-q3-v232-rs2-ar2-ag1-slice-job-0-1-h9mn9`
  averaged **59,707 tok/s** over steps 2-19, **59,766 tok/s** over steps
  15-19, and **59,745 tok/s** excluding profiled steps. Best post-warmup step:
  **59,897 tok/s**.

This is inside the confirmed v230 band but below v230's best-worker
**59,749 tok/s** full-window average.

## Profile

XProf run:
`2026-06-05-qwen3-jax-v232-no-scan-sparsecore-rs-ar-rs2-ar2-max-async-ag1/2026_06_05_07_22_59`.

- Step time: **4402.4 ms** average.
- MXU utilization: **67.5%**.
- Peak HBM: **30.93 GiB**.
- The op profile remains materially matched to v229/v230; `convolution fusion`
  is **56,779.4 ms / 53.7%** and `data formatting` is
  **2,910.0 ms / 2.8%**.

## HLO Dump

Optimized train-step HLO is byte-identical to v229/v230:

- SHA256:
  `8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`
- Size: **19,925,715 bytes**
- Lines: **119,437**

## Verdict

Refuted/tie. `--xla_max_concurrent_async_all_gathers=1` did not improve
throughput, profile step time, HBM, or MXU on the confirmed RS2+AR2 frontier.
Do not carry this cap; keep v230 as the confirmed native-JAX frontier.
