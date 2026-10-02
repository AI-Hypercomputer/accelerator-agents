---
title: "Qwen3 JAX v228 - rerun no-scan SparseCore RS/AR with RS latency multiplier 2"
type: experiment
hypothesis: "Exact rerun of v227 will reproduce the RS latency multiplier 2 gain and confirm it as the preferred latency-model setting over multiplier 3."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs-ar-rslat2-rerun
status: supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, all-gather, latency-model, rerun, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v228 - rerun no-scan SparseCore RS/AR with RS latency multiplier 2

## Hypothesis under test

**Hypothesis**: v227's
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=2` result is durable
and should replace latency3 as the carried RS latency-model setting. An exact
rerun should reproduce the **59.6k tok/s / 42.9% MFU** band with normal loss,
the same HLO hash, and a profile step time near **4406.7 ms**.

**Mechanism**: Exactly rerun v227: keep the v220/v221 collective split fixed and
set only `--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=2`.

**Predicted signal**: Support requires clean compile/loss, full-window
throughput above the conservative v226 latency3 rerun (**59,519 tok/s**) and
optimized train-step HLO hash
`247229cca7ccc50e7d695af1efe8f798dbc119cabe31b220617e3c7f7d3f2dd1`.

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO drift
without a clear reason, or throughput returning to the v220/v221/v226 band.

## Setup

GKE workload `alekseyv-qwen3-v228-sc-rsar-rslat2-r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v228-no-scan-sparsecore-rs-ar-rs-latency2-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an exact XLA runtime flag rerun of v227 and
preserves model math.

## Result

Completed cleanly as workload `alekseyv-qwen3-v228-sc-rsar-rslat2-r` with
`EXIT_CODE=0` on both workers. Loss stayed normal and ended at 12.0459.

- Worker0 pod: `alekseyv-qwen3-v228-sc-rsar-rslat2-r-slice-job-0-0-rf4n2` on
  `gke-tpu-46dd3e54-g30s`
  - Avg steps 2-19: **59,681 tok/s**
  - Best step 2-19: **59,832 tok/s**
  - Avg steps 15-19: **59,712 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,693 tok/s**
  - Tok/s/chip from full average: **7,460**
  - Nominal MFU: **42.9%**
- Worker1 pod: `alekseyv-qwen3-v228-sc-rsar-rslat2-r-slice-job-0-1-79q6w` on
  `gke-tpu-46dd3e54-rkr2`
  - Avg steps 2-19: **59,654 tok/s**
  - Best step 2-19: **59,833 tok/s**
  - Avg steps 15-19: **59,719 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,697 tok/s**
  - Tok/s/chip from full average: **7,457**
  - Nominal MFU: **42.9%**

This confirms v227's latency2 throughput band and is the strongest confirmed
full-window result so far. v225 remains the single highest full-window
measurement (**59,732 tok/s**) but was not reproduced by its latency3 rerun.

## Profile

Profile capture completed on both hosts:

- `2026-06-05-qwen3-jax-v228-no-scan-sparsecore-rs-ar-rs-latency2-rerun/2026_06_05_06_37_35`
  (`gke-tpu-46dd3e54-rkr2`)
- `2026-06-05-qwen3-jax-v228-no-scan-sparsecore-rs-ar-rs-latency2-rerun/2026_06_05_06_37_36`
  (`gke-tpu-46dd3e54-g30s`)

XProf summaries:

- Average step time: **4405.3 ms**
- MXU utilization: **66.9%** and **67.9%**
- Peak memory: **30.93 GiB / 31.25 GiB**

Top buckets for `2026_06_05_06_37_35`:

| Bucket | Time | Share | Note |
|---|---:|---:|---|
| `convolution fusion` | 27,972.3 ms | 52.8% | one-host profile |
| `custom-call` | 14,705.7 ms | 27.8% | Splash attention |
| `loop fusion` | 5,316.8 ms | 10.0% | bandwidth-heavy fusions |
| `custom fusion` | 1,981.8 ms | 3.7% | higher one-host share than v227 aggregate |
| `data formatting` | 1,451.4 ms | 2.7% | copies/formatting |
| `reduce` | 500.7 ms | 0.9% | flat |
| `async-done` | 463.3 ms | 0.9% | one-host, still below v220/v221 aggregate band |
| `all-reduce-scatter fusion` | 155.0 ms | 0.3% | flat |
| `all-gather` | 131.0 ms | 0.2% | flat |

## HLO Dump

- Optimized train-step HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v228-no-scan-sparsecore-rs-ar-rs-latency2-rerun/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: **19,921,350 bytes**
- Lines: **119,391**
- SHA256:
  `247229cca7ccc50e7d695af1efe8f798dbc119cabe31b220617e3c7f7d3f2dd1`

The optimized train-step HLO is byte-identical to v227.

## Verdict

**Supported.** v228 confirms the RS latency multiplier `2` result with the same
HLO as v227 and a slightly stronger full-window average (**59,681 tok/s /
42.9% MFU**). Carry
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=2` as the confirmed
native-JAX frontier setting over latency3. Best observed single full-window
measurement remains v225 at **59,732 tok/s**, but the strongest confirmed
repeatable stack is v227/v228 latency2.
