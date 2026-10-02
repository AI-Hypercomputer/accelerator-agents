---
title: "Qwen3 JAX v227 - no-scan SparseCore RS/AR with RS latency multiplier 2"
type: experiment
hypothesis: "A smaller reduce-scatter SparseCore latency multiplier may keep the lower async-done schedule seen in v225/v226 while avoiding the weaker convolution-fusion timing in the v226 rerun."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs-ar-rslat2
status: provisional-supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, all-gather, latency-model, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v227 - no-scan SparseCore RS/AR with RS latency multiplier 2

## Hypothesis under test

**Hypothesis**: v225/v226 show that increasing
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier` from the default to
`3` selects a new HLO and lowers `async-done`, but v226 did not reproduce the
full v225 throughput high-water mark. A smaller multiplier of `2` may keep the
beneficial RS/AR scheduling change with less secondary pressure on convolution
fusion.

**Mechanism**: Keep the confirmed v220/v221 stack fixed and change only
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=2`.

**Predicted signal**: Support requires clean compile/loss and throughput above
the conservative RS-latency3 rerun value (**59,519 tok/s**) or above v225's
best-observed **59,732 tok/s**, with low `async-done` and no HLO/memory
regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, throughput
at or below v220/v221/v226, or XProf showing that lowering the multiplier loses
the `async-done` reduction.

## Setup

GKE workload `alekseyv-qwen3-v227-sc-rsar-rslat2`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v227-no-scan-sparsecore-rs-ar-rs-latency2`
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

Completed cleanly as workload `alekseyv-qwen3-v227-sc-rsar-rslat2` with
`EXIT_CODE=0` on both workers. Loss stayed normal and ended at 12.0459.

- Worker0 pod: `alekseyv-qwen3-v227-sc-rsar-rslat2-slice-job-0-0-hj86n` on
  `gke-tpu-46dd3e54-rkr2`
  - Avg steps 2-19: **59,640 tok/s**
  - Best step 2-19: **59,828 tok/s**
  - Avg steps 15-19: **59,706 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,675 tok/s**
  - Tok/s/chip from full average: **7,455**
  - Nominal MFU: **42.8%**
- Worker1 pod: `alekseyv-qwen3-v227-sc-rsar-rslat2-slice-job-0-1-5cpzt` on
  `gke-tpu-46dd3e54-g30s`
  - Avg steps 2-19: **59,661 tok/s**
  - Best step 2-19: **59,828 tok/s**
  - Avg steps 15-19: **59,704 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,673 tok/s**
  - Tok/s/chip from full average: **7,458**
  - Nominal MFU: **42.9%**

This beats the v226 conservative RS-latency3 confirmation by **142 tok/s** and
is close to v225's best-observed **59,732 tok/s** high-water mark. Because the
HLO changed from both v220/v221 and v225/v226, exact rerun is required before
carrying latency2.

## Profile

Profile capture completed:

- `2026-06-05-qwen3-jax-v227-no-scan-sparsecore-rs-ar-rs-latency2/2026_06_05_06_27_10`

XProf summary:

- Average step time: **4406.7 ms**
- MXU utilization: **67.4%**
- Peak memory: **30.93 GiB / 31.25 GiB**

Top buckets:

| Bucket | Time | Share | Note |
|---|---:|---:|---|
| `convolution fusion` | 56,771.1 ms | 53.7% | lower than v226 |
| `custom-call` | 29,417.5 ms | 27.8% | Splash attention |
| `loop fusion` | 10,691.1 ms | 10.1% | flat |
| `custom fusion` | 3,324.7 ms | 3.1% | between v225 and v226 |
| `data formatting` | 2,919.3 ms | 2.8% | flat |
| `reduce` | 1,002.9 ms | 0.9% | flat |
| `async-done` | 722.5 ms | 0.7% | lower than v221, higher than latency3 rerun |
| `all-reduce-scatter fusion` | 309.9 ms | 0.3% | flat |
| `all-gather` | 231.7 ms | 0.2% | flat |

The latency2 profile sits between v225/v226 latency3 and v220/v221: it keeps
much of the `async-done` reduction while improving convolution-fusion time over
the weaker latency3 rerun.

## HLO Dump

- Optimized train-step HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v227-no-scan-sparsecore-rs-ar-rs-latency2/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: **19,921,350 bytes**
- Lines: **119,391**
- SHA256:
  `247229cca7ccc50e7d695af1efe8f798dbc119cabe31b220617e3c7f7d3f2dd1`

The optimized train-step HLO differs from both v220/v221
(`a33c7940...`) and v225/v226 latency3 (`50dae614...`).

## Verdict

**Provisional supported.** The RS latency multiplier `2` selects a third HLO
variant and reaches **59,661 tok/s / 42.9% MFU**, beating the conservative
latency3 rerun and nearly matching v225's high-water run. Exact rerun before
carry; if confirmed, prefer latency2 over latency3 because it is faster in the
full-window and late-window stdout metrics while retaining a lower `async-done`
bucket than v220/v221.
