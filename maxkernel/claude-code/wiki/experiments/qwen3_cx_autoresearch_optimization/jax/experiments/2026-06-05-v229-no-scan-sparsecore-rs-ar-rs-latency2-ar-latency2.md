---
title: "Qwen3 JAX v229 - RS latency2 plus all-reduce latency multiplier 2"
type: experiment
hypothesis: "All-reduce SparseCore offload may still be under-modeled after confirming reduce-scatter latency2; increasing the all-reduce latency multiplier may improve overlap without changing model math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs-ar-rslat2-arlat2
status: provisional-supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, all-gather, latency-model, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v229 - RS latency2 plus all-reduce latency multiplier 2

## Hypothesis under test

**Hypothesis**: v227/v228 confirm
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=2` as the best
confirmed RS latency-model setting. XProf's SparseCore offload guidance also
identifies `xla_tpu_sparse_core_all_reduce_latency_multiplier` for under-modeled
SparseCore all-reduce work. Since all-reduce remains offloaded in the confirmed
frontier, setting the all-reduce multiplier to `2` may improve overlap or reduce
residual async waiting.

**Mechanism**: Keep v228 fixed and add only
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=2`.

**Predicted signal**: Support requires clean compile/loss and throughput above
v228's **59,681 tok/s** confirmed value, or lower profile step time/async wait
with the same throughput band and no HBM regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, throughput
regression below v227/v228, HBM pressure, or no measurable profile improvement.

## Setup

GKE workload `alekseyv-q3-v229-rs2-ar2`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v229-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency2`
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

Completed cleanly as workload `alekseyv-q3-v229-rs2-ar2` with `EXIT_CODE=0`
on both workers. Loss stayed normal and ended at 12.0459.

- Worker0 pod: `alekseyv-q3-v229-rs2-ar2-slice-job-0-0-z9d7h` on
  `gke-tpu-46dd3e54-g30s`
  - Avg steps 2-19: **59,738 tok/s**
  - Best step 2-19: **59,902 tok/s**
  - Avg steps 15-19: **59,780 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,752 tok/s**
  - Tok/s/chip from full average: **7,467**
  - Nominal MFU: **42.9%**
- Worker1 pod: `alekseyv-q3-v229-rs2-ar2-slice-job-0-1-j4xdd` on
  `gke-tpu-46dd3e54-rkr2`
  - Avg steps 2-19: **59,721 tok/s**
  - Best step 2-19: **59,906 tok/s**
  - Avg steps 15-19: **59,782 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **59,754 tok/s**
  - Tok/s/chip from full average: **7,465**
  - Nominal MFU: **42.9%**

This is the best full-window result observed so far and slightly exceeds v225's
previous high-water mark (**59,732 tok/s**). Because the HLO changed, exact
rerun is required before carrying the all-reduce latency multiplier.

## Profile

Profile capture completed:

- `2026-06-05-qwen3-jax-v229-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency2/2026_06_05_06_49_16`

XProf summary:

- Average step time: **4402.4 ms**
- MXU utilization: **67.5%**
- Peak memory: **30.93 GiB / 31.25 GiB**

Top buckets:

| Bucket | Time | Share | Note |
|---|---:|---:|---|
| `convolution fusion` | 56,772.1 ms | 53.7% | comparable to v227 |
| `custom-call` | 29,411.6 ms | 27.8% | Splash attention |
| `loop fusion` | 10,702.8 ms | 10.1% | flat |
| `custom fusion` | 3,310.7 ms | 3.1% | flat |
| `data formatting` | 2,910.3 ms | 2.8% | flat |
| `reduce` | 991.3 ms | 0.9% | slightly lower than v227 |
| `async-done` | 632.8 ms | 0.6% | below v227 and v220/v221 |
| `all-reduce-scatter fusion` | 310.7 ms | 0.3% | flat |
| `all-gather` | 229.1 ms | 0.2% | flat |

## HLO Dump

- Optimized train-step HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v229-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency2/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: **19,925,715 bytes**
- Lines: **119,437**
- SHA256:
  `8906838f769ee38abcec2c0498a68eace061c517c362b19e7efa7e4cc5854278`

The optimized train-step HLO differs from v227/v228 latency2
(`247229cca7...`), so the all-reduce latency multiplier changes scheduling.

## Verdict

**Provisional supported.** Adding
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=2` on top of confirmed
RS-latency2 selects a new HLO and reaches **59,738 tok/s / 42.9% MFU**, the
best full-window result so far, with lower profile step time than v228. Exact
rerun before carry.
