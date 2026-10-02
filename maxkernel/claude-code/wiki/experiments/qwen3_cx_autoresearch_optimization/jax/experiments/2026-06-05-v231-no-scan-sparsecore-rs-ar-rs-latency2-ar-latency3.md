---
title: "Qwen3 JAX v231 - RS latency2 plus all-reduce latency multiplier 3"
type: experiment
hypothesis: "After confirming RS2+AR2, a higher all-reduce SparseCore latency multiplier may further improve all-reduce overlap and reduce residual async waiting."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs-ar-rslat2-arlat3
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, all-gather, latency-model, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v231 - RS latency2 plus all-reduce latency multiplier 3

## Hypothesis under test

**Hypothesis**: v229/v230 confirm that adding
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=2` on top of confirmed RS
latency multiplier `2` improves throughput. A slightly larger all-reduce
latency multiplier of `3` may model SparseCore all-reduce latency more
accurately and further reduce residual async waiting, without changing model
math.

**Mechanism**: Keep v230 fixed and change only
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=3`.

**Predicted signal**: Support requires clean compile/loss and throughput above
v230's **59,749 tok/s** confirmed value, or a lower profile step time/async
wait with the same throughput band and no HBM regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO/memory
regression, or clean completion at or below v230's confirmed band.

## Setup

GKE workload `alekseyv-q3-v231-rs2-ar3`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v231-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency3`
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

Completed cleanly as workload `alekseyv-q3-v231-rs2-ar3` with `EXIT_CODE=0`
on both workers. Loss stayed normal and ended at 12.0459, but throughput
regressed hard relative to v230.

- Worker0 pod: `alekseyv-q3-v231-rs2-ar3-slice-job-0-0-rk5ff` on
  `gke-tpu-46dd3e54-g30s`
  - Avg steps 2-19: **57,586 tok/s**
  - Best step 2-19: **57,753 tok/s**
  - Avg steps 15-19: **57,625 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **57,598 tok/s**
  - Tok/s/chip from full average: **7,198**
  - Nominal MFU: **41.3%**
- Worker1 pod: `alekseyv-q3-v231-rs2-ar3-slice-job-0-1-8jdsg` on
  `gke-tpu-46dd3e54-rkr2`
  - Avg steps 2-19: **57,563 tok/s**
  - Best step 2-19: **57,757 tok/s**
  - Avg steps 15-19: **57,625 tok/s**
  - Avg steps 2-19 excluding profiled steps 12-14: **57,598 tok/s**
  - Tok/s/chip from full average: **7,195**
  - Nominal MFU: **41.3%**

This is roughly **2,163 tok/s** below v230's confirmed best-worker full-window
average.

## Profile

Profile capture completed:

- `2026-06-05-qwen3-jax-v231-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency3/2026_06_05_07_11_47`

XProf summary:

- Average step time: **4565.7 ms**
- MXU utilization: **66.0%**
- Peak memory: **30.15 GiB / 31.25 GiB**

Top buckets:

| Bucket | Time | Share | Note |
|---|---:|---:|---|
| `convolution fusion` | 57,717.6 ms | 52.6% | higher FLOPs/bytes than v230 |
| `custom-call` | 29,417.8 ms | 26.8% | Splash attention |
| `loop fusion` | 10,729.6 ms | 9.8% | flat |
| `all-gather` | 3,184.9 ms | 2.9% | newly exposed large bucket |
| `data formatting` | 2,902.6 ms | 2.6% | flat |
| `async-done` | 2,821.8 ms | 2.6% | much worse than v230's 626.4 ms / 0.6% |
| `custom fusion` | 1,134.1 ms | 1.0% | lower share, but not beneficial |
| `reduce` | 1,003.4 ms | 0.9% | flat |
| `all-reduce-scatter fusion` | 310.5 ms | 0.3% | flat |

The profile shows that AR latency `3` over-penalizes the offloaded all-reduce
schedule: direct `all-gather` and `async-done` become visible bottlenecks, step
time regresses by **165.0 ms** versus v230, and MXU drops.

## HLO Dump

- Optimized train-step HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v231-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency3/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: **16,297,820 bytes**
- Lines: **97,951**
- SHA256:
  `3c4126a41ca2ac948d1e2ae3aced9ba66b47fae10d95381d1d2a8d2d99f20743`

The optimized train-step HLO is much smaller and different from v229/v230
(`8906838f...`). The graph/schedule selected by AR latency `3` is not the
desired frontier family.

## Verdict

**Refuted.** Raising the all-reduce SparseCore latency multiplier from `2` to
`3` compiles and runs but regresses throughput, XProf step time, MXU, and
collective overlap. Do not carry AR latency `3`; keep v230's RS2+AR2 stack as
the confirmed frontier.
