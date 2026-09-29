---
title: "Qwen3 JAX v212 - same-shape MaxText comparison"
type: experiment
hypothesis: "Rerunning the v210 native-JAX frontier stack at MaxText's bs3/chip shape should preserve a throughput lead over MaxText v006 and isolate the reported MFU difference as accounting rather than wall-clock performance."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-all-collectives-aggregator-bs3
status: supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collective-aggregator, collectives, all-gather, reduce-scatter, all-reduce, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8, maxtext-comparison]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v212 - same-shape MaxText comparison

## Hypothesis under test

**Hypothesis**: The MaxText v006 reference reports higher MFU than the native
JAX v210 frontier, but it uses a smaller bs3/chip shape. Rerunning the v210
native-JAX stack at the same seq8192, global-batch-24 shape should show whether
there is an actual MaxText wall-clock throughput advantage or just a reporting
difference.

**Mechanism**: Keep the confirmed v210 no-scan SparseCore collective-aggregator
runtime stack fixed and change only per-chip batch from 4 to 3.

**Predicted signal**: Support requires native JAX at bs3/chip to exceed MaxText
v006's **55,064 tok/s aggregate**, **6,883 tok/s/chip** result at the same
sequence length and global batch.

**Falsification criterion**: Clean completion below the MaxText v006 throughput
band, invalid loss, missing profile capture, or compile/runtime failure.

## Setup

GKE workload `alekseyv-qwen3-v212-noscan-scagg-bs3`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v212-no-scan-sparsecore-aggregator-bs3-comparison`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

The runtime flags match the v210 carried stack: async all-gather enabled,
collective-matmul modes forced to `none`, sparse-core offload enabled for
all-gather/reduce-scatter/all-reduce, sparse-core collective aggregator enabled,
experimental scheduler bundle disabled, selective resources enabled,
latency-hiding scheduler rerun 3, scoped VMEM 100352 KiB, and scheduler shared
memory limit 100%.

## Equivalence

No model-code edit is made. This is a shape-only comparison run and preserves
model math.

## Results

Completed cleanly with `EXIT_CODE=0` on both workers.

- Worker 0 reported **58,970 tok/s** aggregate, **7,371 tok/s/chip**, and
  **42.3% MFU**.
- Worker 1 reported **58,940 tok/s** aggregate, **7,368 tok/s/chip**, and
  **42.3% MFU**.
- Loss decreased normally through step 19, ending at **12.0507**.

Worker0 steady steps stayed tightly clustered:
59,139, 59,086, 59,070, 59,029, 59,007, 58,997, 58,961, 58,920, 58,916,
58,878, 58,855, 58,846, 58,791, 59,082, 59,021, 58,964, 58,963, and
58,932 tok/s for steps 2-19.

This same-shape native-JAX run beats MaxText v006's **55,063.97 tok/s**
aggregate and **6,883 tok/s/chip** by about **+7.1%**.

## Profile

Profile capture completed on both hosts:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v212-no-scan-sparsecore-aggregator-bs3-comparison/plugins/profile/2026_06_05_03_19_27/gke-tpu-964065d9-c180.{trace.json.gz,xplane.pb}`
- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v212-no-scan-sparsecore-aggregator-bs3-comparison/plugins/profile/2026_06_05_03_19_28/gke-tpu-964065d9-9vh8.{trace.json.gz,xplane.pb}`

XProf runs:

- `2026-06-05-qwen3-jax-v212-no-scan-sparsecore-aggregator-bs3-comparison/2026_06_05_03_19_27`
- `2026-06-05-qwen3-jax-v212-no-scan-sparsecore-aggregator-bs3-comparison/2026_06_05_03_19_28`

XProf overview:

- c180: step time average **3349.5 ms**, stddev **2.3 ms**, MXU **66.5%**,
  TC idle **8.16 ms**, SparseCore compute **15.12 ms**.
- 9vh8: step time average **3349.5 ms**, stddev **2.3 ms**, MXU **65.9%**,
  TC idle **8.18 ms**, SparseCore compute **15.10 ms**.

Top c180 op-profile buckets:

- `convolution fusion`: **21,921.493 ms** / **54.6%**.
- `custom-call`: **11,044.289 ms** / **27.5%**.
- `loop fusion`: **4,166.509 ms** / **10.4%**.

Top 9vh8 op-profile buckets:

- `convolution fusion`: **21,671.158 ms** / **53.8%**.
- `custom-call`: **11,047.513 ms** / **27.4%**.
- `loop fusion`: **4,157.373 ms** / **10.3%**.

## HLO Dump

Fresh train-step HLO was emitted:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v212-no-scan-sparsecore-aggregator-bs3-comparison/hlo/module_0115.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `3a615983c09ec3ed3e9aba43c225c6f09f9862bd726b30549867abb60e48f292`
- HLO size: **93,855 lines**, **16,513,927 bytes**.
- Compiled memory: `24588362712` bytes / **22.90 GiB**.
- Preallocated temp: **17.15 GiB**.

Text counts:

| Pattern | Count |
|---|---:|
| `_one_hot` | 11 |
| `all-gather` | 8,288 |
| `all-reduce` | 165 |
| `convolution` | 3,251 |
| `copy(` | 1,196 |
| `custom-call` | 4,016 |
| `dot_general` | 6,355 |
| `fusion` | 17,381 |
| `reduce-scatter` | 2,772 |
| `splash` | 2,861 |

## Verdict

**Supported for comparison, not a new frontier.** At the same seq8192/global
batch 24 shape as MaxText v006, native JAX is still faster in aggregate and
per-chip token throughput: **58,970 vs 55,064 tok/s**, **7,371 vs
6,883 tok/s/chip**. The reported MFU remains lower (**42.3% vs 44.92%**),
which confirms the MaxText/JAX MFU delta is mainly accounting/formula
difference rather than a wall-clock token-throughput gap. Keep v210 as the
throughput frontier because bs4/chip remains faster overall.
