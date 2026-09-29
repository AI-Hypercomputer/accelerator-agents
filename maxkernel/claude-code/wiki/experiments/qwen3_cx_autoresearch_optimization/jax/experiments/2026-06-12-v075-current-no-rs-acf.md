---
title: "Qwen3 JAX current v075 - explicit-SiLU stack without reduce-scatter ACF"
type: experiment
hypothesis: "Reduce-scatter async collective fusion may be redundant under SparseCore reduce-scatter offload on the current explicit-SiLU RS3+AR2 HLO."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, reduce-scatter, async-collective-fusion, scheduler, compile-cache, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v075 - explicit-SiLU stack without reduce-scatter ACF

## Hypothesis under test

**Hypothesis**: Earlier pre-explicit-SiLU ACF subtype probes showed
reduce-scatter ACF was often redundant after collective-matmul simplification.
The current stack also uses SparseCore reduce-scatter offload, so explicitly
disabling reduce-scatter ACF may be inert or may simplify the scheduler without
changing model math.

**Mechanism**: Rerun the v067/v068 explicit-SiLU image on np0 with a fresh
compile cache and add only:

```text
--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false
```

Everything else stays fixed: no-scan/remat, activation sharding, MaxText CE,
Tokamax Splash max-logit 30, batch size 4 per chip, `fsdp=8,tp=1`,
all-gather on async collective fusion, SparseCore reduce-scatter/all-reduce
offload with aggregator, RS3+AR2, scoped VMEM 100352, shared-memory limit 90,
scheduler rerun 3, normal offload tracing, and np0 placement.

**Predicted signal**: Support requires clean finite loss and either throughput
above the current durable **59.33k tok/s / 4433.8 ms** band or a lower XProf
step time/profile breakdown with unchanged HBM. Same/lower throughput and
same/slower profile refutes the probe.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v075-norsacf`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v075-v067-no-rs-acf`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v075_no_rs_acf`
- Placement: nodepool `<your-cluster>-np-0`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Status

Completed cleanly after v074 refuted RS4+AR2 as a changed-HLO memory/profile
regression.

## Results

Worker summaries:

- `gke-tpu-0d0013c1-9rkl`: **59,299 tok/s**, **7,412 tok/s/chip**,
  **42.5%** old MFU, about **48.35% MaxText-style MFU**.
- `gke-tpu-0d0013c1-l5jf`: **59,278 tok/s**, **7,410 tok/s/chip**,
  **42.5%** old MFU, about **48.34% MaxText-style MFU**.

XProf
`2026-06-12-qwen3-jax-v075-v067-no-rs-acf/2026_06_12_16_18_44`
reported **4436.1 ms** average step time, **1.2 ms** stddev, **66.7% MXU**,
and **0.3%** idle. Peak HBM stayed at **29.03 GiB / 31.25 GiB**, leaving
**2.2202 GiB** free.

Top XProf buckets:

- `convolution fusion`: **56,941.368 ms / 53.5%**
- `custom-call`: **29,433.840 ms / 27.6%**
- `loop fusion`: **10,727 ms / 10.1%**

Fresh train-step HLO was copied to `/tmp/qwen3-v075-hlo` and was byte-identical
to v068/v071/v073:

- SHA256:
  `84df522a5e6dfdd158288d856af82a5b24506c30db8c0fb29dfb46c952378a38`
- Size: **19,895,815 bytes / 119,353 lines**
- Counts: `all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
  `async-start=394`, `async-done=394`, `convolution=2106`,
  `custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
  `splash=2211`, `tokamax=2`, `jvp_jit_silu__=0`, `sigmoid=0`,
  `logistic=0`, `exponential=183`.

## Verdict

Refuted. Disabling reduce-scatter async collective fusion is accepted but inert
under the current SparseCore reduce-scatter offload stack: the optimized HLO is
byte-identical to the current RS3+AR2 frontier, host throughput is lower than
the durable **59.33k tok/s** band, and XProf regresses to **4436.1 ms**. Keep
the default reduce-scatter ACF setting.
