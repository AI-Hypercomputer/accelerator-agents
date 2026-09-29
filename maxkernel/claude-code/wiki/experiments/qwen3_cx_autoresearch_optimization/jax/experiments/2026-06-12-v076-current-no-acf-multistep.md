---
title: "Qwen3 JAX current v076 - explicit-SiLU stack without ACF multistep"
type: experiment
hypothesis: "The current explicit-SiLU shmem90 HLO may benefit from disabling async collective fusion multiple-steps scheduling while keeping the individual collective fusion subtypes enabled."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, async-collective-fusion, scheduler, compile-cache, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v076 - explicit-SiLU stack without ACF multistep

## Hypothesis under test

**Hypothesis**: Older pre-current probes found
`--xla_tpu_enable_async_collective_fusion_multiple_steps=false` harmful, but
they were run before the current explicit-SiLU shmem90 HLO and SparseCore
RS3+AR2 schedule. On the current stack, disabling only the multi-step ACF path
may reduce scheduler pressure while preserving all-gather/reduce-scatter fusion.

**Mechanism**: Rerun the v067/v068 explicit-SiLU image on np0 with a fresh
compile cache and change only:

```text
--xla_tpu_enable_async_collective_fusion_multiple_steps=false
```

Everything else stays fixed: no-scan/remat, activation sharding, MaxText CE,
Tokamax Splash max-logit 30, batch size 4 per chip, `fsdp=8,tp=1`,
all-gather and reduce-scatter async collective fusion enabled, all-reduce ACF
disabled, SparseCore reduce-scatter/all-reduce offload with aggregator, RS3+AR2,
scoped VMEM 100352, shared-memory limit 90, scheduler rerun 3, normal offload
tracing, and np0 placement.

**Predicted signal**: Support requires clean finite loss and either throughput
above the current durable **59.33k tok/s / 4433.8 ms** band or a lower XProf
step time/profile breakdown with unchanged HBM. Same/lower throughput and
same/slower profile refutes the probe.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v076-nomstep`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v076-v067-no-acf-multistep`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v076_no_acf_multistep`
- Placement: nodepool `<your-cluster>-np-0`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Status

Completed cleanly after v075 refuted disabling reduce-scatter ACF as an
identical-HLO profile regression.

## Results

Worker summaries:

- `gke-tpu-0d0013c1-l5jf`: **59,190 tok/s**, **7,399 tok/s/chip**,
  **42.4%** old MFU, about **48.26% MaxText-style MFU**.
- `gke-tpu-0d0013c1-9rkl`: **59,208 tok/s**, **7,401 tok/s/chip**,
  **42.5%** old MFU, about **48.28% MaxText-style MFU**.

XProf
`2026-06-12-qwen3-jax-v076-v067-no-acf-multistep/2026_06_12_16_31_03`
reported **4442.5 ms** average step time, **1.2 ms** stddev, **66.8% MXU**,
and **0.3%** idle. Peak HBM dropped slightly to **28.96 GiB / 31.25 GiB**,
leaving **2.2829 GiB** free.

Top XProf buckets:

- `convolution fusion`: **60,745.317 ms / 57.0%**
- `custom-call`: **29,461.243 ms / 27.6%**
- `loop fusion`: **10,853.464 ms / 10.2%**

Fresh train-step HLO was copied to `/tmp/qwen3-v076-hlo` and changed from the
v068/v071/v073/v075 family:

- SHA256:
  `9fd4c99f6f83890d3cf14e9cfb83aec092c0f4d8bf55b0637307053b2c15e97d`
- Size: **13,793,666 bytes / 79,880 lines**
- Counts: `all-gather=1489`, `all-reduce=158`, `reduce-scatter=1766`,
  `async-start=253`, `async-done=253`, `convolution=2136`,
  `custom-call=870`, `copy=7100`, `fusion=10394`, `dot_general=5451`,
  `splash=2211`, `tokamax=2`, `jvp_jit_silu__=0`, `sigmoid=0`,
  `logistic=0`, `exponential=183`.

## Verdict

Refuted. Disabling ACF multiple-steps selects a much smaller, materially
different HLO and saves about **0.07 GiB** peak HBM, but it slows the device
profile to **4442.5 ms** and lowers host throughput to **59.21k tok/s**. Keep
`--xla_tpu_enable_async_collective_fusion_multiple_steps=true`.
