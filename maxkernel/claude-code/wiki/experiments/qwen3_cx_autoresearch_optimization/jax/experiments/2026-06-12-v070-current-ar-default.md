---
title: "Qwen3 JAX current v070 - AR default on explicit-SiLU lean HLO"
type: experiment
hypothesis: "Dropping the SparseCore all-reduce latency multiplier on the current explicit-SiLU shmem90 stack may select a better schedule under the fresh v064-v069 runtime environment."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, all-reduce, scheduler, compile-cache, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v070 - AR default on explicit-SiLU lean HLO

## Hypothesis under test

**Hypothesis**: v067-v069 confirmed that the explicit-SiLU lean HLO cleanup is
stable but only ties the v064/v068 **4433.8 ms** band. Historical v262/v274
showed that omitting
`--xla_tpu_sparse_core_all_reduce_latency_multiplier=2` changes the schedule
and slightly regresses older RS3 frontiers, but the exact isolate has not been
retested on the current fresh-cache v067/v068 explicit-SiLU HLO family. The
current environment may prefer the default all-reduce latency estimate.

**Mechanism**: Rerun the v067/v068 explicit-SiLU image on np0 with a fresh
compile cache and omit only:

```text
--xla_tpu_sparse_core_all_reduce_latency_multiplier=2
```

Everything else stays fixed: no-scan/remat, activation sharding, MaxText CE,
Tokamax Splash max-logit 30, batch size 4 per chip, `fsdp=8,tp=1`,
all-gather on async collective fusion, SparseCore reduce-scatter/all-reduce
offload with aggregator, RS latency multiplier 3, scoped VMEM 100352,
shared-memory limit 90, scheduler rerun 3, and np0 placement.

**Predicted signal**: Support requires clean finite loss and either throughput
above the v064/v068/v069 **59.3k tok/s / 4433.8-4433.9 ms** band, or a
same-band result with lower XProf convolution/custom-call/loop-fusion time and
unchanged HBM. A changed HLO with no throughput/profile win refutes the probe,
matching the older v262/v274 AR-default pattern.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v070-ard`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v070-v067-ar-default`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v070_ar_default`
- Placement: nodepool `<your-cluster>-np-0`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Status

Completed after v069 refuted the all-gather concurrency cap as a same-HLO tie.

## Result

Workload `alekseyv-qwen3-cc5-jax-v070-ard` completed cleanly on np0 with
finite loss.

- `gke-tpu-0d0013c1-l5jf`: **59,300 tok/s**, **7,413 tok/s/chip**,
  **42.5% old MFU**.
- `gke-tpu-0d0013c1-9rkl`: **59,314 tok/s**, **7,414 tok/s/chip**,
  **42.5% old MFU**.

XProf
`2026-06-12-qwen3-jax-v070-v067-ar-default/2026_06_12_15_17_45`
reported **4435.5 ms** average step time, **1.0 ms** stddev, **66.9% MXU**,
and **0.2%** idle. Peak HBM was unchanged at **29.03 GiB / 31.25 GiB** with
**2.2202 GiB** free. Top op-profile buckets stayed in the same shape but did
not improve: convolution fusion **56,959.127 ms / 53.5%**, custom-call
**29,433.462 ms / 27.7%**, and loop fusion **10,719.414 ms / 10.1%**.

The fresh train-step HLO was copied to `/tmp/qwen3-v070-hlo` and changed only
at the schedule-text level versus v067/v068/v069: SHA256
`9405450af843e65afe4e721de51b043ab33c4c01a1b3faedd39cc60a675349af`,
**19,895,815 bytes / 119,353 lines**. Aggregate counts match v068:
`all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
`async-start=394`, `async-done=394`, `convolution=2106`,
`custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
`splash=2211`, `tokamax=2`, `jvp_jit_silu__=0`, `sigmoid=0`,
`logistic=0`, `exponential=183`, `exp.=180`.

## Verdict

**Refuted.** Omitting `--xla_tpu_sparse_core_all_reduce_latency_multiplier=2`
selects a distinct HLO hash with the same aggregate HLO surface and memory
footprint, but worsens the actual performance signal: trainer throughput drops
to **59,300-59,314 tok/s** and XProf step time worsens to **4435.5 ms** from
v064/v068/v069's **4433.8-4433.9 ms** band. Keep AR latency multiplier `2` on
the current explicit-SiLU shmem90 stack.
