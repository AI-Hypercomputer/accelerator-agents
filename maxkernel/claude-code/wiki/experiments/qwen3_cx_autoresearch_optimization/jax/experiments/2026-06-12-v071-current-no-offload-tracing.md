---
title: "Qwen3 JAX current v071 - explicit-SiLU stack without SparseCore offload tracing"
type: experiment
hypothesis: "Disabling active SparseCore reduce-scatter/all-reduce offload tracing may keep the current explicit-SiLU HLO while reducing runtime/profile overhead."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, tracing, scheduler, compile-cache, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v071 - explicit-SiLU stack without SparseCore offload tracing

## Hypothesis under test

**Hypothesis**: v067-v070 show the current explicit-SiLU shmem90 stack is
stable around **59.3k tok/s** and **4433.8-4435.5 ms**. Historical v238/v247/v261
showed that disabling active SparseCore offload tracing was a tie on older
frontiers, but the exact current v067 HLO family has not been closed. Removing
the tracing records may reduce runtime/profile overhead without changing model
math or the optimized HLO surface.

**Mechanism**: Rerun the v067/v068 explicit-SiLU image on np0 with a fresh
compile cache and change only the active SparseCore tracing flags:

```text
--xla_tpu_enable_all_reduce_offload_tracing=false
--xla_tpu_enable_reduce_scatter_offload_tracing=false
```

Everything else stays fixed: no-scan/remat, activation sharding, MaxText CE,
Tokamax Splash max-logit 30, batch size 4 per chip, `fsdp=8,tp=1`,
all-gather on async collective fusion, SparseCore reduce-scatter/all-reduce
offload with aggregator, RS3+AR2, scoped VMEM 100352, shared-memory limit 90,
scheduler rerun 3, and np0 placement.

**Predicted signal**: Support requires clean finite loss and either throughput
above the v064/v068/v069 **59.3k tok/s / 4433.8-4433.9 ms** band, or same-band
throughput with a lower XProf step time/profile overhead and unchanged HBM. A
byte-identical or schedule-equivalent HLO with no throughput/profile gain
refutes the probe.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v071-notrace`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v071-v067-no-offload-tracing`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v071_no_offload_tracing`
- Placement: nodepool `<your-cluster>-np-0`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Status

Completed cleanly after v070 refuted AR default as a changed-schedule
regression.

## Results

Worker summaries:

- `gke-tpu-0d0013c1-9rkl`: **59,329 tok/s**, **7,416 tok/s/chip**,
  **42.5%** old MFU.
- `gke-tpu-0d0013c1-l5jf`: **59,309 tok/s**, **7,414 tok/s/chip**,
  **42.5%** old MFU.

XProf
`2026-06-12-qwen3-jax-v071-v067-no-offload-tracing/2026_06_12_15_32_26`
reported **4432.8 ms** average step time, **1.7 ms** stddev, **66.8% MXU**,
and **0.3%** idle. Peak HBM was unchanged at **29.03 GiB / 31.25 GiB** with
**2.2202 GiB** free.

Top XProf buckets were same-band with v068-v070:

- `convolution fusion`: **56,920.506 ms / 53.5%**
- `custom-call`: **29,433.4 ms / 27.7%** (same family as v068-v070)
- `data formatting`: **2,900.706 ms / 2.7%**

Fresh train-step HLO was copied to `/tmp/qwen3-v071-hlo` and is byte-identical
to v068:

- SHA256:
  `84df522a5e6dfdd158288d856af82a5b24506c30db8c0fb29dfb46c952378a38`
- Size: **19,895,815 bytes / 119,353 lines**
- Counts: `all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
  `async-start=394`, `async-done=394`, `convolution=2106`,
  `custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
  `splash=2211`, `tokamax=2`, `jvp_jit_silu__=0`, `sigmoid=0`,
  `logistic=0`, `exponential=183`, `exp.=180`.

## Verdict

Refuted/tie. Disabling active SparseCore reduce-scatter/all-reduce offload
tracing does not change the compiled train-step HLO, memory footprint, or
profile breakdown, and throughput remains within the v064/v068/v069 band rather
than beating it. Do not carry these tracing-disable flags in the current stack.
