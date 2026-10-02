---
title: "Qwen3 JAX current v072 - explicit-SiLU stack with scheduler resource sharing"
type: experiment
hypothesis: "Latency-hiding scheduler resource sharing may improve overlap on the current explicit-SiLU RS3+AR2 HLO even though it regressed older frontier families."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, latency-hiding, resource-sharing, scheduler, compile-cache, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v072 - explicit-SiLU stack with scheduler resource sharing

## Hypothesis under test

**Hypothesis**: The v064-v071 current stack is stable around **59.3k tok/s** and
**4433 ms**, with most remaining time in convolution fusion, Splash custom-calls,
and loop/data-formatting fusions. Earlier resource-sharing probes regressed
older HLO families, but the current explicit-SiLU RS3+AR2 HLO has a smaller
collective/fusion surface. Enabling scheduler resource sharing may alter
resource assignment enough to reduce the residual schedule tail without changing
model math.

**Mechanism**: Rerun the v067/v068 explicit-SiLU image on np0 with a fresh
compile cache and add only:

```text
--xla_latency_hiding_scheduler_resource_sharing=true
```

Everything else stays fixed: no-scan/remat, activation sharding, MaxText CE,
Tokamax Splash max-logit 30, batch size 4 per chip, `fsdp=8,tp=1`,
all-gather on async collective fusion, SparseCore reduce-scatter/all-reduce
offload with aggregator, RS3+AR2, scoped VMEM 100352, shared-memory limit 90,
scheduler rerun 3, normal offload tracing, and np0 placement.

**Predicted signal**: Support requires clean finite loss and either throughput
above the current durable **59.33k tok/s / 4433.8 ms** band or a lower XProf
step time/profile breakdown with unchanged HBM. A changed HLO or same-HLO
runtime at or below the band refutes the probe.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v072-rshare`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v072-v067-resource-sharing`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v072_resource_sharing`
- Placement: nodepool `<your-cluster>-np-0`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Status

Completed cleanly after v071 refuted no-offload-tracing as an identical-HLO tie.

## Results

Worker summaries:

- `gke-tpu-0d0013c1-9rkl`: **59,306 tok/s**, **7,413 tok/s/chip**,
  **42.5%** old MFU.
- `gke-tpu-0d0013c1-l5jf`: **59,229 tok/s**, **7,404 tok/s/chip**,
  **42.5%** old MFU.

XProf
`2026-06-12-qwen3-jax-v072-v067-resource-sharing/2026_06_12_15_45_27`
reported **4437.3 ms** average step time, **0.2 ms** stddev, **66.4% MXU**,
and **0.6%** idle. Peak HBM stayed unchanged at **29.03 GiB / 31.25 GiB** with
**2.2202 GiB** free.

Top XProf buckets:

- `convolution fusion`: **56,930.473 ms / 53.3%**
- `custom-call`: **29,433.840 ms / 27.6%**
- `loop fusion`: **10,727.079 ms / 10.0%**

Fresh train-step HLO was copied to `/tmp/qwen3-v072-hlo` and is byte-identical
to v068/v071:

- SHA256:
  `84df522a5e6dfdd158288d856af82a5b24506c30db8c0fb29dfb46c952378a38`
- Size: **19,895,815 bytes / 119,353 lines**
- Counts: `all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
  `async-start=394`, `async-done=394`, `convolution=2106`,
  `custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
  `splash=2211`, `tokamax=2`, `jvp_jit_silu__=0`, `sigmoid=0`,
  `logistic=0`, `exponential=183`, `exp.=180`.

## Verdict

Refuted. Scheduler resource sharing does not change the current explicit-SiLU
HLO or memory footprint and regresses the profile from the v068/v071
**4432.8-4433.8 ms** band to **4437.3 ms** with lower host throughput. Keep
resource sharing disabled/default.
