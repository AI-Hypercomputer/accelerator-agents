---
title: "Qwen3 JAX current v077 - explicit-SiLU stack with shmem100"
type: experiment
hypothesis: "The smaller explicit-SiLU HLO may recover the faster historical shared-memory-limit 100 schedule without exceeding v6e HBM."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: provisional-supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, scheduler, shared-memory, compile-cache, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v077 - explicit-SiLU stack with shmem100

## Hypothesis under test

**Hypothesis**: The historical v273/v276 frontier used scheduler shared-memory
limit 100 and profiled around **4358 ms**, while current v036/v037 needed
shmem90 for the best current-cluster band. v067 explicit-SiLU makes the current
HLO smaller and leaves about **2.22 GiB** free HBM at shmem90. Raising only the
shared-memory limit back to 100 on this lean HLO may recover a faster schedule
without hitting the tighter old-v273 memory envelope.

**Mechanism**: Rerun the v067/v068 explicit-SiLU image on np0 with a fresh
compile cache and change only:

```text
--xla_tpu_scheduler_percent_shared_memory_limit=100
```

Everything else stays fixed: no-scan/remat, activation sharding, MaxText CE,
Tokamax Splash max-logit 30, batch size 4 per chip, `fsdp=8,tp=1`,
all-gather/reduce-scatter ACF enabled, all-reduce ACF disabled, SparseCore
reduce-scatter/all-reduce offload with aggregator, RS3+AR2, scoped VMEM 100352,
scheduler rerun 3, normal offload tracing, and np0 placement.

**Predicted signal**: Support requires clean finite loss and either throughput
above the current durable **59.33k tok/s / 4433.8 ms** band or a lower XProf
step time/profile breakdown without HBM/compile failure. OOM, higher HBM with
slower profile, or same/lower throughput refutes the probe.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v077-shm100`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v077-v067-shmem100`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v077_explicit_silu_shmem100`
- Placement: nodepool `<your-cluster>-np-0`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Status

Completed cleanly after v076 refuted disabling ACF multiple-steps as a smaller
but slower schedule.

## Results

Worker summaries:

- `gke-tpu-0d0013c1-l5jf`: **59,320 tok/s**, **7,415 tok/s/chip**,
  **42.5%** old MFU, about **48.37% MaxText-style MFU**.
- `gke-tpu-0d0013c1-9rkl`: **59,340 tok/s**, **7,418 tok/s/chip**,
  **42.5%** old MFU, about **48.39% MaxText-style MFU**.

XProf
`2026-06-12-qwen3-jax-v077-v067-shmem100/2026_06_12_16_44_04`
reported **4432.6 ms** average step time, **0.1 ms** stddev, **66.8% MXU**,
and **0.2%** idle. Peak HBM increased to **30.93 GiB / 31.25 GiB**, leaving
only **0.3181 GiB** free.

Top XProf buckets:

- `convolution fusion`: **56,850.644 ms / 53.4%**
- `custom-call`: **29,423.942 ms / 27.7%**
- `loop fusion`: **10,725.841 ms / 10.1%**

Fresh train-step HLO was copied to `/tmp/qwen3-v077-hlo` and changed slightly
from the shmem90 v068/v071/v073/v075 family:

- SHA256:
  `4e271536a745449917e002dcb8f554b509bb0d7cff82da955d0ad7005ca97ebc`
- Size: **19,895,806 bytes / 119,353 lines**
- Counts: `all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
  `async-start=394`, `async-done=394`, `convolution=2106`,
  `custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
  `splash=2211`, `tokamax=2`, `jvp_jit_silu__=0`, `sigmoid=0`,
  `logistic=0`, `exponential=183`.

## Verdict

Provisionally supported. Restoring shared-memory limit 100 on the lean
explicit-SiLU HLO recovers a slightly faster profile and the best current
host-summary sample (**59,340 tok/s**, **4432.6 ms**) without OOM. The gain is
small and HBM margin is tight at **0.3181 GiB**, so run an exact rerun before
carrying shmem100 as the new runtime default.
