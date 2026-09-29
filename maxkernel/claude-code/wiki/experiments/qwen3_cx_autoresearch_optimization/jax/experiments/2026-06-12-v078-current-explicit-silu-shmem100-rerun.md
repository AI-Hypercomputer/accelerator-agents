---
title: "Qwen3 JAX current v078 - explicit-SiLU shmem100 rerun"
type: experiment
hypothesis: "An exact fresh-cache rerun should reproduce v077's small shmem100 gain if it is real rather than noise."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, scheduler, shared-memory, compile-cache, rerun, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v078 - explicit-SiLU shmem100 rerun

## Hypothesis under test

**Hypothesis**: v077's explicit-SiLU + shared-memory limit 100 run produced a
small but coherent improvement (**59,340 tok/s**, **4432.6 ms**) over the
shmem90 band while keeping the same aggregate HLO counts. Because the gain is
small and HBM margin is tight (**0.3181 GiB** free), an exact fresh-cache rerun
is required before carrying shmem100.

**Mechanism**: Rerun v077 exactly with a fresh compile cache and run directory.
No model, trainer, runtime flag, placement, batch, or profile-window change.

**Predicted signal**: Support requires clean finite loss, no HBM OOM, and
host/XProf metrics in the v077 band: about **59.34k tok/s**, **4432-4433 ms**,
and the same HLO hash or an equivalent same-count schedule. Falling back to the
shmem90 band or worse refutes v077 as noise.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v078-shm100r`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v078-v077-shmem100-rerun`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v078_explicit_silu_shmem100_rerun`
- Placement: nodepool `<your-cluster>-np-0`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Status

Completed cleanly after v077 provisional support.

## Results

Worker summaries:

- `gke-tpu-0d0013c1-l5jf`: **59,306 tok/s**, **7,413 tok/s/chip**,
  **42.5%** old MFU, about **48.36% MaxText-style MFU**.
- `gke-tpu-0d0013c1-9rkl`: **59,329 tok/s**, **7,416 tok/s/chip**,
  **42.5%** old MFU, about **48.37% MaxText-style MFU**.

XProf
`2026-06-12-qwen3-jax-v078-v077-shmem100-rerun/2026_06_12_16_55_37`
reported **4434.7 ms** average step time, **1.4 ms** stddev, **66.9% MXU**,
and **0.3%** idle. Peak HBM matched v077 at **30.93 GiB / 31.25 GiB**, leaving
only **0.3181 GiB** free.

Top XProf buckets:

- `convolution fusion`: **56,857.906 ms / 53.4%**
- `custom-call`: **29,423.697 ms / 27.6%**
- `loop fusion`: **10,719.645 ms / 10.1%**

Fresh train-step HLO was copied to `/tmp/qwen3-v078-hlo` and was byte-identical
to v077:

- SHA256:
  `4e271536a745449917e002dcb8f554b509bb0d7cff82da955d0ad7005ca97ebc`
- Size: **19,895,806 bytes / 119,353 lines**
- Counts match v077: `all-gather=4628`, `all-reduce=158`,
  `reduce-scatter=1766`, `async-start=394`, `async-done=394`,
  `convolution=2106`, `custom-call=2256`, `copy=9918`, `fusion=22711`,
  `dot_general=7122`, `splash=2211`, `tokamax=2`, `jvp_jit_silu__=0`.

## Verdict

Refuted as confirmation. v078 reproduced the shmem100 HLO and tight HBM margin,
but not v077's small performance gain: host throughput returned to the current
shmem90 band and XProf regressed to **4434.7 ms**. Do not carry shmem100 as the
default; treat v077 as a noisy/tie sample.
