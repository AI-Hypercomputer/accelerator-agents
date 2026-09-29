---
title: "Qwen3 JAX current v079 - explicit-SiLU shmem95"
type: experiment
hypothesis: "A shared-memory limit midpoint at 95 may recover scheduler latitude from shmem100 without the full HBM squeeze, or prove v077's gain was noise."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, scheduler, shared-memory, compile-cache, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v079 - explicit-SiLU shmem95

## Hypothesis under test

**Hypothesis**: v077's shmem100 HLO produced one lower-profile sample
(**4432.6 ms**) but v078 did not reproduce the throughput gain and kept only
**0.3181 GiB** HBM free. A midpoint shared-memory limit of **95** may give the
scheduler more room than the carried shmem90 frontier while avoiding the full
shmem100 memory squeeze.

**Mechanism**: Use the same v067 explicit-SiLU image and current carried runtime
stack, changing only
`--xla_tpu_scheduler_percent_shared_memory_limit=95`.

**Predicted signal**: Support requires clean finite loss and either throughput
above the durable shmem90 band (**~59.33k tok/s**) or XProf step time below
**4433.8 ms** without shmem100's very tight HBM margin. Slower profile,
unchanged same-family throughput, or tight/failed HBM refutes the midpoint and
closes this shared-memory bracket.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v079-shm95`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v079-v067-shmem95`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v079_explicit_silu_shmem95`
- Placement: nodepool `<your-cluster>-np-0`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Status

Completed cleanly after v078 refuted shmem100 confirmation.

## Results

Worker summaries:

- `gke-tpu-0d0013c1-l5jf`: **59,308 tok/s**, **7,414 tok/s/chip**,
  **42.5%** old MFU, about **48.35% MaxText-style MFU**.
- `gke-tpu-0d0013c1-9rkl`: **59,326 tok/s**, **7,416 tok/s/chip**,
  **42.5%** old MFU, about **48.37% MaxText-style MFU**.

XProf
`2026-06-12-qwen3-jax-v079-v067-shmem95/2026_06_12_17_08_35`
reported **4434.2 ms** average step time, **1.3 ms** stddev, **67.0% MXU**,
and **0.2%** idle. Peak HBM matched shmem100 at
**30.93 GiB / 31.25 GiB**, leaving only **0.3181 GiB** free.

Top XProf buckets:

- `convolution fusion`: **28,504.917 ms / 53.6%** on the single-host view.
- `custom-call`: included under `custom fusion`, **1,902.119 ms / 3.6%** on
  the single-host view; the aggregate program remained the same shmem100-family
  schedule.

Fresh train-step HLO was copied to `/tmp/qwen3-v079-hlo` and was byte-identical
to v077/v078:

- SHA256:
  `4e271536a745449917e002dcb8f554b509bb0d7cff82da955d0ad7005ca97ebc`
- Size: **19,895,806 bytes / 119,353 lines**
- Counts: `all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
  `async-start=394`, `async-done=394`, `convolution=2106`,
  `custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
  `splash=2211`, `tokamax=2`, `jvp_jit_silu__=0`, `exponential=183`.

## Verdict

Refuted. shmem95 selected the same HLO and tight HBM profile as shmem100, while
host throughput and XProf step time stayed below the durable shmem90
explicit-SiLU frontier. Do not carry shmem95; close this shared-memory bracket
unless a future source-code change creates a new schedule.
