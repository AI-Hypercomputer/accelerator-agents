---
title: "Qwen3 JAX current v066 - v037 nodepool np1 placement"
type: experiment
hypothesis: "A fresh-cache exact v064 rerun pinned to nodepool np1 tests whether v063's slow np1 result was caused by cache state or by placement."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, scheduler, shared-memory, compile-cache, placement, nodepool, control, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v066 - v037 nodepool np1 placement

## Hypothesis under test

**Hypothesis**: v063 ran on np1 and was slow, but it hit the shared compile
cache and emitted no fresh train-step HLO. Pinning an exact v064 fresh-cache
control to np1 separates cache state from nodepool placement.

**Mechanism**: Reuse the v064/v037 stack exactly with a node selector for
`cloud.google.com/gke-nodepool=<your-cluster>-np-1` and a unique
compile cache:

- `JAX_COMPILATION_CACHE_DIR=gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v066_np1`

No model code, math, shape, scheduler flags, profile window, or HLO dump
settings change otherwise.

**Predicted signal**: If np1 with fresh HLO recovers v064's **59,326 tok/s /
4433.8 ms**, v063 was mostly cache-path noise. If np1 stays near or below
v063/v065, prefer np0 for the live frontier and close placement probing.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v066-np1`
- Image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- Run dir: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v066-v037-np1-placement`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Result

Completed cleanly on `2026-06-12` on np1 nodes
`gke-tpu-01f93eb9-43vk` and `gke-tpu-01f93eb9-ghj6`.

- Worker0 summary: **58,948 tok/s**, **7,369 tok/s/chip**, **42.3% old MFU**
- XProf: `2026-06-12-qwen3-jax-v066-v037-np1-placement/2026_06_12_14_14_28`
- XProf step time: **4463.2 ms** average, **1.3 ms** stddev
- MXU utilization: **66.7%**
- Peak HBM: **29.03 GiB / 31.25 GiB**, **2.2202 GiB** free
- Op profile: convolution fusion **57,034.0 ms / 53.3%**; custom-call
  **29,412.0 ms / 27.5%**; loop fusion **10,713.0 ms / 10.0%**

## HLO Dump

- Local copy: `/tmp/qwen3-v066-hlo/train_step.after_optimizations.txt`
- SHA256: `81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`
- Size: **19,915,660 bytes / 119,349 lines**
- Counts: `all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
  `async-start=394`, `async-done=394`, `convolution=2106`,
  `custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
  `splash=2211`, `tokamax=2`, `jvp_jit_silu__=576`.

## Verdict

**Refuted.** Fresh-cache np1 improves over the v063 cache-hit run but remains
below v064/np0 (**58,948 tok/s / 4463.2 ms** versus **59,326 tok/s /
4433.8 ms**) with identical HLO and HBM. Prefer np0 for the live frontier and
close placement probing unless a new cluster/runtime signal appears.
