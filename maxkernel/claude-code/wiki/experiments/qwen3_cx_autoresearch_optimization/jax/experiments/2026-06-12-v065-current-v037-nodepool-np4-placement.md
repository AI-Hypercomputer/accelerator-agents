---
title: "Qwen3 JAX current v065 - v037 nodepool np4 placement"
type: experiment
hypothesis: "The same optimized HLO shows different live step times across current runs; pinning the v064 fresh-cache frontier to nodepool np4 tests whether placement recovers part of the 60.3k historical gap."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, scheduler, shared-memory, compile-cache, placement, nodepool, control, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v065 - v037 nodepool np4 placement

## Hypothesis under test

**Hypothesis**: v064 recovered the lean frontier HLO but still ran at
**4433.8 ms**, while old v273/v277 reported the same train-step HLO SHA with
**4358 ms** profiles. Pinning the exact v064 runtime to a different available
2x4 v6e nodepool can reveal whether nodepool placement explains part of this
same-HLO runtime gap.

**Mechanism**: Reuse `qwen3-8b-jax:v258-mlp-up-first` with the v064/v037 flags,
fresh compile-cache isolation, and an added node selector:

- `cloud.google.com/gke-nodepool=<your-cluster>-np-4`
- `JAX_COMPILATION_CACHE_DIR=gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v065_np4`

No model code, math, shape, runtime scheduler flags, profile window, or HLO
dump settings change otherwise.

**Predicted signal**: Support requires the same lean HLO family with step time
below v064's **4433.8 ms** and throughput above **59,326 tok/s**. A same-band
or slower result closes this placement probe and keeps v064 as the live
baseline.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v065-np4`
- Image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- Run dir: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v065-v037-np4-placement`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Result

Completed cleanly on `2026-06-12` on np4 nodes
`gke-tpu-32cb1c36-2tqg` and `gke-tpu-32cb1c36-fjrb`.

- Worker0 summary: **58,568 tok/s**, **7,321 tok/s/chip**, **42.0% old MFU**
- XProf: `2026-06-12-qwen3-jax-v065-v037-np4-placement/2026_06_12_14_02_43`
  and `/2026_06_12_14_02_44`
- XProf step time: **4487.0 ms** average, **1.7 ms** stddev
- MXU utilization: **65.9%** / **67.5%** across the two host traces
- Peak HBM: **29.03 GiB / 31.25 GiB**, **2.2202 GiB** free
- Op profile, host `gke-tpu-32cb1c36-fjrb`: convolution fusion
  **28,180.5 ms / 52.3%**; custom-call **14,703.2 ms / 27.3%**; loop fusion
  **5,338.7 ms / 9.9%**; idle **0.4%**

## HLO Dump

- Local copy: `/tmp/qwen3-v065-hlo/train_step.after_optimizations.txt`
- SHA256: `81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`
- Size: **19,915,660 bytes / 119,349 lines**
- Counts: `all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
  `async-start=394`, `async-done=394`, `convolution=2106`,
  `custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
  `splash=2211`, `tokamax=2`, `jvp_jit_silu__=576`.

## Verdict

**Refuted.** The np4 placement keeps the exact same optimized train-step HLO
and memory envelope as v064 but regresses from **59,326 tok/s / 4433.8 ms** to
**58,568 tok/s / 4487.0 ms**. Do not use np4 for the current frontier; prefer
the np0 placement seen in v036/v037/v064 when capacity permits.
