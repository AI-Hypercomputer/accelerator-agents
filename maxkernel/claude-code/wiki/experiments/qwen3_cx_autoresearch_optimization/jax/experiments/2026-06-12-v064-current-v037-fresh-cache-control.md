---
title: "Qwen3 JAX current v064 - v037 fresh-cache control"
type: experiment
hypothesis: "Forcing a unique persistent compile-cache directory on the exact v037 frontier rerun will emit fresh HLO and show whether the current 58.9k band is cache/HLO-family dependent."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, scheduler, shared-memory, compile-cache, hlo, control, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v064 - v037 fresh-cache control

## Hypothesis under test

**Hypothesis**: v063 used the shared lane compile cache and produced no fresh
HLO dump, while landing in the live **58.9k tok/s / 4460 ms** band. A unique
`JAX_COMPILATION_CACHE_DIR` should force compilation and HLO emission for the
exact same v258/v037 implementation. If fresh HLO differs from the earlier
frontier family, the cache/compiler path is part of the current performance
gap; if it matches and throughput remains 58.9k, this becomes the live baseline
for subsequent source work.

**Mechanism**: Reuse `qwen3-8b-jax:v258-mlp-up-first` without code changes and
run the v037 runtime stack exactly, but set:

- `JAX_COMPILATION_CACHE_DIR=gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v064_fresh`
- `XLA_FLAGS=--xla_dump_to=.../2026-06-12-qwen3-jax-v064-v037-fresh-cache/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto`

All training flags, topology, profile window, runtime scheduler flags, and
batch/sequence shape remain fixed from v037/v063.

**Predicted signal**: Fresh HLO should appear under the v064 `hlo/` prefix. A
healthy reproduction would recover v036/v037's **59,330 tok/s / 4433.0 ms**
band. A stable **58.9k / 4460 ms** result with fresh HLO means the current live
baseline is lower than the earlier carried frontier.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v064-freshcache`
- Image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- Image digest: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- Run dir: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v064-v037-fresh-cache`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Result

Completed cleanly on `2026-06-12`.

- Worker0 summary: **59,326 tok/s**, **7,416 tok/s/chip**, **42.5% old MFU**
- XProf: `2026-06-12-qwen3-jax-v064-v037-fresh-cache/2026_06_12_13_49_23`
  and `/2026_06_12_13_49_24`
- XProf step time: **4433.8 ms** average, **1.6 ms** stddev
- MXU utilization: **67.0%** / **66.8%** across the two host traces
- Peak HBM: **29.03 GiB / 31.25 GiB**, **2.2202 GiB** free
- Op profile, host `gke-tpu-0d0013c1-9rkl`: convolution fusion
  **28,542.0 ms / 53.7%**; custom-call **14,720.8 ms / 27.7%**; loop fusion
  **5,365.9 ms / 10.1%**

## HLO Dump

- Local copy: `/tmp/qwen3-v064-hlo/train_step.after_optimizations.txt`
- SHA256: `81e3c3c58b3fbfa12c0b203bdc9d0df428913fad3462ea5552e36cd07dcd5a96`
- Size: **19,915,660 bytes / 119,349 lines**
- Counts from the fresh local file: `all-gather=4628`, `all-reduce=158`,
  `reduce-scatter=1766`, `async-start=394`, `async-done=394`,
  `convolution=2106`, `custom-call=2256`, `copy=9918`, `fusion=22711`,
  `dot_general=7122`, `splash=2211`, `tokamax=2`, `jvp_jit_silu__=576`.

The fresh-cache HLO is the lean current frontier family and matches the SHA
reported by older v273/v276/v277 pages. Those older pages list larger
per-pattern counts despite the same SHA; the SHA/text are authoritative here.

## Verdict

**Supported as the live current baseline.** Isolating the compile cache forced
fresh HLO emission and recovered the **59.3k tok/s / 4433 ms** frontier band
from v063's cache-hit **58.96k / 4461 ms** run. The remaining gap to the older
v273/v277 **60.3k / 4358 ms** result is not explained by optimized HLO text,
because the SHA matches; it is likely runtime placement, scheduler execution,
or cluster/libtpu behavior. Keep fresh-cache isolation for control reruns when
testing small source perturbations.
