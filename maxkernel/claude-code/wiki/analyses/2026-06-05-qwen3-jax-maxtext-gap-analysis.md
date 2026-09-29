---
title: "Qwen3 JAX vs MaxText v6e-8 gap analysis"
type: analysis
model: qwen3-cc-jax
created: 2026-06-05
updated: 2026-06-05
tags: [qwen3-cc, jax, maxtext, v6e-8, mfu, xprof, hlo, gap-analysis]
---

# Qwen3 JAX vs MaxText v6e-8 gap analysis

## Scope

The user asked to stop MaxText experiments after collecting/analyzing the
profile and to identify what is missing for the native-JAX/Flax NNX
implementation to reach the same performance. This note compares:

- Native JAX frontier:
  [v210](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v210-no-scan-sparsecore-aggregator-rerun3.md),
  with [v211](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v211-no-scan-sparsecore-concurrent-offload.md)
  as a refuted follow-up, and
  [v212](../experiments/qwen3_cx_autoresearch_optimization/jax/experiments/2026-06-05-v212-no-scan-sparsecore-aggregator-bs3-comparison.md)
  as the same-shape MaxText comparison.
- MaxText reference:
  [MaxText v006 bs3](../experiments/qwen3_cx_autoresearch_optimization/maxtext/experiments/2026-06-05-maxtext-qwen3-8b-v6e8-bs3-no-vocabtiling.md).

No more MaxText variants should be launched from this point unless the user
explicitly reopens that lane.

## Headline

Native JAX already exceeds the MaxText v006 reference in token throughput, but
MaxText reports a higher MFU:

| Stack | Shape | Aggregate tok/s | tok/s/chip | Reported MFU |
|---|---:|---:|---:|---:|
| Native JAX v210 | seq8192, global batch 32 | **59,370-59,372** | **7,421-7,422** | 42.6% |
| Native JAX v211 | seq8192, global batch 32 | 59,230-59,268 | 7,404-7,408 | 42.5% |
| Native JAX v212 | seq8192, global batch 24 | **58,970** | **7,371** | 42.3% |
| MaxText v006 | seq8192, global batch 24 | 55,064 | 6,883 | **44.92%** |

Therefore the remaining "gap" is not wall-clock token throughput at the tested
frontier. The gap is a mix of:

1. Different MFU accounting between trainers.
2. Different batch shape: MaxText v006 is bs3/chip, native JAX frontier is
   bs4/chip. v212 removes this shape difference and still shows a native-JAX
   token-throughput lead.
3. Different HLO structure and offload strategy.
4. A real native-JAX memory ceiling for the next batch step with the faster
   MaxText-style CE path.

## Evidence

### Throughput and profile

Native JAX v210:

- Best-worker aggregate throughput: **59,370-59,372 tok/s**.
- Per-chip throughput: **7,421-7,422 tok/s/chip**.
- Reported MFU: **42.6%**.
- XProf train-step profile is nearly fully busy: **0.3% idle**.
- Dominant buckets:
  - `convolution fusion`: **57,712.763 ms** / **54.2%**.
  - `async-done`: **3,645.993 ms** / **3.4%**.
  - `data formatting`: **2,909.465 ms** / **2.7%**.

Native JAX v211:

- Best-worker aggregate throughput: **59,268 tok/s**.
- Per-chip throughput: **7,408 tok/s/chip**.
- Reported MFU: **42.5%**.
- XProf MXU: **67.2%**.
- Dominant buckets:
  - `convolution fusion`: **57,674.682 ms** / **54.1%**.
  - `async-done`: **3,787.366 ms** / **3.5%**.
  - `data formatting`: **2,952.746 ms** / **2.8%**.

Native JAX v212:

- Same shape as MaxText v006: seq8192, bs3/chip, global batch 24.
- Best-worker aggregate throughput: **58,970 tok/s**.
- Per-chip throughput: **7,371 tok/s/chip**.
- Reported MFU: **42.3%**.
- XProf step time average: **3349.5 ms** on both captured hosts.
- XProf MXU: **66.5%** on c180, **65.9%** on 9vh8.
- Dominant buckets on c180:
  - `convolution fusion`: **21,921.493 ms** / **54.6%**.
  - `custom-call`: **11,044.289 ms** / **27.5%**.
  - `loop fusion`: **4,166.509 ms** / **10.4%**.

MaxText v006:

- Worker1 median step: **3.571 s**.
- Per-device throughput: **6,883 tok/s/device**.
- Aggregate throughput: **55,063.97 tok/s**.
- Reported MFU: **44.92%**.
- TFLOP/s/device: **412.32**.

The MaxText run is slower in tokens/s/chip than both native-JAX comparisons:
native JAX v210 is about **+7.8%** higher per chip (`7422 / 6883 - 1`), and
same-shape native JAX v212 is about **+7.1%** higher per chip
(`7371 / 6883 - 1`).

### HLO structure

The HLOs are very different:

| Metric | Native JAX v210 | Native JAX v211 | Native JAX v212 | MaxText v006 |
|---|---:|---:|---:|---:|
| Optimized HLO SHA | `d67433daa...` | `ba6e2d7a...` | `3a615983...` | `598fe616...` |
| Lines | 94,169 | 94,199 | 93,855 | 8,013 |
| Bytes | 16,542,528 | 16,545,183 | 16,513,927 | 1,498,186 |
| Compiled memory | 27.66 GiB | 27.68 GiB | 22.90 GiB | 48.20 GiB total incl. 23.62 GiB host/offload color |
| `all-gather` text occurrences | 8,285 | 8,289 | 8,288 | 284 |
| `all-reduce` text occurrences | 165 | 165 | 165 | 121 |
| `reduce-scatter` text occurrences | 2,772 | 2,772 | 2,772 | 0 in text count |
| `convolution` text occurrences | 3,251 | 3,251 | 3,251 | 177 |
| `custom-call` text occurrences | 4,019 | 4,025 | 4,016 | 128 |
| `fusion` text occurrences | 17,344 | 17,328 | 17,381 | 1,768 |
| `splash` text occurrences | 2,861 | 2,861 | 2,861 | 191 |

The native no-scan program is heavily unrolled and much larger than MaxText's
compiled program. That has not prevented higher token throughput, but it leaves
less margin for compile memory and makes batch-scaling failures harder to work
around.

### Memory and batch scaling

Native JAX can run the current faster MaxText-CE-style path at bs4/chip
global batch 32, but direct bs5/chip global batch 40 fails:

- v203/v204 OOM at **32.10 GiB** on a 31.25 GiB v6e-8 device.
- Largest culprit is an **11.59 GiB**
  `bf16[5,8192,151936]` full-vocab lm-head/logit temp.

Tokamax CE proves bs5 can fit but is too slow:

- v202 compiles at **25.62 GiB** but only reaches about **54.9k tok/s**.

That means the missing native-JAX lever is not another scheduler flag; it is a
fast CE/logit path that keeps MaxText-CE-like speed while avoiding the full
vocab temp at bs5.

## What is missing

1. **Comparable MFU accounting.**

   MaxText reports MFU from its own TFLOP/s/device accounting. The native-JAX
   trainer uses a different approximate FLOP formula. Since native JAX has
   higher tok/s/chip but lower reported MFU, the MFU difference is not direct
   proof of lower hardware performance.

2. **Same-shape native-JAX confirmation is now complete.**

   v212 reran the exact current no-scan SparseCore-aggregator stack at
   bs3/chip. It reached **58,970 tok/s** and **7,371 tok/s/chip**, exceeding
   MaxText v006's **55,064 tok/s** and **6,883 tok/s/chip** at the same
   seq8192/global-batch-24 shape.

3. **A fast vocab-tiled or streamed CE path.**

   The clearest native-JAX performance opportunity is to make bs5/chip fit
   without falling back to the slower Tokamax CE. The target is a vocab-tiled or
   streamed cross-entropy/lm-head path that avoids materializing
   `bf16[batch,8192,151936]` full logits while preserving the faster MaxText CE
   profile shape.

4. **MaxText-style compactness is not obviously the throughput bottleneck.**

   MaxText's HLO is much smaller, but the native no-scan HLO is faster in
   tokens/sec. Compacting the HLO may help compile/cache ergonomics and memory,
   but the profile says the steady-state bottleneck is still mostly matmul
   convolution fusion, not host idleness.

## Recommended next native-JAX experiments

1. **v213/v214 CE/logit memory prototypes are invalid as implemented**: the
   naive Python/XLA vocab-tiled CE passed toy CPU equivalence, but failed TPU
   train-step compile at **36.99G / 31.25G HBM**. It removed full-vocab logits
   but autodiff saved per-chunk logits and masks (`bf16/f32/pred[32768,4096]`).
   v214 rematerialized chunk functions and improved the compile OOM to
   **35.78G / 31.25G HBM**, still **4.54G** over capacity. v215 tried chunk
   size 2048 and worsened to **54.36G / 31.25G HBM**. v216 tried an explicit
   custom VJP and removed chunk predicates, but worsened to **38.12G / 31.25G
   HBM** because the unrolled backward retained many f32 chunk-dot temps. Stop
   remat-only and unrolled custom-VJP tuning. A local scan-shaped custom-VJP
   candidate then failed CPU CE equivalence before launch, so the remaining CE
   path is a more faithful streaming/Pallas kernel or a different structural
   approach with an explicit equivalence gate.

2. **Only after CE succeeds, retry bs5**: use the v210 runtime stack plus the
   new CE path at `--batch_size=5`. Support requires fitting and beating the
   59.37k tok/s frontier.

## Verdict

MaxText v006 is a valid reference and its profile/HLO were collected, but
native JAX is already ahead on token throughput, including at the exact
same-shape bs3/chip comparison. The actionable missing piece for further
native-JAX progress is not the MaxText recipe itself; it is a fast logit/CE
memory reduction that allows the next batch-size step without the Tokamax CE
slowdown.
