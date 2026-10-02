---
title: "Qwen3 JAX current v080 - sparse-label MaxText CE"
type: experiment
hypothesis: "Replacing the one-hot-target MaxText CE custom VJP with a sparse-label equivalent may reduce CE live range or HLO clutter without changing loss math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v080-sparse-maxtext-ce
status: invalid
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, sparse-labels, cross-entropy, splash, tokamax-splash, sparsecore, collectives, scheduler, compile-cache, long-seq, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v080 - sparse-label MaxText CE

## Hypothesis under test

**Hypothesis**: The current MaxText/T5X CE path creates an explicit one-hot
`[B,T,V]` target tensor before calling the custom VJP. A sparse-label custom VJP
can compute the same per-token loss and logits gradient using gather/scatter
instead of a full target tensor. This may reduce CE live range, HLO size, or
profile time without changing model math.

**Mechanism**: Build a thin image from `qwen3-8b-jax:v067-explicit-silu` and
replace only `train.py`'s `use_maxtext_ce` implementation:

- Keep the original one-hot `cross_entropy_with_logits` function available.
- Add `sparse_cross_entropy_with_logits(logits, labels)` with a custom VJP.
- In `_ce_maxtext`, call the sparse-label function directly instead of
  constructing `jax.nn.one_hot(labels, vocab)`.

All model code, sharding, remat, Splash, batch shape, and runtime flags are
unchanged from the carried shmem90 explicit-SiLU stack.

**Predicted signal**: Support requires clean finite loss plus either lower HLO
memory/size or lower XProf CE/convolution-fusion time without reducing host
throughput below the current durable **~59.33k tok/s / 4433.8 ms** frontier.
Compile failure, scatter-heavy HLO, higher memory, or slower throughput refutes
the sparse-label CE path.

## Equivalence

Local CPU equivalence inside the v080 image context passed against the original
one-hot MaxText CE:

- Dense loss: **3.2754132747650146**
- Sparse loss: **3.2754132747650146**
- Max loss diff: **0.0**
- Max logits-gradient diff: **7.45e-09**
- `allclose(atol=1e-6, rtol=1e-6)`: **true**

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v080-sparsece`
- Base image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Experiment image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v080-sparse-maxtext-ce`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v080-v067-sparse-maxtext-ce`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v080_sparse_maxtext_ce`
- Placement: nodepool `<your-cluster>-np-0`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Status

Failed during first train-step compile after v079 closed the shared-memory
bracket.

## Results

The workload `alekseyv-qwen3-cc5-jax-v080-sparsece` failed before step 0 with
`EXIT_CODE=1`. XLA reported a permanent TPU compile OOM:

- Used **39.19 GiB** of **31.25 GiB** HBM, exceeding capacity by **7.95 GiB**.
- Total HBM usage including reservation was at least **39.45 GiB**.
- Program HBM requirement: **39.19 GiB**
- HLO temp: **39.11 GiB**

The OOM top allocations show the sparse-label CE path made the full-vocab logits
live as both fp32 and bf16 temporaries:

- **18.55 GiB** `f32[4,8192,151936]`
- **9.27 GiB** `bf16[4,8192,151936]`

Fresh failed train-step HLO was copied to `/tmp/qwen3-v080-hlo`:

- SHA256:
  `b53b3502a8c775f87c39060a6da2a0b53392bff8dca0ff5108d1a0901d3e2d5a`
- Size: **18,953,768 bytes / 114,408 lines**
- Compile memory report total bytes: **44.86 GiB**
- Counts: `all-gather=4134`, `all-reduce=158`, `reduce-scatter=1766`,
  `async-start=414`, `async-done=414`, `convolution=2070`,
  `custom-call=2216`, `copy=10427`, `fusion=19939`, `dot_general=6673`,
  `scatter=2044`, `gather=4168`, `dynamic-update-slice=216`,
  `splash=2210`, `tokamax=2`, `jvp_jit_silu__=0`, `exponential=182`.

## Verdict

Invalid/refuted. The sparse-label CE VJP was locally equivalent, but on the full
Qwen3-8B shape it selected a scatter/gather-heavy HLO and worsened the logits
live range enough to OOM. Do not carry this implementation; a viable CE path
must avoid the full `[B,T,V]` logits allocation instead of reshaping its target
gradient.
