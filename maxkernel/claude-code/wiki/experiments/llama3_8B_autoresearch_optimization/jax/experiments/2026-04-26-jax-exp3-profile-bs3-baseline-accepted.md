---
title: "JAX exp 3 — First xprof capture of the JAX baseline at bs=3 seq=8192. Profile saved to GCS; "
type: experiment
tags: [llama3, jax, flax-nnx, supported]
hypothesis: jax-llama3-match-maxtext
model: llama3-8b-jax
created: 2026-04-26
updated: 2026-04-26
commit: "v6e8-llama3-8b-jax-20260426-exp3-profile-bs3-baseline (image precast-1)"
verdict: supported
---

First xprof capture of the JAX baseline at bs=3 seq=8192. Profile saved to GCS; identifies async-AR + matmul as primary cost categories.

## Setup

- Hardware: v6e-8 (1× slice, 8 chips, FSDP=8, TP=1)
- Shape: bs=3 seq=8192
- Compute: bf16 + AMP (fp32 master weights, fp32 adamw mu/nu)
- Stack: scan-over-layers + tokamax CE (chunked_xla, autotune) + tokamax-splash with `base2/fuse_recip/mlc=30` (unless noted) + full MaxText XLA flag stack + SparseCore offload of all 3 FSDP collectives + `nothing_saveable` remat
- Image: `<your-registry>/test/llama3-8b-jax-container:precast-1`
- Submission script: `/tmp/llama3_run/xpk/exp_jax_maxtext_flags.sh` (or `exp_jax_minimal.sh` for the minimal-flags-baseline variants)

## Results

| Metric | Value |
|--------|------:|
| tok/s/chip | 6,529 |
| Reported MFU (v6e bf16 peak 918 TFLOPs/s/chip) | 36.6 % |
| Verdict | **supported** |

Capture-only run; same throughput as exp 1e. Used to author the bottleneck-breakdown table that motivated exp 12's MaxText XLA flag stack experiment.

## Verdict

**supported.** This experiment is part of the post-frontier ablation series; full sweep table + analysis lives in [JAX exp 27/28b SparseCore offload frontier](2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md#wave-45-ablation-summary-exp-3960). Loss values matched the frontier step-for-step (no semantic regression).

## See also

- [JAX exp 27/28b SparseCore RS+AG offload frontier (parent writeup)](2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md)
- [JAX exp 13/15/18 chronicle](2026-04-26-jax-exp13-maxtext-xla-stack-bs5-accepted.md)
- [Loss-curve validation 100 steps](2026-04-27-jax-exp65-67-loss-validation-100steps.md)
