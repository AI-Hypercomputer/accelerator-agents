---
title: "JAX exp 51 — Set `xla_jf_spmd_threshold_for_windowed_einsum_mib=4` (ENABLE collective matmul;"
type: experiment
tags: [llama3, jax, flax-nnx, refuted]
hypothesis: jax-llama3-match-maxtext
model: llama3-8b-jax
created: 2026-04-27
updated: 2026-04-27
commit: "v6e8-llama3-8b-jax-20260427-exp51-enable-collective-matmul (image precast-1)"
verdict: refuted
---

Set `xla_jf_spmd_threshold_for_windowed_einsum_mib=4` (ENABLE collective matmul; we had 1000000 to disable) — 6,568/chip 36.8 % MFU, **-14.7 %**. HARD REFUTE.

## Setup

- Hardware: v6e-8 (1× slice, 8 chips, FSDP=8, TP=1)
- Shape: bs=4 seq=8192
- Compute: bf16 + AMP (fp32 master weights, fp32 adamw mu/nu)
- Stack: scan-over-layers + tokamax CE (chunked_xla, autotune) + tokamax-splash with `base2/fuse_recip/mlc=30` (unless noted) + full MaxText XLA flag stack + SparseCore offload of all 3 FSDP collectives + `nothing_saveable` remat
- Image: `<your-registry>/test/llama3-8b-jax-container:precast-1`
- Submission script: `/tmp/llama3_run/xpk/exp_jax_maxtext_flags.sh` (or `exp_jax_minimal.sh` for the minimal-flags-baseline variants)

## Results

| Metric | Value |
|--------|------:|
| tok/s/chip | 6,568 |
| Reported MFU (v6e bf16 peak 918 TFLOPs/s/chip) | 36.8 % |
| Verdict | **refuted** |

Confirms MaxText's DISABLE_COLLECTIVE_MATMUL is essential for this shape; collective matmul is a regression on v6e-8 with FSDP.

## Verdict

**refuted.** This experiment is part of the post-frontier ablation series; full sweep table + analysis lives in [JAX exp 27/28b SparseCore offload frontier](2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md#wave-45-ablation-summary-exp-3960). Loss values matched the frontier step-for-step (no semantic regression).

## See also

- [JAX exp 27/28b SparseCore RS+AG offload frontier (parent writeup)](2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md)
- [JAX exp 13/15/18 chronicle](2026-04-26-jax-exp13-maxtext-xla-stack-bs5-accepted.md)
- [Loss-curve validation 100 steps](2026-04-27-jax-exp65-67-loss-validation-100steps.md)
