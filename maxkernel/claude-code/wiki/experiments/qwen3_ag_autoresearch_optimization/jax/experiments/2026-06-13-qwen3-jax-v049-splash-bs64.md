---
title: "Qwen3 jax — Splash attention (bs=64)"
type: experiment
hypothesis: "Replacing XLA SDPA with Splash attention will eliminate the materialization of the full [seq, seq] attention logits matrix in HBM, preventing the OOM encountered in v048 and allowing bs=64 to run successfully."
model: qwen3-cc-jax
variant: 8B/v6e-8
commit: ""
status: completed
tags: [qwen3, jax, splash, batch-scaling]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 jax — Splash attention (bs=64)

## Hypothesis under test

**Hypothesis**: Replacing the XLA-SDPA attention in `model/modeling_qwen3.py` with the TPU splash attention kernel will eliminate the `[B,H,S,S]` score materialization. This will reduce attention HBM traffic and peak memory, bypassing the 34.3 GB OOM that blocked batch size 64 (per-chip batch 8) at seqlen 8192 in the `v048` experiment.
**Mechanism**: Run with `--use_splash True`.
**Predicted signal**: Profile will show `custom-call` (splash) taking a larger % of the step time instead of `convolution fusion` or `loop fusion` matching standard attention. The run should succeed without OOM.
**Falsification criterion**: The run still OOMs at `bs=8` or MFU degrades relative to the baseline.

## Setup
- **Global Batch Size**: 64
- **Per-Chip Batch Size**: 8
- **Attention**: Splash Attention (`--use_splash True`, `SPLASH_BQ_DKV=1024`, `SPLASH_BKV_DKV=1024`)
- **Cluster**: charles-v6e

## Results

**Status**: FAILED (OOM)
The job failed with `RESOURCE_EXHAUSTED` during XLA:TPU compilation:
```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 33.07G of 31.25G hbm. Exceeded hbm capacity by 1.83G.
```
While Splash Attention avoided the `[seq, seq]` logits matrix materialization, the overall memory footprint of `33.07G` still exceeded the `31.25G` HBM limit by `1.83G`.

## Profile
(run failed before producing a profile)

## HLO Dump
(run failed before producing HLO)

## Verdict
refuted (OOM)
