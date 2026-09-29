---
title: "Qwen3 jax — Tokamax CE only (bs=64)"
type: experiment
hypothesis: "Tokamax chunked-XLA cross-entropy without Splash attention will avoid the compilation hang seen in v050 while still saving ~4GB of HBM by avoiding the `[seq, vocab]` logits materialization. This memory saving should be enough to bypass the 34.3GB OOM seen in v048 and allow the 31.25GB v6e-8 to run at bs=64 (per-chip bs=8)."
model: qwen3-cc-jax
variant: 8B/v6e-8
commit: ""
status: completed
tags: [qwen3, jax, tokamax, batch-scaling]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 jax — Tokamax CE only (bs=64)

## Hypothesis under test

**Hypothesis**: The combination of Splash attention and Tokamax CE caused an XLA compilation hang in `v050`. However, the logits matrix for standard cross-entropy (`[batch=8, seq=8192, vocab=151936]` in FP32) consumes ~3.98 GB of HBM. By enabling *only* Tokamax CE (`--use_tokamax_ce True` and `--use_splash False`), we can save ~4GB of memory. Since `v048` (baseline at bs=64) OOM'd at 34.3GB (which is ~3GB over the 31.25GB limit), this saving should be sufficient to fit the model in memory without the compilation complexity of Splash.
**Mechanism**: Run `train.py` with `--use_splash False` and `--use_tokamax_ce True`.
**Predicted signal**: The model will compile successfully and train at `bs=64` (per-chip batch 8) without hitting the `RESOURCE_EXHAUSTED` OOM.
**Falsification criterion**: The run still OOMs, hangs during compilation, or MFU degrades.

## Setup
- **Global Batch Size**: 64
- **Per-Chip Batch Size**: 8
- **Attention**: XLA SDPA (`--use_splash False`)
- **Loss**: Tokamax CE (`--use_tokamax_ce True`)
- **Cluster**: charles-v6e

## Results

**Status**: FAILED (Hard Crash)
The job compiled successfully and started the training loop, but the pod was deleted immediately upon executing the first step. This hard crash (exit code `null`, pod deleted) without a Python traceback is indicative of a fatal TPU runtime error, likely triggered by an internal Pallas assertion failure or a VMEM OOM during the `tokamax` kernel autotuning phase at this tensor size.

## Profile
(run crashed before producing a profile)

## HLO Dump
(partial dump generated)

## Verdict
refuted (hard crash during step execution)
