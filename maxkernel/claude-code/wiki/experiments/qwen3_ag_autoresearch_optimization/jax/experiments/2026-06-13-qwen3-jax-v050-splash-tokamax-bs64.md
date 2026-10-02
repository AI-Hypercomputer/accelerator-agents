---
title: "Qwen3 jax — Splash attention + Tokamax CE (bs=64)"
type: experiment
hypothesis: "Replacing XLA SDPA with Splash attention combined with replacing the standard softmax cross-entropy with the Tokamax streamed chunked-XLA cross-entropy will eliminate the materialization of both the [seq, seq] attention logits and the [seq, vocab] lm_head logits. This combined memory savings will fit the 8B model into the 31.25GB HBM limit at bs=64 (per-chip bs=8)."
model: qwen3-cc-jax
variant: 8B/v6e-8
commit: ""
status: completed
tags: [qwen3, jax, splash, tokamax, batch-scaling]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 jax — Splash attention + Tokamax CE (bs=64)

## Hypothesis under test

**Hypothesis**: By combining the Splash attention kernel (`--use_splash=True`) and the Tokamax streamed chunked cross-entropy (`--use_tokamax_ce=True`), we can eliminate both the `[seq, seq]` score materialization in the attention layers and the `[seq, vocab]` logits materialization in the language model head. This should free enough HBM to bypass the memory capacity limit that blocked `v049` (which only used Splash attention and exceeded HBM by 1.83 GB).
**Mechanism**: Run `train.py` with `--use_splash True` and `--use_tokamax_ce True`.
**Predicted signal**: The model will compile successfully and train at `bs=64` (per-chip batch 8) without hitting the `RESOURCE_EXHAUSTED` OOM. MFU should increase significantly relative to the `bs=32` baseline due to the larger batch size and reduced memory bandwidth overhead.
**Falsification criterion**: The run still OOMs, or the overhead of the two kernels reduces the MFU below the `v047` baseline.

## Setup
- **Global Batch Size**: 64
- **Per-Chip Batch Size**: 8
- **Attention**: Splash Attention (`--use_splash True`, `SPLASH_BQ_DKV=1024`, `SPLASH_BKV_DKV=1024`)
- **Loss**: Tokamax CE (`--use_tokamax_ce True`)
- **Cluster**: charles-v6e

## Results

**Status**: FAILED (Hung)
The job was terminated by the cluster runner after hitting a 60-minute hard timeout. The process stalled during the JAX/XLA compilation phase (immediately after `[opt] adamw lr=1e-05 wd=0.0`). While it successfully sharded the 8.19B parameters, the compilation of the training step involving both Splash Attention and Tokamax chunked cross-entropy either took too long or entered an infinite loop in the compiler (1198 HLO modules generated).

## Profile
(run hung before producing a profile)

## HLO Dump
(partial dump generated up to `1198` modules, but compilation never completed)

## Verdict
refuted (hung during compilation)
