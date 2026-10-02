---
title: "Exp 55 — Scan-Over-Layers at seq=2048 b=1 fp32-master (CRASH: OOM)"
type: experiment
tags: [experiment, gemma4, jax, scan, fp32-master, crash, oom]
hypothesis: scan-over-layers-fp32-master
model: gemma4-e4b-torchax-jax
created: 2026-05-12
updated: 2026-05-12
commit: pending
verdict: crash
---

Attempted to apply `JAX_SCAN_LAYERS=1` (scan-over-layers, implemented during Exp 49/50) to the new mixed-precision AMP regime (`seq=2048`, `batch=1`, `fp32` master weights + `bf16` compute).

**Result: CRASH (OOM at compile time).**
`jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: E1000: CompileTimeHbmOom: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 31.91G of 31.25G hbm. Exceeded hbm capacity by 681.08M.`

## Hypothesis under test

Scan-over-layers dramatically reduces compile time (180s -> 70s as seen in Exp 49). Applying it to the new `fp32-master` regime baseline (`Exp 52`: `seq=2048`, `batch=1`) should yield the same compile-time benefits and allow us to measure if the TPS penalty previously observed (-16%) persists or changes under the mixed-precision memory constraints.

## Setup

- **Hardware:** TPU v6e-4 (4 chips, 31.25 GiB HBM/chip)
- **Env:** `JAX_COMPILATION_CACHE_DIR=/tmp/jax_compile_cache JAX_ATTENTION_IMPL=splash JAX_SCAN_LAYERS=1`
- **Command:**
  ```bash
  JAX_COMPILATION_CACHE_DIR=/tmp/jax_compile_cache JAX_ATTENTION_IMPL=splash JAX_SCAN_LAYERS=1 \
    python -u -m train --steps 20 --batch_size 1 --seq_len 2048 \
    --weights-dtype fp32 --compute-dtype bf16 \
    --profile_dir /mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki/raw/profiles/2026-05-12-gemma4-jax-exp55-scan-layers-fp32master \
    --profile_steps 10 11 12
  ```

## Results & Analysis

The run crashed during `step 0` HLO compilation.

- **Baseline (`Exp 52` without Scan):** Fit comfortably enough to run, but peaked at **30.89 GiB / 31.25 GiB (98.85% utilization)** during execution.
- **This Run (`Exp 55` with Scan):** The XLA compiler required **11.06 GiB** for the program and **20.85 GiB** for arguments/outputs, pushing the total requirement to **31.91 GiB** (exceeding the limit by ~681 MB).

**Mechanism:**
The inner `jax.checkpoint` policy `checkpoint_dots_with_no_batch_dims` used inside the scan body requires slightly more temporary memory to trace and execute than the fully unrolled graph under the global checkpoint policy in the new `fp32-master` regime. Since the `seq=2048` `fp32-master` configuration was already operating at 98.8% memory capacity (leaving only ~350 MB free), any structural change to the HLO graph that incurs even minor overhead results in an immediate Out-Of-Memory failure.

## Profile

- No execution trace was produced because the run crashed during compilation. The attempted log directory `2026-05-12-gemma4-jax-exp55-scan-layers-fp32master` remains empty of XPlane data.

## Verdict

**CRASH.** The `scan-over-layers` implementation cannot be run at the current `seq=2048, batch=1, fp32-master` shape on v6e-4 due to strict memory limits.

## Next Hypotheses

Per the "win memory before throughput" heuristic:
1. **exp 56 — 2D mesh (dp=2, tp=2):** Switch to tensor parallelism to replicate K/V and see if it drops the overall parameter/activation footprint enough to unblock `seq=8192` or `batch=2`.
2. **exp 57 — Offload PLE embedding to host:** The `embed_tokens_per_layer` tensor is 11 GiB in `fp32`. Streaming it from the host could instantly free up the ~1-4 GiB needed to unblock scaling.
