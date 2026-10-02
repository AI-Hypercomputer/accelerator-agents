---
title: "Qwen3 jax — Splash VMEM + Tokamax CE (bs=8, seqlen=8k)"
type: experiment
hypothesis: "The `v056-f` run showed that limiting Splash attention tile sizes (SPLASH_BKV=512) resolves the VMEM OOM, but compiling with pure Splash without Tokamax CE triggers a 132GB HBM OOM due to large logits materialization. Adding Tokamax CE back alongside the Splash VMEM tunings should allow the model to fit into the v6e-8 HBM."
model: qwen3-cc-jax
variant: 8B/v6e-8
commit: ""
status: running
tags: [qwen3, jax, tokamax, splash, vmem, seq8k]
created: 2026-06-15
updated: 2026-06-15
---

# Splash VMEM tuning + Tokamax CE to resolve Pallas OOM
**Date**: 2026-06-15

## Hypothesis under test
**Hypothesis**:
1. The `v055` experiment failed with a VMEM OOM (`35.63M` vs `32.00M` scoped limit) during compilation of the `splash_mha_dkv_no_residuals` custom call. By reducing the splash attention block sizes (`bq_dkv`, `bkv_dkv`, etc.) via environment variables, we lowered the static VMEM allocation requirement.
2. The `v056-f` run verified that Splash compilation succeeds with these limits, but then failed with a 132GB HBM allocation request due to the `[batch=4, seq=8192, vocab=151936]` logits materialization.
3. Adding Tokamax CE back into the loop (which was missing in `v055`'s codebase) alongside the Splash tuning should keep both VMEM and HBM in bounds.

**Mechanism**:
- `SPLASH_BKV=512 SPLASH_BKV_COMPUTE=512 SPLASH_BKV_DKV=512 SPLASH_BKV_DKV_COMPUTE=512` applied to launch command.
- `--use_tokamax_ce True` added back to `train.py`.

**Predicted signal**: The compilation will succeed (VMEM fits within 32MB, HBM fits within 31GB) and the workload will execute and generate an xprof profile.

**Falsification criterion**: Compilation still fails with VMEM OOM, HBM OOM, or the XLA compiler hangs (like it did in `v050` when combining Splash and Tokamax).

## Setup
* Cluster: `<your-cluster>`
* Same docker image as `v047` baseline, but injecting custom `train.py` from `qwen3-jax-v055-ac-selective-remat`.
* Batch size: 8

## Results
(To be updated)
