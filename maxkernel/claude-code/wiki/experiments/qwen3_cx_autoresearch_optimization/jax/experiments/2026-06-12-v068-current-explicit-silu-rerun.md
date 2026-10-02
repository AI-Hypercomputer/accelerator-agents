---
title: "Qwen3 JAX current v068 - explicit SwiGLU SiLU rerun"
type: experiment
hypothesis: "The v067 explicit gate * sigmoid(gate) MLP rewrite is a real small improvement over v064 rather than run-to-run noise."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, silu, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, scheduler, compile-cache, long-seq, v6e-8, rerun]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v068 - explicit SwiGLU SiLU rerun

## Hypothesis under test

**Hypothesis**: v067's explicit SwiGLU algebra result (**59,347 tok/s /
4432.5 ms**) is a durable improvement over the v064/v037 fresh-cache baseline
(**59,326 tok/s / 4433.8 ms**), not a one-run high-side sample.

**Mechanism**: Run the exact v067 image again on the same preferred np0
placement with a fresh compile cache and fresh profile/HLO output directory.
Do not change source, sharding, remat, CE, Splash, scheduler flags, batch size,
or placement.

**Predicted signal**: Support requires a clean finite-loss completion in the
same or better band as v067 and the same HLO family
(`jvp_jit_silu__=0`, lean `copy=9918/fusion=22711`). A clean result falling
back to v064 or below treats v067 as noise/tie rather than a carried source
change.

## Setup

- Workload: `alekseyv-qwen3-cc5-jax-v068-silur`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Image digest:
  `sha256:d752d2d9720ddc54602c2476689a40c20d322386c47339f047dbf7efa60ede21`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v068-v067-explicit-silu-rerun`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v068_silu_rerun`
- Placement: nodepool `<your-cluster>-np-0`
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Status

Launched after recording v067 as provisional support.

## Result

The workload completed cleanly with finite loss on np0.

Worker summaries:

- Host `gke-tpu-0d0013c1-l5jf`: **59,103 tok/s**, **7,388/chip**,
  **42.4% old MFU**. This average includes a post-profile step-15 outlier at
  **4672.1 ms / 56,108 tok/s**.
- Host `gke-tpu-0d0013c1-9rkl`: **59,318 tok/s**, **7,415/chip**,
  **42.5% old MFU**.

Representative steady steps outside compile/warmup were in the same band as
v064/v067:

- Step 2: **4411.1 ms / 59,428 tok/s**
- Step 3: **4413.2 ms / 59,400 tok/s**
- Step 16: **4419.6 ms / 59,314 tok/s**
- Step 19: **4419.4 ms / 59,317 tok/s**

XProf:

- Run: `2026-06-12-qwen3-jax-v068-v067-explicit-silu-rerun/2026_06_12_14_43_32`
- Average step time: **4433.8 ms**
- Step-time stddev: **1.0 ms**
- MXU utilization: **66.9%**
- Peak HBM: **29.03 GiB / 31.25 GiB**, **2.2202 GiB** free
- Op profile: convolution fusion **56,929.609 ms / 53.5%**,
  custom fusion **3,848.180 ms / 3.6%**, idle **0.3%**. The full op profile
  still places the train step in the same convolution/Splash/loop-fusion shape
  as v064-v067.

Fresh train-step HLO was copied to
`/tmp/qwen3-v068-hlo/train_step.after_optimizations.txt`:

- SHA256:
  `84df522a5e6dfdd158288d856af82a5b24506c30db8c0fb29dfb46c952378a38`
- Size: **19,895,815 bytes / 119,353 lines**
- Counts: `all-gather=4628`, `all-reduce=158`, `reduce-scatter=1766`,
  `async-start=394`, `async-done=394`, `convolution=2106`,
  `custom-call=2256`, `copy=9918`, `fusion=22711`, `dot_general=7122`,
  `splash=2211`, `tokamax=2`, `jvp_jit_silu__=0`, `sigmoid=0`,
  `logistic=0`, `exponential=183`, `exp.=180`.

The HLO is byte-identical to v067 and differs from v064, confirming that the
explicit MLP SiLU algebra reliably selects the intended HLO family.

## Verdict

**Tie / not confirmed as a throughput win.** v068 reproduced the v067 HLO
exactly, but XProf landed at **4433.8 ms**, matching v064 rather than v067's
single **4432.5 ms** high-side sample. Treat explicit SiLU algebra as a valid
semantics-preserving cleanup that removes `jvp_jit_silu__`, but do not count it
as a durable MFU improvement over the v064/v037 frontier.
