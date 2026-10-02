---
title: "Qwen3 JAX current v119 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 scheduler rerun 4"
type: experiment
hypothesis: "Increasing latency-hiding scheduler reruns from 3 to 4 on the v114 frontier may find a better schedule without changing the closed collective latency multipliers."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, reduce-scatter, scheduler, latency-model, scheduler-rerun, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v119 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 scheduler rerun 4

## Hypothesis under test

**Hypothesis**: The collective latency multiplier brackets are closed around
v114's reduce-scatter latency 4 and all-reduce latency 2. Increasing
`--xla_latency_hiding_scheduler_rerun` from 3 to 4 may let the scheduler find a
better schedule without moving those latency-model settings.

**Mechanism**: Reuse v114 exactly but change:

```bash
--xla_latency_hiding_scheduler_rerun=4
```

Keep all other v114 settings fixed.

## Setup

- Workload: `alekseyv-qwen3-v119-2krsl4rer4`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v119-2k-bs16-vmem98304-splash2048-rslat4-rerun4`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v119_2k_bs16_vmem98304_splash2048_rslat4_rerun4`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Worker0 summary: 71,047 tok/s global, 8,881 tok/s/chip, approx 45.7% MFU.
- Worker1 summary: 71,053 tok/s global, 8,882 tok/s/chip, approx 45.7% MFU.
- Profile: `2026_06_13_08_48_33`.
- XProf summary: 3718.9 ms average step time, 67.9% MXU utilization.
- XProf op profile: 89,130.053 ms total, 186.293 ms idle, 0.2% idle.
- Runtime memory: 31.14 GiB / 31.25 GiB peak, with 0.1016 GiB free and
  1.92% fragmentation.
- Optimized train-step HLO:
  `b03d411afe24c9ff0ed55829979aa76f3b9d3b8de4932139e8d398f29a54eff3`,
  19,351,131 bytes / 115,235 lines.
- HLO counts: `all-gather=7063`, `all-reduce=165`,
  `reduce-scatter=2774`, `async-start=373`, `async-done=373`,
  `convolution=3071`, `custom-call=3828`, `copy=17285`,
  `fusion=30238`, `dot_general=7288`, `tokamax=1`.
- Compiled memory: 27.88 GiB total bytes used, with 22.12 GiB
  preallocated temp.

## Verdict

Promoted as the current measurement best, but not as a distinct HLO frontier.
The optimized train-step HLO is byte-identical to v114, so rerun 4 did not
produce a new schedule. Treat the 71,053 tok/s result as run-to-run best for
the v114 graph; do not spend more trials on scheduler rerun depth unless a
different graph-changing mechanism makes it relevant.
