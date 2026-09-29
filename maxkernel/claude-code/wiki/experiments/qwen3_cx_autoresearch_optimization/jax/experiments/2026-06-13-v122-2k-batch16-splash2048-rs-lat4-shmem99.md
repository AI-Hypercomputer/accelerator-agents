---
title: "Qwen3 JAX current v122 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 shmem99"
type: experiment
hypothesis: "A very small reduction in scheduler shared-memory pressure may alter the v114/v119 schedule enough to recover a better HLO without the regressions seen at lower shmem limits."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, reduce-scatter, scheduler, latency-model, scoped-vmem, shmem99, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v122 - 2k batch16 Splash BKV2048 reduce-scatter latency 4 shmem99

## Hypothesis under test

**Hypothesis**: The wider shared-memory reductions regressed, but the current
frontier has only about 0.10 GiB runtime free. A one-point reduction from
`--xla_tpu_scheduler_percent_shared_memory_limit=100` to `99` may change the
latency-hiding schedule or memory packing while avoiding the larger shmem90-95
regressions.

**Mechanism**: Reuse the v114/v119 graph settings exactly but change:

```bash
--xla_tpu_scheduler_percent_shared_memory_limit=99
```

Keep reduce-scatter latency multiplier 4, all-reduce latency multiplier 2,
SparseCore all-gather offload disabled, aggregator enabled, batch 16, sequence
2048, and Splash BKV/BKV_COMPUTE 2048 fixed.

## Setup

- Workload: `alekseyv-qwen3-v122-2krsl4sh99`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v122-2k-bs16-vmem98304-splash2048-rslat4-shmem99`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v122_2k_bs16_vmem98304_splash2048_rslat4_shmem99`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`.

## Results

- Worker0 summary: 71,043 tok/s global, 8,880 tok/s/chip, approx 45.7% MFU.
- Worker1 summary: 71,054 tok/s global, 8,882 tok/s/chip, approx 45.7% MFU.
- Profile: `2026_06_13_09_29_38`.
- XProf summary: 3721.5 ms average step time, 68.0% MXU utilization.
- XProf op profile: 89,190.425 ms total, 203.564 ms idle, 0.2% idle.
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

Refuted as a measurement tie with no graph change. The worker1 average is
within one token/sec of v119, but XProf step time and HLO/memory are
byte-identical to the v114/v119 graph. Do not carry shmem99 as a distinct
frontier setting; keep shmem100 and move to a graph-changing VMEM midpoint.
