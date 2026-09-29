---
title: "Qwen3 JAX current v100 - 2k batch16 Splash BKV2048 DQ reduction steps 3"
type: experiment
hypothesis: "At 2k context with BKV2048, Tokamax DQ reduction steps may reduce attention custom-call work without the long-context convolution regression seen earlier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: failed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, tokamax-dq-reduction, sparsecore, collectives, scheduler, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v100 - 2k batch16 Splash BKV2048 DQ reduction steps 3

## Hypothesis under test

**Hypothesis**: v095/v096 established the BKV2048 frontier, but custom-call /
Splash remains about **13.1%** of the profiled device time. Prior long-context
`TOKAMAX_DQ_REDUCTION_STEPS=3` probes changed HLO and reduced DKV custom-call
time but regressed convolution time. At sequence length **2048** with
full-sequence BKV2048, the DQ-reduction tradeoff may be different.

**Mechanism**: Reuse v096 exactly, adding only:

```bash
export TOKAMAX_DQ_REDUCTION_STEPS=3
```

Keep `--batch_size=16`, `--seqlen=2048`, scoped VMEM **98304 KiB**, shmem100,
no-scan remat, activation sharding, Tokamax Splash, MaxText CE, SparseCore
reduce-scatter/all-reduce offload, SparseCore collective aggregator, and the
forward Splash tile overrides:

```bash
export SPLASH_BKV=2048
export SPLASH_BKV_COMPUTE=2048
```

**Predicted signal**: Support requires clean finite loss plus throughput above
the confirmed v095/v096 **70.95k tok/s** band, or a same-band run with a
material custom-call/attention profile win and no wall-time loss. Lower
throughput, worse convolution/loop time, OOM, or invalid numerics refutes
carrying DQ reduction steps.

## Setup

- Workload: `alekseyv-qwen3-jax-v100-2ksp2048dq3`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v100-2k-bs16-vmem98304-splash2048-dqsteps3`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v100_2k_bs16_vmem98304_splash2048_dqsteps3`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`
- Mesh: `fsdp=8,tp=1`; per-chip batch 16 / global batch 128; sequence length
  **2048**; profile window steps 12-14.

## Status

Failed during first train-step compile on 2026-06-13.

## Results

- Workload: `alekseyv-qwen3-jax-v100-2ksp2048dq3`
- Outcome: compile-time OOM before any measured throughput.
- Error: `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory
  in memory space hbm. Used 32.62G of 31.25G hbm. Exceeded hbm capacity by
  1.38G.`
- Runtime failure memory headline: total HBM usage at least **32.88 GiB**:
  reserved **260.00 MiB**, program **26.90 GiB**, arguments **5.72 GiB**.
- Copied train-step HLO/memory report:
  `/tmp/qwen3-v100-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Optimized train-step HLO SHA:
  `9b4f56aa346c8eff4ba2826cbd834efbfeb3db1fe51d3a8541a6916b0cda0c0d`
  (**19,559,653 bytes**, **116,098 lines**).
- HLO memory report: **32.56 GiB** total bytes used, with
  **26.80 GiB** preallocated temp. The largest temp bin includes two
  `bf16[16,2048,151936]` logit-shaped buffers and reaches **9.27 GiB**.
- HLO counts:
  `all-gather=7055`, `all-reduce=165`, `reduce-scatter=2774`,
  `async-start=397`, `async-done=397`, `convolution=3138`,
  `custom-call=3924`, `copy=17381`, `fusion=30379`, `dot_general=7347`,
  `tokamax=1`.
- The runtime Splash config still logged
  `bq=2048 bkv=2048 bq_dkv=2048 bkv_dkv=2048 fused_bwd=True bq_dq=None bkv_dq=None`;
  this means `TOKAMAX_DQ_REDUCTION_STEPS=3` did not surface as explicit DQ tile
  settings in the runtime config line.

## Verdict

Refuted. Do not carry `TOKAMAX_DQ_REDUCTION_STEPS=3` on the 2k BKV2048
frontier: it creates a distinct, larger train-step HLO and fails HBM by about
1.38 GiB before producing throughput. The next DQ-related probe should use the
explicit DQ tile environment knobs, or be combined with a memory-reducing change,
rather than this reduction-steps knob alone.
