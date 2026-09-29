---
title: "Qwen3 JAX v131 - shard acts s8k bs4 tokamax splash collective matmul none no base async collective fusion"
type: experiment
hypothesis: "Base async collective fusion may still be required after collective-matmul modes are disabled; revalidate on the simplified v127 stack."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, async-collective-fusion, all-gather, reduce-scatter, latency-hiding, scheduler, ablation, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v131 - shard acts s8k bs4 tokamax splash collective matmul none no base async collective fusion

## Hypothesis under test

**Hypothesis**: v101 proved base async collective fusion was required on the
pre-collective-matmul-none layeragg stack because disabling it exposed all-gather
cost and regressed throughput. After both collective-matmul modes were disabled
and the final HLO simplified, the base ACF flag should be revalidated because it
may either remain required or become redundant.

**Mechanism**: Start from v127 and change only:

```text
--xla_tpu_enable_async_collective_fusion=false
```

Keep the v127 simplified stack fixed:

```text
--xla_tpu_enable_all_experimental_scheduler_features=false
--xla_tpu_enable_scheduler_memory_pressure_tracking=true
--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true
--xla_latency_hiding_scheduler_enable_selective_resources=true
--xla_tpu_enable_latency_hiding_layer_scheduler=false
--xla_tpu_enable_sparse_core_collective_aggregator=false
--xla_tpu_all_gather_collective_matmul_mode=none
--xla_tpu_reduce_scatter_collective_matmul_mode=none
```

**Predicted signal**: If ACF remains required, HLO/program hash should change,
all-gather/collective exposure should rise, and throughput should regress below
the 56.4k-56.6k tok/s band. If it is redundant post-cmnone, the final HLO may
remain byte-identical with no throughput loss.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with regression, or clean completion below the confirmed v123/v127 band.

## Setup

GKE workload `alekseyv-qwen3-v131-bs4-noacf`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v131-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-base-async-collective-fusion`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v127 stack with base async collective fusion disabled.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v131-bs4-noacf` completed successfully on both
workers.

- **Worker 0**: 55,482 tok/s avg over 18 measured steps, 6,935 tok/s/chip,
  39.8% MFU; best observed steady step 55,530 tok/s; final loss 12.0462.
- **Worker 1**: 55,459 tok/s avg over 18 measured steps, 6,932 tok/s/chip,
  39.8% MFU; best observed steady step 55,533 tok/s; final loss 12.0462.
- **Profile dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v131-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-base-async-collective-fusion`

## Profile

XProf run:
`2026-06-04-qwen3-jax-v131-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-base-async-collective-fusion/2026_06_04_13_31_02`

- **Step time average**: 2033.7 ms
- **MXU utilization**: 63.2%
- **Total device time**: 113,739.4 ms
- **Idle**: 316.5 ms / 0.3%
- **Peak HBM**: 27.10 GiB of 31.25 GiB
- **Train-step program**: `jit_train_step(16650495805246986607)`
- **Top buckets**:
  - convolution fusion: 59,436.8 ms / 52.3%
  - custom-call: 29,440.3 ms / 25.9%
  - `splash_mha_dkv_no_residuals.1`: 15,480.0 ms / 13.6%
  - forward residual Splash custom-calls: 13,960.3 ms / 12.2% combined
  - loop fusion: 11,578.4 ms / 10.2%

Compared with v127/v130, wall time increases by about 1.9 s over the profiled
train-step program, MXU drops by ~0.9 pp, and HBM rises by about 0.19 GiB.

## HLO Dump

- **Train-step module**:
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **Optimized HLO size**: 632,630 bytes
- **Optimized HLO SHA256**:
  `49e2af3ecfcb130fedcaa85842d312728e433a38dd3fd89aab1b1793a6e7631f`
- **Memory report**: 29,065,508,228 bytes / 27.07 GiB
- **Flagfile confirmation**:
  `--xla_tpu_enable_async_collective_fusion=false`,
  `--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true`,
  `--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true`,
  `--xla_tpu_enable_async_collective_fusion_multiple_steps=true`,
  AG backward pipelining `true`, selective resources `true`, layer scheduler and
  sparse-core aggregator explicitly `false`, and both collective-matmul modes
  `NONE`.

This is a real changed final program, not a cached/HLO-identical tie.

## Verdict

Refuted. Base async collective fusion remains required after collective-matmul
modes are disabled. Disabling it changes the final HLO/program hash and
regresses throughput from the 56.4k-56.6k tok/s frontier band to 55.5k tok/s
with lower MXU and slightly higher HBM.

Do not carry this ablation. Return to v127 and continue the narrower subtype
revalidation, starting with disabling only the all-gather ACF subtype.
