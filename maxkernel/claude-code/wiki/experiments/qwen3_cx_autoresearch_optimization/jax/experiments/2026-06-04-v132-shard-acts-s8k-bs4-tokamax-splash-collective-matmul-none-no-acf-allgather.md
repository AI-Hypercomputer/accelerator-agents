---
title: "Qwen3 JAX v132 - shard acts s8k bs4 tokamax splash collective matmul none no ACF all-gather"
type: experiment
hypothesis: "After collective-matmul modes are disabled, the all-gather async-collective-fusion subtype may still be required; revalidate it on the simplified v127 stack."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, async-collective-fusion, all-gather, latency-hiding, scheduler, ablation, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v132 - shard acts s8k bs4 tokamax splash collective matmul none no ACF all-gather

## Hypothesis under test

**Hypothesis**: v113 showed that disabling only the all-gather async collective
fusion subtype was strongly regressive on the pre-collective-matmul-none
layeragg stack. Because v116-v127 removed both collective-matmul modes and
substantially changed the final HLO/program, revalidate whether the all-gather
ACF subtype remains necessary in the current simplified frontier stack.

**Mechanism**: Start from v127 and change only:

```text
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false
```

Keep base ACF enabled and keep reduce-scatter plus multiple-step ACF enabled:

```text
--xla_tpu_enable_async_collective_fusion=true
--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true
--xla_tpu_enable_async_collective_fusion_multiple_steps=true
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

**Predicted signal**: If all-gather ACF remains required, the final HLO/program
should change, explicit all-gather cost should rise, and throughput should fall
below the 56.4k-56.6k tok/s frontier band. If collective-matmul-none removed the
pressure that made this subtype useful, the HLO may remain identical or
throughput may stay in-band.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with regression, or clean completion below the confirmed v123/v127 band.

## Setup

GKE workload `alekseyv-qwen3-v132-bs4-noacfag`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v132-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-allgather`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v127 stack with only all-gather ACF subtype disabled.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v132-bs4-noacfag` completed successfully on both
workers.

- **Worker 0**: 55,479 tok/s avg over 18 measured steps, 6,935 tok/s/chip,
  39.8% MFU; best observed steady step 55,547 tok/s; final loss 12.0462.
- **Worker 1**: 55,462 tok/s avg over 18 measured steps, 6,933 tok/s/chip,
  39.8% MFU; best observed steady step 55,538 tok/s; final loss 12.0462.
- **Profile dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v132-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-allgather`

## Profile

XProf run:
`2026-06-04-qwen3-jax-v132-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-acf-allgather/2026_06_04_13_43_58`

- **Step time average**: 2032.1 ms
- **MXU utilization**: 63.3%
- **Total device time**: 113,699.9 ms
- **Idle**: 274.2 ms / 0.2%
- **Peak HBM**: 27.10 GiB of 31.25 GiB
- **Train-step program**: `jit_train_step(16650495805246986607)`
- **Top buckets**:
  - convolution fusion: 59,470.9 ms / 52.3%
  - custom-call: 29,439.9 ms / 25.9%
  - `splash_mha_dkv_no_residuals.1`: 15,479.9 ms / 13.6%
  - forward residual Splash custom-calls: 13,960.0 ms / 12.2% combined
  - loop fusion: 11,574.8 ms / 10.2%
  - all-gather: 4,556.2 ms / 4.0%

Compared with v127, all-gather becomes an explicit 4.0% bucket, total device
time rises by about 1.87 s, MXU drops by about 0.8 pp, and peak HBM rises by
about 0.19 GiB. The program hash matches v131's no-base-ACF program hash rather
than the v127 frontier hash.

## HLO Dump

- **Train-step module**:
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **Optimized HLO size**: 632,630 bytes
- **Optimized HLO SHA256**:
  `49e2af3ecfcb130fedcaa85842d312728e433a38dd3fd89aab1b1793a6e7631f`
- **Memory report**: 29,065,508,228 bytes / 27.07 GiB
- **Flagfile confirmation**:
  `--xla_tpu_enable_async_collective_fusion=true`,
  `--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false`,
  `--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true`,
  `--xla_tpu_enable_async_collective_fusion_multiple_steps=true`,
  AG backward pipelining `true`, selective resources `true`, all-experimental
  scheduler bundle `false`, layer scheduler and sparse-core aggregator
  explicitly `false`, and both collective-matmul modes `NONE`.

## Verdict

Refuted. The all-gather async collective fusion subtype remains required after
collective-matmul modes are disabled. Disabling only that subtype is enough to
select the same HLO/program hash as the v131 no-base-ACF run, expose a 4.0%
all-gather bucket, and regress throughput from the 56.4k-56.6k tok/s frontier
band to about 55.5k tok/s / 39.8% MFU.

Do not carry this ablation. Return to v127 and continue the subtype sweep by
disabling only reduce-scatter ACF.
