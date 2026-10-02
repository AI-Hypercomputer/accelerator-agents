---
title: "Qwen3 JAX v129 - shard acts s8k bs4 tokamax splash collective matmul none no customcall vmem adjust"
type: experiment
hypothesis: "Custom-call scoped VMEM adjustments may be redundant after the simplified v127 collective-matmul-none stack."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, all-gather, reduce-scatter, latency-hiding, scheduler, scoped-vmem, custom-call, ablation, selective-resources, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v129 - shard acts s8k bs4 tokamax splash collective matmul none no customcall vmem adjust

## Hypothesis under test

**Hypothesis**: v072/v073 showed
`--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true` was useful on the
pre-collective-matmul-none frontier. After the batch-4 collective-matmul-none
schedule collapsed to the stable v123/v127 HLO, the custom-call scoped-VMEM
adjuster may no longer affect the final program.

**Mechanism**: Start from the v127 simplified baseline and change only:

```text
--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=false
```

Keep TPU memory-pressure tracking enabled and the other v127 flags fixed:

```text
--xla_tpu_enable_all_experimental_scheduler_features=false
--xla_tpu_enable_scheduler_memory_pressure_tracking=true
--xla_latency_hiding_scheduler_enable_selective_resources=true
--xla_tpu_enable_latency_hiding_layer_scheduler=false
--xla_tpu_enable_sparse_core_collective_aggregator=false
--xla_tpu_all_gather_collective_matmul_mode=none
--xla_tpu_reduce_scatter_collective_matmul_mode=none
```

**Predicted signal**: If the custom-call VMEM adjuster is now redundant, v129
will preserve the exact v127 HLO/profile and 56.45k tok/s band. If it still
matters, expect a changed HLO or a slight regression in custom-call/loop-fusion
balance.

**Falsification criterion**: Compile/runtime failure, invalid loss, HBM OOM, or
clean completion below the confirmed v123/v127 band.

## Setup

GKE workload `alekseyv-qwen3-v129-bs4-noccvmem`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v129-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-customcall-vmem-adjust`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v127 stack with custom-call scoped VMEM adjustments disabled.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v129-bs4-noccvmem` completed cleanly on both workers.

- Worker0: average **56,430 tok/s** / 7,054 tok/s/chip / **40.5% MFU**;
  best observed step **56,587 tok/s**; final loss 12.0461.
- Worker1: average **56,449 tok/s** / 7,056 tok/s/chip / **40.5% MFU**;
  best observed step **56,587 tok/s**; final loss 12.0461.
- Profile window: steps 12-14 captured successfully.
- Correctness: no NaNs, normal loss trajectory, exit code 0 on both workers.

This is an in-band tie/slight regression relative to v127 and does not improve
the v123 best-observed-step frontier.

## Profile

XProf run:

```text
2026-06-04-qwen3-jax-v129-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-customcall-vmem-adjust/2026_06_04_13_06_38
```

Profile summary:

- Step-time average: **1999.6 ms**
- MXU utilization: **64.1%**
- Total profile time: **111,859.6 ms**
- Idle: **297.6 ms / 0.3%**
- Peak HBM: **26.91 GiB** of 31.25 GiB

Top device buckets:

- Convolution fusion: **60,203.1 ms / 53.8%**
- Custom-call: **29,348.7 ms / 26.2%**
  - `splash_mha_dkv_no_residuals.1`: **15,477.6 ms / 13.8%**
  - `splash_mha_fwd_residuals.2/.3`: **13,871.0 ms / 12.4%**
- Loop fusion: **11,676.7 ms / 10.4%**

The train-step program hash remains unchanged:

```text
jit_train_step(16296087512277873197)
```

## HLO Dump

Fresh dumps were emitted under:

```text
gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v129-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-no-customcall-vmem-adjust/hlo/
```

The final train-step module is `module_0267.jit_train_step.cl_854318611`.

- Optimized HLO size: **787,978 bytes**
- SHA256: `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`
- Memory report: **28,862,302,652 bytes / 26.88 GiB**

This is byte-identical to v123/v126/v127/v128. The flagfile confirms the
intended change was applied:

```text
--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments='false'
--xla_tpu_enable_scheduler_memory_pressure_tracking='true'
--xla_tpu_enable_all_experimental_scheduler_features=false
--xla_latency_hiding_scheduler_enable_selective_resources='true'
--xla_tpu_enable_latency_hiding_layer_scheduler='false'
--xla_tpu_enable_sparse_core_collective_aggregator='false'
--xla_tpu_all_gather_collective_matmul_mode=NONE
--xla_tpu_reduce_scatter_collective_matmul_mode=NONE
```

## Verdict

**Refuted / tie.** Disabling custom-call scoped VMEM adjustments is valid and
does not change the final HLO, HBM footprint, or train-step program hash, but it
does not improve throughput or profile time. Keep the v127 simplified baseline
with custom-call scoped VMEM adjustments enabled.

Next test: disable AG backward pipelining from the v127 stack. The old v067
ablation predated the collective-matmul-none/batch-4 frontier, so this remaining
scheduler/collective knob should be revalidated on the current stack.
