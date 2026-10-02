---
title: "Qwen3 JAX v136 - shard acts s8k bs4 tokamax splash collective matmul none host-transfer overlap48"
type: experiment
hypothesis: "After collective-matmul modes are disabled, a higher host-transfer overlap limit may expose extra overlap or preserve the same HLO with a small runtime win."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, host-transfer-overlap, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v136 - shard acts s8k bs4 tokamax splash collective matmul none host-transfer overlap48

## Hypothesis under test

**Hypothesis**: v135 showed lowering host-transfer overlap from 24 to 12 keeps the
frontier HLO but does not improve throughput. Revalidate the opposite direction
after collective-matmul-none: a higher overlap limit of 48 may expose more useful
overlap now that collective lowering and runtime buckets differ from the earlier
v108 pre-cmnone test.

**Mechanism**: Start from v127 and change only:

```text
--xla_tpu_host_transfer_overlap_limit=48
```

Keep the v127 simplified stack fixed:

```text
--xla_tpu_scoped_vmem_limit_kib=81920
--xla_tpu_enable_tpu_custom_call_scoped_vmem_adjustments=true
--xla_latency_hiding_scheduler_enable_selective_resources=true
--xla_tpu_enable_all_experimental_scheduler_features=false
--xla_tpu_enable_scheduler_memory_pressure_tracking=true
--xla_tpu_enable_ag_backward_pipelining=true
--xla_tpu_scheduler_percent_shared_memory_limit=100
--xla_latency_hiding_scheduler_rerun=3
--xla_max_concurrent_host_send_recv=100
--xla_tpu_enable_async_collective_fusion=true
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true
--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true
--xla_tpu_enable_async_collective_fusion_multiple_steps=true
--xla_tpu_enable_latency_hiding_layer_scheduler=false
--xla_tpu_enable_sparse_core_collective_aggregator=false
--xla_tpu_all_gather_collective_matmul_mode=none
--xla_tpu_reduce_scatter_collective_matmul_mode=none
```

**Predicted signal**: If the post-cmnone stack benefits from more host-transfer
overlap, v136 should exceed v127's 56,453 tok/s best-worker average or v123's
56,599 best-observed step without invalid loss. If it is only a noise knob, the
HLO/profile will remain byte-identical and throughput will be in-band.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with regression, or clean completion below the confirmed v123/v127 band.

## Setup

GKE workload `alekseyv-qwen3-v136-bs4-hto48`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v136-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-transfer-overlap48`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v127 stack with only host-transfer overlap limit raised from 24 to 48.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v136-bs4-hto48` completed cleanly on both workers with
normal loss ending at 12.0461.

- **Worker 0**: 56,456 tok/s avg, 7,057 tok/s/chip, 40.5% MFU; best steady
  step 56,589 tok/s.
- **Worker 1**: 56,423 tok/s avg, 7,053 tok/s/chip, 40.5% MFU; best steady
  step 56,585 tok/s.
- **Profile dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v136-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-transfer-overlap48`

Worker0 is only 3 tok/s above v127's 56,453 tok/s best-worker average, and the
best steady step remains below v123's 56,599 tok/s best observed. Treat this as
a noise-level provisional tie pending rerun.

## Profile

- **XProf run**:
  `2026-06-04-qwen3-jax-v136-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-transfer-overlap48/2026_06_04_14_33_34`
- **Step time avg**: 2001.1 ms
- **MXU utilization**: 64.1%
- **Total time**: 111,876.0 ms
- **Idle**: 343.4 ms / 0.3%
- **Peak HBM**: 26.91 GiB
- **Train-step program**: `jit_train_step(16296087512277873197)`

Top train-step buckets:

- `convolution fusion`: 60,202.0 ms / 53.8%
- `custom-call`: 29,348.5 ms / 26.2%
  - `splash_mha_dkv_no_residuals.1`: 15,477.6 ms / 13.8%
  - `splash_mha_fwd_residuals.*`: 13,870.9 ms / 12.4%
- `loop fusion`: 11,670.8 ms / 10.4%
- `all-reduce-scatter fusion`: 3,761.7 ms / 3.4%

## HLO Dump

- **HLO prefix**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v136-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-transfer-overlap48/hlo/`
- **Optimized train-step**:
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- **HLO bytes**: 787,978
- **HLO SHA256**:
  `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`
- **Memory report**: 28,862,302,652 bytes / 26.88 GiB

The optimized train-step is byte-identical to the v123/v126/v127/v135 frontier
HLO. The flagfile confirms `--xla_tpu_host_transfer_overlap_limit=48` with the
rest of the v127 simplified stack held fixed.

## Verdict

**Provisional tie.** Raising host-transfer overlap limit from 24 to 48 keeps the
frontier HLO/profile shape and produces one worker average 3 tok/s above v127,
but the effect is far too small to carry without confirmation and the best
steady step remains below v123. Rerun overlap48 before deciding whether this is
noise or a real scalar preference.
