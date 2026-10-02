---
title: "Qwen3 JAX v271 - Selective-resources frontier with RS5+AR2"
type: experiment
hypothesis: "Selective resources changed the v264 RS4+AR2 schedule family and confirmed a new v269 frontier; raising only the SparseCore reduce-scatter latency multiplier from 4 to 5 may improve overlap/resource assignment within this new family."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs5-ar2-selective-resources
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, scheduler, selective-resources, latency, rs5, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v271 - Selective-resources frontier with RS5+AR2

## Hypothesis under test

**Hypothesis**: v269 confirms that forcing selective resources on the MLP
up-proj-first + RS4+AR2 stack selects a faster HLO (**60,338 tok/s / 43.3%
MFU**, **4359.6 ms / 68.2% MXU**). Because selective resources changed the HLO
family, the pre-selective RS latency bracket may no longer be closed. Raising
only `--xla_tpu_sparse_core_reduce_scatter_latency_multiplier` from `4` to `5`
may improve SparseCore reduce-scatter placement/overlap on the new schedule.

**Mechanism**: Reuse v269 exactly, changing only RS latency multiplier from
`4` to `5`. Keep selective resources forced on, AR latency multiplier `2`,
scoped VMEM `100352`, SparseCore reduce-scatter/all-reduce offload,
SparseCore collective aggregator, Tokamax Splash, MaxText CE, no-scan, remat,
and activation sharding.

**Predicted signal**: Support requires clean finite loss and either throughput
above v269's confirmed **60,338 tok/s / 43.3% MFU**, or same-band throughput
with XProf step time below **4359.6 ms** and no additional HBM pressure.

**Falsification criterion**: Compile/runtime failure, invalid loss, missing
Tokamax markers, OOM or HBM allocator instability, throughput/profile
regression, or increased HBM without speedup. Because v269 has only
**0.0896 GiB** free HBM, any memory increase must buy a clear step-time gain.

## Setup

GKE workload: `alekseyv-q3-v271-sr-rs5-ar2`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Image digest**: `sha256:3a08bd533913355b79d8d9c27243646bf8dd76809878cf41372fee0c6407071e`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v271-no-scan-sparsecore-rs-ar-rs5-ar2-mlp-up-proj-first-selective-resources`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Runtime scheduler isolate only. Model code, params, math, random fake-data
setup, MLP up-proj-first image, and all runtime flags match v269 except
`--xla_tpu_sparse_core_reduce_scatter_latency_multiplier`.

## Results

Workload `alekseyv-q3-v271-sr-rs5-ar2` failed before step 0 during
`jit_train_step` program load. Worker1 reported:

```text
RESOURCE_EXHAUSTED: Error loading program 'jit_train_step': Attempting to
allocate 2.12G. That was not possible. There are 995.89M free.
```

The container exited with `EXIT_CODE=1`. No throughput summary was produced.
This is a direct memory failure on the v269 selective-resources frontier after
changing only RS latency multiplier `4 -> 5`.

## Profile

No XProf was produced because the run failed before step 0 and before the
configured profile window.

## HLO Dump

Optimized train-step HLO:
`/tmp/qwen3-v271-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.

- SHA256:
  `a88aac126db1ab1d3d04db6f74b572b2f51091637084b82d470246003e9a7d8a`
- Size: **19,874,947 bytes / 119,032 lines**.
- Compared with v269: `cmp_v269=1`, changed HLO.
- Counts: `all-gather` 6976, `all-reduce` 165, `reduce-scatter` 2774,
  `async-start`/`async-done` 364/364, `convolution` 3094, `custom-call` 3819,
  `fusion` 30813, `copy` 22221, `tokamax` 2, `jvp_jit_silu__` 580.
- Flagfile confirmed the intended isolate:
  `--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=5`,
  `--xla_tpu_sparse_core_all_reduce_latency_multiplier=2`,
  `--xla_latency_hiding_scheduler_enable_selective_resources='true'`, and
  `--xla_tpu_scoped_vmem_limit_kib=100352`.
- XLA memory usage report shows **29.14 GiB** total bytes used in the static
  allocation report, led by **23.38 GiB** preallocated temp plus many 768 MiB
  `bf16[4,8192,12288]` remat/MLP temporaries; runtime loading still failed
  due to insufficient free HBM for an additional **2.12 GiB** allocation.

## Verdict

**Invalid/refuted.** RS5 under selective resources selects a distinct HLO but
cannot load on v6e-8 with the v269 memory budget. This closes the high-side RS
latency multiplier probe for the selective-resources frontier. Keep RS latency
multiplier `4`; next bracket probe should test the low side (`RS3+AR2` with
selective resources) because RS5 fails before step 0.
