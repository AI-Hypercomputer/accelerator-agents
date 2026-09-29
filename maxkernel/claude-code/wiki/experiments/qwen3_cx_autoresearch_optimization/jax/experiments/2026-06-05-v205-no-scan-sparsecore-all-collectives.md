---
title: "Qwen3 JAX v205 - no-scan SparseCore offload for AG/RS/AR"
type: experiment
hypothesis: "The confirmed no-scan frontier still carries many FSDP collectives; moving all-gather, reduce-scatter, and all-reduce onto SparseCore may free TensorCore time and improve throughput."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-all-collectives
status: supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, all-gather, reduce-scatter, all-reduce, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v205 - no-scan SparseCore offload for AG/RS/AR

## Hypothesis under test

**Hypothesis**: v182/v183 established a no-scan frontier at
58,288-58,330 tok/s / 41.8% MFU, but the unrolled HLO still contains thousands
of FSDP collective sites. The Llama3 JAX lane improved its frontier by relaying
collectives onto SparseCore. Applying the same structural idea here may free
TensorCore issue slots for matmul/custom-call work and improve steady-state
throughput.

**Mechanism**: Reuse the confirmed v183 runtime stack and change only the
collective placement flags:

- Disable async collective-fusion subtypes for all-gather, reduce-scatter, and
  all-reduce.
- Enable SparseCore collective offload for all-gather, reduce-scatter, and
  all-reduce, including offload tracing and the required SparseCore base flags.

Keep `--use_scan=False`, no-MLP-intermediate `_sac`, remat, tokamax Splash,
MaxText CE, activation sharding, batch 4/seq8192, scoped VMEM 100352,
collective-matmul modes `none`, and all other scheduler flags fixed.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v182/v183 confirmed 58,288-58,330 tok/s average / 41.8% MFU band.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the confirmed no-scan frontier.

## Setup

GKE workload `alekseyv-qwen3-v205-noscan-scall`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v205-no-scan-sparsecore-all-collectives`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14 if the run reaches training.

Changed XLA flags relative to v183:

```text
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false
--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false
--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
--xla_tpu_enable_all_gather_offload_tracing=true
--xla_tpu_enable_reduce_scatter_offload_tracing=true
--xla_tpu_enable_all_reduce_offload_tracing=true
--xla_tpu_use_tc_device_shape_on_sc=true
--xla_sc_enable_instruction_fusion=false
--xla_sc_disjoint_spmem=false
--xla_sc_disable_megacore_partitioning=true
```

## Equivalence

No model-code edit is made. This is an XLA placement/scheduling experiment only
and preserves model math.

## Results

Completed cleanly on `alekseyv-qwen3-v205-noscan-scall` with `EXIT_CODE=0`
on both workers.

- Worker 0 reported **59,338 tok/s** aggregate, **7,417 tok/s/chip**, and
  **42.5% MFU**.
- Worker 1 reported **59,319 tok/s** aggregate, **7,415 tok/s/chip**, and
  **42.5% MFU**.
- Final measured steady steps after profile export stayed in the same band:
  worker0 steps 15-19 were 59,481, 59,405, 59,351, 59,311, and 59,267 tok/s;
  worker1 steps 15-19 were 59,488, 59,401, 59,347, 59,312, and 59,271 tok/s.
- Loss decreased normally through step 19, ending at **12.0459**.

This beats the confirmed v182/v183 no-scan frontier
(58,288-58,330 tok/s / 41.8% MFU) by about **+1.7% throughput** and
**+0.7 pp MFU**.

## Profile

Profile capture completed on both hosts:

- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v205-no-scan-sparsecore-all-collectives/plugins/profile/2026_06_05_01_31_01/gke-tpu-964065d9-c180.{trace.json.gz,xplane.pb}`
- `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v205-no-scan-sparsecore-all-collectives/plugins/profile/2026_06_05_01_31_02/gke-tpu-964065d9-9vh8.{trace.json.gz,xplane.pb}`

Trace spans for `jit_train_step(2150120830615090428)` on `gke-tpu-964065d9-c180`
showed profiled-step durations around **4.425 s**, matching the stdout
throughput band for global batch 32 at seq8192.

## HLO Dump

HLO was written under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v205-no-scan-sparsecore-all-collectives/hlo/`

The two train-step modules were byte-identical:

- `module_0288.jit_train_step.cl_854318611.after_optimizations.txt`
- `module_0294.jit_train_step.cl_854318611.after_optimizations.txt`

Optimized train-step HLO SHA256:

`d67433daa35f81edbfbd0fb25d79bd4cd210fc9697bd39133cd50dfe11da9573`

Compiled memory:

- `Total bytes used: 29700367320 (27.66GiB)`

Text counts from `module_0288`:

- all-gather 5,393
- all-reduce 158
- reduce-scatter 1,766
- collective-permute 3
- convolution 2,276
- custom-call 3,041
- copy 10,956
- fusion 9,652
- `tokamax` 2
- `linear_softmax_cross_entropy_loss` 0
- `log_softmax` 0
- `_one_hot` 11

SparseCore metadata exists, but contains repeated all-gather offload factory
messages such as `Unexpected SpmemWindow offset` /
`SetSpmemWindow() :: spmem_window must be starting at 0` for all-gather
instructions. The run still compiled, executed, and improved throughput; treat
this as a compiler-metadata caveat rather than a runtime failure.

## Verdict

**Supported.** Carry the all-collective SparseCore-offload flag set as the new
provisional JAX frontier: **59,338 tok/s / 42.5% MFU** at seq8192, global
batch 32. Next step is an exact rerun to confirm that the +1.7% lift is stable
before layering another mechanism on top.
