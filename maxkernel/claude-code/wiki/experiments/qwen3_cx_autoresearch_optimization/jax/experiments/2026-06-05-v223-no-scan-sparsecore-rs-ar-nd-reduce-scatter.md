---
title: "Qwen3 JAX v223 - no-scan SparseCore RS/AR with ND reduce-scatter offload"
type: experiment
hypothesis: "The confirmed v220/v221 frontier depends on reduce-scatter SparseCore offload; enabling the newer ND reduce-scatter SparseCore offload flag from TPU recipes may improve RS placement without re-enabling all-gather SparseCore offload."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs-ar-ndrs
status: invalid
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, all-gather, nd-reduce-scatter, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v223 - no-scan SparseCore RS/AR with ND reduce-scatter offload

## Hypothesis under test

**Hypothesis**: v220/v221 confirmed that reduce-scatter and all-reduce should
stay on SparseCore while all-gather should remain on async collective fusion.
Several newer TPU recipe stacks also enable
`--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true`. Adding
only this RS-specific flag may improve reduce-scatter scheduling/placement and
raise throughput above the v220/v221 band.

**Mechanism**: Keep v220/v221 fixed and add only
`--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true`.

**Predicted signal**: Support requires clean compile/loss and throughput above
v220/v221, preferably with lower `async-done`, lower all-reduce-scatter bucket,
or higher MXU in XProf.

**Falsification criterion**: Compile/runtime failure, invalid loss, unchanged
throughput, or regression below v220/v221.

## Setup

GKE workload `alekseyv-qwen3-v223-sc-rsar-ndrs`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v223-no-scan-sparsecore-rs-ar-nd-reduce-scatter`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4
  --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This is an XLA runtime/offload flag isolate and
preserves model math.

## Result

Failed before step 0 on both workers with `EXIT_CODE=134`.

- Worker0 pod: `alekseyv-qwen3-v223-sc-rsar-ndrs-slice-job-0-0-w66cf`
- Worker1 pod: `alekseyv-qwen3-v223-sc-rsar-ndrs-slice-job-0-1-pjqlv`

Both workers hit the same compiler abort:

```text
F0605 05:41:52 standalone_reduce_scatter_strategies.cc:1001]
Check failed: span_counts_for_colors_[i] == span_count
```

The stack includes:

```text
xla::tpu::sparse_core::collective::SinglePhaseStandaloneRSStrategy::GetSpanCount()
xla::tpu::sparse_core::collective::VerifyReduceScatter()
xla::jellyfish::ReduceScatterLegalizer::IsValidReduceScatterForSparseCoreOffload()
xla::jellyfish::ReduceScatterLegalizer::RunImpl()
```

This indicates the added ND reduce-scatter offload flag is not legal for this
native-JAX train-step reduce-scatter pattern/runtime.

## Profile

No profile was produced; the run aborted during train-step compilation before
step 0.

## HLO Dump

Partial HLO artifacts were uploaded:

- Prefix:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v223-no-scan-sparsecore-rs-ar-nd-reduce-scatter/hlo/xla-dump/`
- Size: **19.53 MiB**
- Train-step MLIR directory example:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v223-no-scan-sparsecore-rs-ar-nd-reduce-scatter/hlo/xla-dump/jit_train_step_1780638109683/`

No optimized train-step HLO text was produced because the compiler aborted in
the reduce-scatter legalizer.

## Verdict

**Invalid/refuted.** Do not carry
`--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true` into
the current native-JAX frontier. The flag causes a compiler abort in
`VerifyReduceScatter` for this shape. Keep the v220/v221 RS/AR SparseCore
offload stack without ND reduce-scatter offload.
