---
title: "Qwen3-8B jax v002 selective SparseCore offload"
type: experiment
hypothesis: qwen3-jax-selective-sparsecore-offload
model: qwen3-cc-jax
variant: "8B/v6e-8"
status: completed
verdict: invalid
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v002 selective SparseCore offload

Investigating whether offloading only a subset of FSDP collectives (specifically `reduce-scatter` only) to the SparseCore can reclaim TensorCore compute cycles without introducing excessive PCIe transfer and synchronization overhead.

## Hypothesis under test

**Hypothesis** (one falsifiable sentence):
Offloading only the `reduce-scatter` FSDP collective to SparseCore will minimize SparseCore-to-TensorCore transfer and synchronization latency, while freeing up TensorCore cycles for matmul execution.

**Mechanism** (what code / flag / kernel / sharding change the hypothesis introduces, in 1-3 sentences):
We will pass `--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true` via `XLA_FLAGS`, while keeping all-reduce and all-gather collectives on TensorCore. This targets the backward-pass gradient reduction scatter, which is typically a major bottleneck, without incurring the synchronization overhead of offloading all three collectives.

**Predicted signal** (what the profile should show if the hypothesis fired correctly — bucket attribution change, custom_call presence, module count shift, etc.):
The overall step time should decrease compared to the baseline (`526 ms` for reproduction). In the op profile, we expect the time spent on collective communication wait states (`collective-permute-done`) to be reduced or better overlapped with backward-pass matmuls (`convolution fusion`).

**Falsification criterion** (metric + threshold that would refute):
Step time does not decrease compared to the baseline reproduction (526 ms), or further degrades performance → refuted.

## Setup

- **Hardware**: TPU v6e-8 — 2 hosts × 4 chips, fsdp=8, tp=1. Cluster `<your-cluster>` (`<your-project>`, zone `<your-region>`), 1 slice.
- **Dispatch**: GKE/XPK via the gke-cluster-runner agent. Workload `alekseyv-qwen3-jax-v002-sc-sel`.
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:latest`
- **Env**:
  - `LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304'`
  - `JAX_COMPILATION_CACHE_DIR=gs://<your-bucket>/autoresearch/qwen3_cc/jax_lane_cache`
  - `XLA_FLAGS='--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_dump_to=gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v002-selective-sc-offload/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto'`
- **Command**:
  ```bash
  cd /app/trainer && LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304' \
    JAX_COMPILATION_CACHE_DIR=gs://<your-bucket>/autoresearch/qwen3_cc/jax_lane_cache \
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 \
    XLA_FLAGS='--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_dump_to=gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v002-selective-sc-offload/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto' \
    python -u train.py --model_id=Qwen/Qwen3-8B --use_real_data=False \
    --batch_size=1 --seqlen=2048 --tp_parallelism=1 \
    --train_steps=20 --weights_dtype=bf16 \
    --profile_dir=gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v002-selective-sc-offload \
    --profile_start_step=12 --profile_steps=3
  ```

## Results

The workload crashed immediately during startup due to environmental/flag incompatibility.

- **Exit Code**: 1
- **Status**: crashed
- **Steps Seen**: 0
- **Peak HBM**: N/A
- **Wall Time**: 5 s
- **Error Headline**: `F0602 21:03:02.305320 315 parse_flags_from_env.cc:234] Unknown flag in XLA_FLAGS: --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true`

## Profile

No profile trace was generated because the process crashed immediately on flag parsing.

- **xprof URL**: N/A
- **GCS run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v002-selective-sc-offload/`
- **Local pointer**: `raw/profiles/2026-06-02-qwen3-jax-v002-selective-sc-offload/`

### HLO Dump

No HLO dump was generated because the process crashed prior to any compilation pass.

- **GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v002-selective-sc-offload/hlo/`

## Verdict

**Verdict**: invalid

The flag `--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true` is not recognized by the XLA compiler version packaged inside the current Docker image (`<your-registry>/torchtitan-images/qwen3-8b-jax:latest`).

## Next hypotheses

- [Selective rematerialization](../../../../hypotheses/qwen3-jax-selective-remat.md) — Pivot to memory-saving techniques to unblock batch size scaling. Since `batch_size=1` is already at 93.15% HBM usage, implementing custom checkpointing of attention/MLP layers will allow scaling to `batch_size=2`.
