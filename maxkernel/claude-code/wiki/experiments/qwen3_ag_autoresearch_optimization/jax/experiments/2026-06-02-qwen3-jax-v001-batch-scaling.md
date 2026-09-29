---
title: "Qwen3-8B jax v001 batch scaling"
type: experiment
hypothesis: qwen3-jax-batch-scaling
model: qwen3-cc-jax
variant: "8B/v6e-8"
status: refuted
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v001 batch scaling

Scaling per-chip batch size from 1 to 2 to measure MXU occupancy improvements and verify HBM headroom on TPU v6e-8.

## Hypothesis under test

**Hypothesis** (one falsifiable sentence): Raising `--batch_size` (per-chip batch > 1) lifts MXU occupancy above the baseline 19.9% with no code change, until HBM (Adam state + activations + `[B,L,V]` logits) binds.

**Mechanism** (what code / flag / kernel / sharding change the hypothesis introduces, in 1-3 sentences): The baseline ran per-chip batch 1 (global 8 on fsdp=8) at ~20% MXU. More work per step amortizes fixed per-step overhead (collectives + dispatch).

**Predicted signal** (what the profile should show if the hypothesis fired correctly — bucket attribution change, custom_call presence, module count shift, etc.): Op profile will show a significant reduction in the idle/dispatch fraction of the step time, and matmul (convolution fusion) execution time will scale sub-linearly with the batch size, resulting in a higher overall MFU (approaching 25% or more).

**Falsification criterion** (metric + threshold that would refute): Throughput/chip does not rise with batch size, or the system OOMs before any throughput gain is realized.

## Setup

- **Hardware**: TPU v6e-8 — 2 hosts × 4 chips, fsdp=8, tp=1. Cluster `<your-cluster>` (`<your-project>`, zone `<your-region>`), 1 slice.
- **Dispatch**: GKE/XPK via the gke-cluster-runner agent. Workload `alekseyv-qwen3-jax-v001-scaling`.
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:latest`
- **Env**:
  - `LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304'`
  - `JAX_COMPILATION_CACHE_DIR=gs://<your-bucket>/autoresearch/qwen3_cc/jax_lane_cache`
  - `XLA_FLAGS='--xla_dump_to=gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v001-batch-scaling/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto'`
- **Data**: synthetic random tokens (`use_real_data=False`); random-init weights.
- **Command**:
  ```bash
  cd /app/trainer && LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304' \
    JAX_COMPILATION_CACHE_DIR=gs://<your-bucket>/autoresearch/qwen3_cc/jax_lane_cache \
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 \
    XLA_FLAGS='--xla_dump_to=gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v001-batch-scaling/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto' \
    python -u train.py --model_id=Qwen/Qwen3-8B --use_real_data=False \
    --batch_size=2 --seqlen=2048 --tp_parallelism=1 \
    --train_steps=20 --weights_dtype=bf16 \
    --profile_dir=gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v001-batch-scaling \
    --profile_start_step=12 --profile_steps=3
  ```

## Results

The training workload crashed during the XLA compilation phase due to memory exhaustion (HBM Out of Memory).

- **Exit Code**: 1
- **Status**: crashed
- **Steps Seen**: 0
- **Peak HBM**: 42.99 GB (exceeded physical v6e-8 capacity of 31.25 GB)
- **Wall Time**: 99 s
- **Error Headline**: `RESOURCE_EXHAUSTED: HBM OOM during compilation (Used 42.99G of 31.25G hbm)`

## Profile

No profile trace was generated because the training workload crashed during the compilation pass prior to executing step 0.

## HLO Dump

**Source**: [hlo/](file:///mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki_ag/raw/profiles/2026-06-02-qwen3-jax-v001-batch-scaling/hlo) (and mirrored in GCS at `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v001-batch-scaling/hlo/`)

The HLO dump directory contains partial compiler intermediate representations produced before memory limits were hit. Because the compilation aborted due to HBM capacity limits, no final executable HLO was generated.

## Verdict

**Verdict**: refuted

The hypothesis was that scaling the per-chip batch size from 1 to 2 would improve performance without OOMing. This is refuted because the compilation phase encountered a fatal HBM OOM:
- **HBM Required**: 42.99 GB
- **HBM Physical Cap**: 31.25 GB
- **Exceeded by**: 11.74 GB

Therefore, batch size 2 is completely unfeasible on TPU v6e-8 for this model size (8B) and FSDP configuration without incorporating more aggressive techniques (e.g., activation checkpointing / selective rematerialization, higher TP/EP parallelism, or memory-efficient attention).

## Next hypotheses

- [Selective SparseCore offload](../../../../hypotheses/qwen3-jax-selective-sparsecore-offload.md) — Since batch scaling is blocked by HBM limits, the focus shifts to optimizing batch size 1 (the baseline) by selectively offloading FSDP collectives to the SparseCore to minimize TPU host transfer overhead.
