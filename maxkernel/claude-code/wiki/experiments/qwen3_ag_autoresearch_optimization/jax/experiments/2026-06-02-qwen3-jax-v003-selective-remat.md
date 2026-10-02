---
title: "Qwen3-8B jax v003 selective rematerialization"
type: experiment
hypothesis: qwen3-jax-selective-remat
model: qwen3-cc-jax
variant: "8B/v6e-8"
status: refuted
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v003 selective rematerialization

Scaling the per-chip batch size from 1 to 2 by implementing selective activation rematerialization (`checkpoint_dots_with_no_batch_dims`) to unblock the HBM OOM limit on TPU v6e-8.

## Hypothesis under test

**Hypothesis** (one falsifiable sentence):
Wrapping the loss computation in `jax.checkpoint` with the `checkpoint_dots_with_no_batch_dims` policy will reduce activation memory sufficiently to enable training Qwen3-8B with a per-chip batch size of 2 on TPU v6e-8 without triggering HBM Out-of-Memory.

**Mechanism** (what code / flag / kernel / sharding change the hypothesis introduces, in 1-3 sentences):
We will wrap the training step loss function using `jax.checkpoint` with `policy=checkpoint_policies.checkpoint_dots_with_no_batch_dims`. This instructs the compiler to persist expensive dot-product intermediates in HBM while recomputing lightweight elementwise intermediate activations during the backward pass, avoiding the double-compute of heavy matmuls and all-gathers.

**Predicted signal** (what the profile should show if the hypothesis fired correctly — bucket attribution change, custom_call presence, module count shift, etc.):
The compilation and training step will complete successfully with `batch_size=2`. In the memory profile, peak activation memory will be significantly reduced compared to standard unrolled scaling, keeping total peak memory well within the physical limit of 31.25 GB. Step time will increase slightly at batch 1 (recomputational tax), but at batch 2 the overall MFU will increase substantially (approaching 30%+) due to better amortized overhead.

**Falsification criterion** (metric + threshold that would refute):
The training run still crashes with HBM OOM at batch size 2, or the step-time overhead is so severe that throughput/chip does not exceed the batch size 1 baseline reproduction (median step time of 526 ms) -> refuted.

## Setup

- **Hardware**: TPU v6e-8 — 2 hosts × 4 chips, fsdp=8, tp=1. Cluster `<your-cluster>` (`<your-project>`, zone `<your-region>`), 1 slice.
- **Dispatch**: GKE/XPK via the gke-cluster-runner agent. Workload `alekseyv-qwen3-jax-v003-remat`.
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-exp03-selective-remat`
- **Env**:
  - `LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304'`
  - `JAX_COMPILATION_CACHE_DIR=gs://<your-bucket>/autoresearch/qwen3_cc/jax_lane_cache`
  - `XLA_FLAGS='--xla_dump_to=gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v003-selective-remat/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto'`
- **Command**:
  ```bash
  cd /app/trainer && LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304' \
    JAX_COMPILATION_CACHE_DIR=gs://<your-bucket>/autoresearch/qwen3_cc/jax_lane_cache \
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 \
    XLA_FLAGS='--xla_dump_to=gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v003-selective-remat/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto' \
    python -u train.py --model_id=Qwen/Qwen3-8B --use_real_data=False \
    --batch_size=2 --seqlen=2048 --tp_parallelism=1 \
    --train_steps=20 --weights_dtype=bf16 \
    --profile_dir=gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v003-selective-remat \
    --profile_start_step=12 --profile_steps=3
  ```

## Results

| Metric | This run (v003) | Notes |
|---|---|---|
| Steady step time | N/A | Crashed during compilation |
| Throughput | N/A | Crashed during compilation |
| MFU | N/A | Crashed during compilation |
| Cold compile | N/A | Aborted due to OOM |
| Exit code | 1 | |

- **Peak HBM**: 33.61 GB (exceeded physical v6e-8 capacity of 31.25 GB)
- **Wall Time**: 169 s
- **Error Headline**: `RESOURCE_EXHAUSTED: HBM OOM during compilation (Used 33.61G of 31.25G hbm)`

## Profile

No profile trace was generated because the training workload crashed during the compilation pass prior to executing step 0.

- **GCS run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v003-selective-remat/`

### HLO Dump

- **GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v003-selective-remat/hlo/`

## Verdict

**Verdict**: refuted

The hypothesis was that selective rematerialization (`checkpoint_dots_with_no_batch_dims`) would reduce activation memory sufficiently to fit Qwen3-8B at a per-chip batch size of 2 on TPU v6e-8. This is refuted because the compilation pass still encountered a fatal HBM OOM:
- **HBM Required**: 33.61 GB
- **HBM Physical Cap**: 31.25 GB
- **Exceeded by**: 2.36 GB

However, this represents a major memory footprint reduction compared to standard unrolled scaling (v001):
- **v001 (Standard scaling)**: used 42.99 GB (exceeded limit by 11.74 GB)
- **v003 (Selective remat)**: used 33.61 GB (exceeded limit by 2.36 GB)
- **Memory reduction**: **9.38 GB** saved!

This confirms that selective activation rematerialization is highly effective, saving ~9.4 GB of HBM, though we are still 2.36 GB over the physical budget. To successfully scale to batch size 2, selective rematerialization must be combined with additional memory-saving strategies, such as:
1. **Fully-fused attention / flash attention** (to shrink attention activation footprint further).
2. **Aggressive Activation Checkpointing** (more layers or full remat, paying a higher compute tax).
3. **Tensor Parallelism (TP=2)** (to split parameters and activations across chips, though this reduces FSDP sharding factor and adds TP collectives).
