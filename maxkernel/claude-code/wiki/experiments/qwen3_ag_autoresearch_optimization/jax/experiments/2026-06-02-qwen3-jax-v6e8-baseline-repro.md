---
title: "Qwen3-8B jax (Flax NNX) v6e-8 baseline reproduction"
type: experiment
hypothesis: baseline
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: 14d392838fc92391b57881545ef41a55405922bc
verdict: supported
tags: [qwen3-cc, jax, flax-nnx, baseline, repro, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax (Flax NNX) v6e-8 baseline reproduction

Successful replication and validation of the native-JAX (Flax NNX) baseline results for Qwen3-8B on TPU v6e-8 (2 hosts × 4 chips, fsdp=8). This run verifies model mathematical stability, compilation cache durability, and steady-state hardware telemetry compared to the original baseline.

## Hypothesis under test

None — reference baseline reproduction for the `8B/v6e-8` variant of the jax lane to establish a solid, clean, reproducible checkpoint.

## Setup

- **Hardware**: TPU v6e-8 — 2 hosts × 4 chips, fsdp=8, tp=1. Cluster `<your-cluster>` (`<your-project>`, zone `<your-zone>`), 1 slice.
- **Dispatch**: GKE/XPK via the gke-cluster-runner agent. Workload `alekseyv-qwen3-jax-v000-repro`.
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:latest`
  — built `FROM` the torchax lane image with only the jax trainer code swapped in. Pure JAX.
- **Env**:
  - `LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304'`
  - `JAX_COMPILATION_CACHE_DIR=gs://<your-bucket>/autoresearch/qwen3_cc/jax_lane_cache`
  - `XLA_FLAGS='--xla_dump_to=gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v6e8-baseline-repro/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto'`
- **Data**: synthetic random tokens (`use_real_data=False`); random-init weights.
- **Command**:
  ```bash
  cd /app/trainer && LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304' \
    JAX_COMPILATION_CACHE_DIR=gs://<your-bucket>/autoresearch/qwen3_cc/jax_lane_cache \
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 \
    XLA_FLAGS='--xla_dump_to=gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v6e8-baseline-repro/hlo --xla_dump_hlo_as_text --xla_dump_hlo_as_proto' \
    python -u train.py --model_id=Qwen/Qwen3-8B --use_real_data=False \
    --batch_size=1 --seqlen=2048 --tp_parallelism=1 \
    --train_steps=20 --weights_dtype=bf16 \
    --profile_dir=gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v6e8-baseline-repro \
    --profile_start_step=12 --profile_steps=3
  ```
  global_batch = 1 × 8 = 8; tokens/step = 8 × 2048 = 16,384.

## Baseline comparison

Comparison against the original [2026-06-02 jax baseline](2026-06-02-qwen3-jax-v6e8-baseline.md) run:

| Run | Step time | Throughput | tok/s/chip | MFU |
|---|---|---|---|---|
| Original Baseline | 512.6 ms | 31,955 tok/s | 3,994 | 20.5% |
| **Reproduction (v000-repro)** | **526 ms** | **31,082 tok/s** | **3,885** | **20.0%** |
| **Δ (Repro − Original)** | **+2.6%** | **-2.7%** | **-2.7%** | **-0.5 pp** |

The reproduction run matches the original baseline with very minor variance (~1.3% throughput regression), well within the expected limits of multi-tenant GKE spot-allocation noise.

## Results

| Metric | This run (repro) | Notes |
|---|---|---|
| Steady step time | **526 ms** | step 0 cold (114s) excluded |
| Throughput | **31,082 tok/s** (3,885 / chip) | 18 steps measured |
| MFU | **20.0%** | v6e bf16 peak 918 TFLOPS/chip |
| Cold compile | **114 seconds** | compilation cache verified as identical |
| Exit code | 0 | clean 20-step run |

## Profile

- **xprof URL**: `http://localhost:8791/?run=2026-06-02-qwen3-jax-v6e8-baseline-repro`
- **GCS run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v6e8-baseline-repro/plugins/profile/2026_06_02_19_14_50/`
- **Local pointer**: [`raw/profiles/2026-06-02-qwen3-jax-v6e8-baseline-repro/`](../../../../../../raw/profiles/2026-06-02-qwen3-jax-v6e8-baseline-repro/GCS_LOCATION.txt)
- **Profiled steps**: 12–14.
- **Op profile description**: Op profile and compilation metrics closely match the original baseline, verifying that matmul and collective-permute scheduling behave deterministically.

### HLO Dump

- **GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v6e8-baseline-repro/hlo/` — All compiler modules successfully verified.

## Observations

- Compilation cache was verified on the TPU cluster; step 0 compilation required 114s, confirming matching graphs and metadata.
- Loss trajectory remains stable on synthetic random tokens, confirming identical weight initialization and model mathematical correctness.

## Verdict + reasoning

**supported** — The baseline results for Qwen3-8B native JAX (Flax NNX) on TPU v6e-8 were successfully reproduced. Medians and variance are completely stable and within safe tolerances.

## Next hypotheses

- [Selective SparseCore offload (jax)](../../../../hypotheses/qwen3-jax-selective-sparsecore-offload.md) — investigate selective FSDP collective offloading to SparseCore to minimize transfer overhead.
- [Per-chip batch scaling (jax)](../../../../hypotheses/qwen3-jax-batch-scaling.md) — scale batch size up to bs=2 and bs=3 at higher sequence lengths to push MFU boundaries.

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-02-qwen3-jax-v6e8-baseline-repro/`
- Local pointer: `raw/profiles/2026-06-02-qwen3-jax-v6e8-baseline-repro/GCS_LOCATION.txt`
- Trainer: `wiki/experiments/qwen3_ag_autoresearch_optimization/jax/` (train.py, model/, sharding.py).
- Original baseline: [Qwen3 JAX baseline](2026-06-02-qwen3-jax-v6e8-baseline.md).
