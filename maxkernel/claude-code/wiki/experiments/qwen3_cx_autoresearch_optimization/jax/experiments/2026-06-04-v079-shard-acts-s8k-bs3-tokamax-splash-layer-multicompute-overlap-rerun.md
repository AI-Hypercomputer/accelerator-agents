---
title: "Qwen3 JAX v079 - shard acts s8k bs3 tokamax splash layer multicompute overlap rerun"
type: experiment
hypothesis: "Exact rerun of v078 to determine whether the tiny multi-compute-overlap gain is durable."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported/tie
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, scheduler, layer-scheduler, multi-compute-overlap, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, rerun, supported, tie, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v079 - shard acts s8k bs3 tokamax splash layer multicompute overlap rerun

## Hypothesis under test

**Hypothesis**: v078's `--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true`
run produced a valid but tiny single-worker high-water mark: 51,025 tok/s
versus the confirmed v074/v075 51,021 tok/s frontier. An exact rerun will show
whether this is a durable scheduler improvement or ordinary run-to-run noise.

**Mechanism**: Exact rerun of v078.

**Predicted signal**: Both workers remain in or above the v074/v075 band, with
at least one worker at or above 51,021 tok/s and no profile regression.

**Falsification criterion**: Clean completion below the v074/v075 band, or a
profile matching v078 without repeatable throughput gain.

## Setup

Planned GKE workload `alekseyv-qwen3-v079-multicomp-rerun`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v079-shard-acts-s8k-bs3-tokamax-splash-layer-multicompute-overlap-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flag stack**: v074/v075 plus
  `--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v079-multicomp-rerun` completed cleanly on
2026-06-04 with normal loss.

- Worker 0: 51,021 tok/s, 36.6% MFU
- Worker 1: 51,009 tok/s, 36.6% MFU
- Final loss: 12.0508 on both workers
- Exit code: 0

This confirms that v078's multi-compute-overlap flag is non-harmful and in the
frontier band, but it does not reproduce v078's 51,025 tok/s single-worker
high-water mark.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v079-shard-acts-s8k-bs3-tokamax-splash-layer-multicompute-overlap-rerun/2026_06_04_04_14_25`

Profile:

- MXU utilization: 56.7%
- Total time: 92,738.7 ms; idle 222.4 ms / 0.2%
- `jit_train_step`: 92,515.6 ms / 99.8%
- Convolution fusion: 45,713.4 ms / 49.3%
- Custom-call: 22,057.5 ms / 23.8%
  - `splash_mha_dkv_no_residuals.1`: 11,641.1 ms / 12.6%
  - `splash_mha_fwd_residuals.2/3`: 10,416.4 ms / 11.2%
- Loop fusion: 16,723.0 ms / 18.0%
- Data formatting: 3,768.5 ms / 4.1%
- Collective-permute-start: 919.2 ms / 1.0%
- Peak HBM: 28.32 GiB of 31.25 GiB; 2.93 GiB free; 0.05% fragmentation

## HLO Dump

No HLO objects were written for v079, consistent with a persistent
compilation-cache hit for the exact v078 graph. Use v078 HLO:

- HLO object count: 1,615
- Train-step after-optimization size: 1,375,271 bytes
- Flagfile confirms
  `--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler='true'` with
  v076 resource sharing and v077 Mosaic custom-call async fusion both false.

## Verdict

Supported/tie. The multi-compute-overlap flag is valid and stays in the
frontier band, but the rerun does not establish a durable improvement beyond
the confirmed 51,021 tok/s band.
