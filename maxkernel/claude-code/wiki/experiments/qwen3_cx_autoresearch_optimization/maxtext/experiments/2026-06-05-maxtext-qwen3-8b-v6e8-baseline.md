---
title: "MaxText Qwen3-8B v6e-8 baseline (SUPPORTED, 353.2 TFLOP/s/device, 5,896 tok/s/device, 38.5% MFU)"
type: experiment
tags: [experiment, qwen3-cc, qwen3-8b, maxtext, v6e-8, baseline, reference]
hypothesis: maxtext-qwen3-8b-reference-baseline
model: qwen3-cc-maxtext
created: 2026-06-05
updated: 2026-06-05
commit: runner-image-tsij-2026-06-02-14-00-13
status: supported
---

# MaxText Qwen3-8B v6e-8 baseline

## Hypothesis Under Test

MaxText's native `qwen3-8b` implementation can run end-to-end on the
autoresearch v6e-8 cluster at the target sequence length and produce the
standard baseline artifacts: step throughput, xplane profile, and HLO dump.

This is a lane-bring-up baseline, not an optimization claim. The initial shape
uses conservative batch 1 because Qwen3-8B's `151936` vocab creates a larger
lm-head/logit memory path than the Llama3.1-8B MaxText recipe.

## Setup

- **Cluster**: `<your-cluster>` (`<your-region>`, project
  `<your-project>`, v6e-8, 2 hosts x 4 chips)
- **Workload**: `alekseyv-qwen3-maxtext-v003`
- **Image**: `<your-registry>/alekseyv_google_com-runner:tsij-2026-06-02-14-00-13`
- **MaxText layout in image**: `/app/MaxText`
- **Model config**: `/app/MaxText/configs/models/qwen3-8b.yml`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-baseline-v003`
- **Mesh**: `ici_fsdp_parallelism=-1` (v6e-8 -> fsdp=8)
- **Global tokens/step**: `1 * 8 * 8192 = 65,536`
- **Model parameters**: `8,190,735,360`

## Command

```bash
cd /app
PYTHONPATH=/app python3 -u -m MaxText.train MaxText/configs/base.yml \
  base_output_directory=gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-baseline-v003 \
  run_name=alekseyv-qwen3-maxtext-v003 \
  model_name=qwen3-8b \
  tokenizer_path=Qwen/Qwen3-8B \
  dataset_path=gs://max-datasets-rogue \
  dataset_type=synthetic \
  steps=20 \
  per_device_batch_size=1 \
  max_target_length=8192 \
  ici_fsdp_parallelism=-1 \
  remat_policy=custom \
  decoder_layer_input=offload \
  out_proj=offload query_proj=offload key_proj=offload value_proj=offload \
  attention=flash \
  use_iota_embed=True \
  enable_checkpointing=False async_checkpointing=False \
  sa_block_q=2048 sa_block_kv=2048 sa_block_kv_compute=2048 \
  sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048 \
  sa_block_q_dq=2048 sa_block_kv_dq=2048 sa_use_fused_bwd_kernel=True \
  profiler=xplane skip_first_n_steps_for_profiler=10 profiler_steps=5 \
  upload_all_profiler_results=True \
  dump_hlo=True dump_step=1 dump_hlo_upload_all=True
```

`LIBTPU_INIT_ARGS` uses the MaxText dense recipe stack: scoped VMEM 98304,
reduce-scatter layout, data-parallel overlap, async all-gather continuation
fusion, SparseCore all-reduce offload, host-offload scheduler flags, and
disabled collective matmul.

## Results

Supported. The valid run was `alekseyv-qwen3-maxtext-v003`. It completed all
20 steps, wrote TensorBoard metrics, captured xplane profiles from both hosts,
and uploaded train-step HLO dumps from both hosts.

`v002` was an invalid config-validation attempt: MaxText rejected
`tokenizer_type=huggingface` because `qwen3-8b.yml` already sets
`tokenizer_type`. `v003` removed that duplicate override.

Steady-state window below uses steps 3-9 and 16-19, excluding step 0/1 warmup,
step 10 profile-start, and step 14 profile-stop overhead.

| Metric | Value |
|---|---:|
| Median step time | **1.3893 s** |
| Median TFLOP/s/device | **353.22** |
| Median tokens/s/device | **5,896.46** |
| Aggregate tokens/s | **47,171.65** |
| MFU | **38.48%** |
| Peak HBM from memory report | **29.07 GiB / 31.25 GiB** |
| Final loss | **0.2334** |

Per-step TensorBoard scalars:

| step | s/step | TFLOP/s/device | tok/s/device | loss | note |
|---:|---:|---:|---:|---:|---|
| 0 | 6.284 | 78.09 | 1,303.62 | 12.4336 | warmup |
| 1 | 2.382 | 206.02 | 3,439.13 | 12.4336 | warmup |
| 2 | 1.416 | 346.45 | 5,783.40 | 11.5790 | warmup tail |
| 3 | 1.387 | 353.92 | 5,908.07 | 10.3047 | steady |
| 4 | 1.388 | 353.51 | 5,901.35 | 8.8483 | steady |
| 5 | 1.389 | 353.41 | 5,899.61 | 7.4399 | steady |
| 6 | 1.388 | 353.50 | 5,901.11 | 6.1765 | steady |
| 7 | 1.389 | 353.28 | 5,897.46 | 4.9886 | steady |
| 8 | 1.389 | 353.18 | 5,895.72 | 3.9628 | steady |
| 9 | 1.389 | 353.22 | 5,896.46 | 3.0782 | steady |
| 10 | 1.499 | 327.43 | 5,465.96 | 2.3485 | profile start |
| 11 | 1.391 | 352.75 | 5,888.65 | 1.7639 | profiled |
| 12 | 1.392 | 352.66 | 5,887.08 | 1.3135 | profiled |
| 13 | 1.392 | 352.66 | 5,887.02 | 0.9772 | profiled |
| 14 | 9.166 | 53.54 | 893.71 | 0.7296 | profile stop/write |
| 15 | 1.413 | 347.23 | 5,796.44 | 0.5514 | post-profile |
| 16 | 1.393 | 352.32 | 5,881.40 | 0.4252 | steady |
| 17 | 1.390 | 353.02 | 5,893.11 | 0.3370 | steady |
| 18 | 1.391 | 352.77 | 5,888.86 | 0.2759 | steady |
| 19 | 1.391 | 352.91 | 5,891.33 | 0.2334 | steady |

## Profile

- `gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-baseline-v003/alekseyv-qwen3-maxtext-v003/tensorboard/plugins/profile/2026_06_05_01_13_31/`
- Files: two `.xplane.pb` and two `.trace.json.gz`, one per host.
- Total profile payload: **122.79 MiB**.
- Trace JSON train-step spans confirm ~1.387-1.389 s `jit_train_step(956769204075967375)` events on both hosts.

## HLO Dump

- `gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-baseline-v003/alekseyv-qwen3-maxtext-v003/xla_dump/`
- Host dirs: `gke-tpu-32cb1c36-gs53/`, `gke-tpu-32cb1c36-jj8f/`
- Optimized train-step HLO: `module_0103.jit_train_step.cl_759148519.after_optimizations.txt`
- Size: **1,507,976 bytes**, 8,408 lines
- SHA256: `3ea3f8e823438e2c7617af3e5a2881083f1a7a8ee30786e6f0fa3520d8b6d365`
- Memory report SHA256: `77e980e596ffe3654e01c3815864aef80c0a7a85de1e6ff8b8c01463256ffc0a`
- Text counts: all-gather 288, all-reduce 121, reduce-scatter 20,
  convolution 168, custom-call 156, copy 1,051, fusion 1,788,
  dot_general 394, splash 311, `_one_hot` 3, `log_softmax` 0.

Memory report headline:

```text
Total bytes used: 31215683040 (29.07GiB)
allocation 366: size 9.72GiB, preallocated-temp
allocation 369: size 7.88GiB, color 5, preallocated-temp
```

## Verdict

Supported baseline. Native upstream MaxText Qwen3-8B runs correctly at
seq=8192 on v6e-8, with xplane and HLO capture both validated. The batch-1
baseline reaches **5,896 tok/s/chip / 38.5% MFU** with peak HBM already at
29.07 GiB, leaving only about 2.18 GiB headroom. Batch scaling should therefore
be tested carefully; batch 2 may be a near-fit or OOM without additional memory
relief.

## Next Hypotheses

- **MaxText Qwen3 batch-2 fit probe**: keep the exact v003 stack and change only
  `per_device_batch_size=1` to `2`. Expected outcome is high-risk near-fit:
  support if it compiles and improves aggregate throughput; invalid if HBM OOMs.
- **MaxText Qwen3 memory relief before batch scaling**: if batch 2 OOMs, test
  removing one or more activation/offload choices only with a clear memory
  hypothesis, using the HLO memory report's `bf16[8192,151936]` and
  `bf16[36,1,8192,4096]` temps as the target.
