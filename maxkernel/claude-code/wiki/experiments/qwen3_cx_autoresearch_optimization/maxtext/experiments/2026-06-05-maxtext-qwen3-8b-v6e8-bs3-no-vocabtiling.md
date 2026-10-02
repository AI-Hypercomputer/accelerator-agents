---
title: "MaxText Qwen3-8B v6e-8 bs3 no-vocab-tiling probe (SUPPORTED, 412.3 TFLOP/s/device, 6,883 tok/s/device, 44.9% MFU)"
type: experiment
tags: [experiment, qwen3-cc, qwen3-8b, maxtext, v6e-8, batch-scaling]
hypothesis: maxtext-qwen3-8b-bs3-no-vocab-tiling
model: qwen3-cc-maxtext
created: 2026-06-05
updated: 2026-06-05
commit: runner-image-tsij-2026-06-02-14-00-13
status: supported
---

# MaxText Qwen3-8B v6e-8 bs3 no-vocab-tiling probe

## Hypothesis Under Test

The closest local recipe to the known Llama3.1 MaxText v6e-8 reference is
`llama3_1_8b_8192_no_collective_matmul`, which uses
`per_device_batch_size=3` plus the dense Trillium flag stack. Qwen3 bs2 fits
and reaches **42.7% MFU** on the same legacy runner image. This probe tests
whether Qwen3 can reach the Llama-like batch density without the newer
vocab-tiling path.

Expected outcomes:

- Supported if Qwen3-8B seq8192 compiles within the v6e-8 memory/offload budget
  and improves aggregate throughput/MFU over bs2.
- Invalid if compile/runtime memory exceeds the available device/host-offload
  budget.

## Setup

- **Cluster**: `<your-cluster>` (`<your-region>`, project
  `<your-project>`, v6e-8, 2 hosts x 4 chips)
- **Workload**: `alekseyv-qwen3-maxtext-v006-bs3`
- **Image**: `<your-registry>/alekseyv_google_com-runner:tsij-2026-06-02-14-00-13`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-recipe-v006-bs3`
- **Mesh**: `ici_fsdp_parallelism=-1` (v6e-8 -> fsdp=8)
- **Global tokens/step**: `3 * 8 * 8192 = 196,608`
- **Changed from v005**: `per_device_batch_size=3`

## Command

```bash
cd /app
LIBTPU_INIT_ARGS='<dense Trillium recipe flags plus disabled collective matmul>' \
PYTHONPATH=/app python3 -u -m MaxText.train MaxText/configs/base.yml \
  base_output_directory=gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-recipe-v006-bs3 \
  run_name=alekseyv-qwen3-maxtext-v006-bs3 \
  model_name=qwen3-8b \
  tokenizer_path=Qwen/Qwen3-8B \
  dataset_path=gs://max-datasets-rogue \
  dataset_type=synthetic \
  steps=20 \
  per_device_batch_size=3 \
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

## Results

Supported. `alekseyv-qwen3-maxtext-v006-bs3` completed all 20 steps, wrote
xplane profiles for both hosts, and uploaded HLO dumps for both hosts.

Steady-state window below uses steps 3-9 and 16-19, excluding step 0/1 warmup,
step 10 profile-start, step 14 profile-stop/write, and step 15 post-profile
recovery.

| Metric | Worker 0 (`c180`) | Worker 1 (`9vh8`) |
|---|---:|---:|
| Median step time | **3.572 s** | **3.571 s** |
| Median TFLOP/s/device | **412.14** | **412.32** |
| Median tokens/s/device | **6,879.95** | **6,883.00** |
| Aggregate tokens/s | **55,039.60** | **55,063.97** |
| MFU | **44.90%** | **44.92%** |
| Final loss | **2.287** | **2.287** |

Per-step TensorBoard/log scalars from worker 1:

| step | s/step | TFLOP/s/device | tok/s/device | loss | note |
|---:|---:|---:|---:|---:|---|
| 0 | 13.275 | 110.90 | 1,851.32 | 12.434 | warmup |
| 1 | 4.553 | 323.38 | 5,398.28 | 12.434 | warmup |
| 2 | 3.632 | 405.38 | 6,767.09 | 11.804 | warmup tail |
| 3 | 3.563 | 413.21 | 6,897.91 | 10.865 | steady |
| 4 | 3.564 | 413.04 | 6,894.97 | 9.859 | steady |
| 5 | 3.566 | 412.84 | 6,891.62 | 8.851 | steady |
| 6 | 3.568 | 412.67 | 6,888.81 | 7.931 | steady |
| 7 | 3.569 | 412.52 | 6,886.25 | 7.094 | steady |
| 8 | 3.571 | 412.32 | 6,883.00 | 6.326 | steady |
| 9 | 3.571 | 412.21 | 6,881.16 | 5.634 | steady |
| 10 | 3.678 | 400.24 | 6,681.40 | 5.015 | profile start |
| 11 | 3.576 | 411.65 | 6,871.86 | 4.470 | profiled |
| 12 | 3.578 | 411.49 | 6,869.11 | 3.994 | profiled |
| 13 | 3.579 | 411.40 | 6,867.60 | 3.587 | profiled |
| 14 | 12.480 | 117.96 | 1,969.17 | 3.245 | profile stop/write |
| 15 | 3.570 | 412.40 | 6,884.30 | 2.961 | post-profile |
| 16 | 3.573 | 411.99 | 6,877.48 | 2.732 | steady |
| 17 | 3.575 | 411.80 | 6,874.23 | 2.549 | steady |
| 18 | 3.576 | 411.69 | 6,872.40 | 2.404 | steady |
| 19 | 3.577 | 411.61 | 6,871.17 | 2.287 | steady |

Memory lines from logs and HLO:

```text
Total memory size: 24.5 GB, Output size: 11.4 GB, Temp size: 13.1 GB,
Argument size: 11.4 GB, Host temp size: 23.6 GB.
Total bytes used: 51749388768 (48.20GiB)
allocation 369: size 23.62GiB, color 5, preallocated-temp
allocation 366: size 13.09GiB, preallocated-temp
```

The run succeeds because the large report total includes host/offload-colored
allocations; runtime device fit is reflected by the `24.5 GB` memory line and
successful execution.

## Profile

- `gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-recipe-v006-bs3/alekseyv-qwen3-maxtext-v006-bs3/tensorboard/plugins/profile/2026_06_05_02_39_21/`
- Files: two `.xplane.pb` and two `.trace.json.gz`, one per host.
- Total profile payload: **126.16 MiB**.

## HLO Dump

- `gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-recipe-v006-bs3/alekseyv-qwen3-maxtext-v006-bs3/xla_dump/`
- Host dirs: `gke-tpu-964065d9-9vh8/`, `gke-tpu-964065d9-c180/`
- Optimized train-step HLO: `module_0103.jit_train_step.cl_759148519.after_optimizations.txt`
- Size: **1,498,186 bytes**, 8,013 lines
- SHA256: `598fe616e4031e66c7a0709c0e0dfd04bec2566041590378c0a4fb1ea931cf84`
- Memory report SHA256: `b3e40bd2b6744b9628e5c03748667619f02f84e16ed02c6bfc548ed6722f4b3b`
- Text counts: all-gather 284, all-reduce 121, reduce-scatter 20,
  convolution 177, custom-call 128, copy 1,500, fusion 1,768,
  dot_general 387, splash 191, `_one_hot` 8, `log_softmax` 0.

## Verdict

Supported and effectively reproduces the suspected **~45% MFU** MaxText recipe
level for Qwen3-8B on v6e-8, using the legacy-compatible Llama3.1
`no_collective_matmul` batch-density shape (`per_device_batch_size=3`) rather
than the newer vocab-tiling adaptation.

## Next Hypotheses

- **bs4 fit probe**: local `llama3_1_8b_8192` uses batch 4 without the
  no-collective-matmul suffix. Qwen3 bs3 already needs 24.5 GB device memory and
  23.6 GB host temp, so bs4 is a high-risk fit probe but may test whether the
  original recipe shape is viable.
- **Refresh MaxText runner for vocab tiling**: local MaxText supports
  `num_vocab_tiling`, but the current runner image does not. This is still the
  cleaner path for Qwen3's larger vocabulary if bs4 fails.
