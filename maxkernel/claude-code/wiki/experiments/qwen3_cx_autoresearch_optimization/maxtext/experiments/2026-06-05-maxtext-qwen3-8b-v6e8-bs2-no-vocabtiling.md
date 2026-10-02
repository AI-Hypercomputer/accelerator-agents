---
title: "MaxText Qwen3-8B v6e-8 bs2 no-vocab-tiling probe (SUPPORTED, 392.3 TFLOP/s/device, 6,550 tok/s/device, 42.7% MFU)"
type: experiment
tags: [experiment, qwen3-cc, qwen3-8b, maxtext, v6e-8, batch-scaling]
hypothesis: maxtext-qwen3-8b-bs2-no-vocab-tiling
model: qwen3-cc-maxtext
created: 2026-06-05
updated: 2026-06-05
commit: runner-image-tsij-2026-06-02-14-00-13
status: supported
---

# MaxText Qwen3-8B v6e-8 bs2 no-vocab-tiling probe

## Hypothesis Under Test

The dense Trillium recipe suspected to produce ~45% MFU is the local MaxText
`llama3_1_8b_8192` recipe. The current Qwen3 runner image cannot enable the
newer `num_vocab_tiling` feature, so this probe tests the largest supported
single-change scaling attempt on the existing validated image: keep baseline
v003's recipe stack and raise only `per_device_batch_size` from 1 to 2.

Expected outcomes:

- Supported if Qwen3-8B seq8192 compiles within v6e-8 HBM and improves aggregate
  throughput.
- Invalid if the compile/runtime memory report exceeds the 31.25 GiB HBM budget.

## Setup

- **Cluster**: `<your-cluster>` (`<your-region>`, project
  `<your-project>`, v6e-8, 2 hosts x 4 chips)
- **Workload**: `alekseyv-qwen3-maxtext-v005-bs2`
- **Image**: `<your-registry>/alekseyv_google_com-runner:tsij-2026-06-02-14-00-13`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-recipe-v005-bs2`
- **Mesh**: `ici_fsdp_parallelism=-1` (v6e-8 -> fsdp=8)
- **Global tokens/step**: `2 * 8 * 8192 = 131,072`
- **Changed from v003**: `per_device_batch_size=2`

## Command

```bash
cd /app
LIBTPU_INIT_ARGS='<dense Trillium recipe flags plus disabled collective matmul>' \
PYTHONPATH=/app python3 -u -m MaxText.train MaxText/configs/base.yml \
  base_output_directory=gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-recipe-v005-bs2 \
  run_name=alekseyv-qwen3-maxtext-v005-bs2 \
  model_name=qwen3-8b \
  tokenizer_path=Qwen/Qwen3-8B \
  dataset_path=gs://max-datasets-rogue \
  dataset_type=synthetic \
  steps=20 \
  per_device_batch_size=2 \
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

Supported. `alekseyv-qwen3-maxtext-v005-bs2` completed all 20 steps, wrote
xplane profiles for both hosts, and uploaded HLO dumps for both hosts.

Steady-state window below uses steps 3-9 and 16-19, excluding step 0/1 warmup,
step 10 profile-start, step 14 profile-stop/write, and step 15 post-profile
recovery on worker 1.

| Metric | Worker 0 (`c180`) | Worker 1 (`9vh8`) |
|---|---:|---:|
| Median step time | **2.503 s** | **2.502 s** |
| Median TFLOP/s/device | **392.14** | **392.35** |
| Median tokens/s/device | **6,546.12** | **6,549.63** |
| Aggregate tokens/s | **52,368.98** | **52,397.01** |
| MFU | **42.72%** | **42.74%** |
| Final loss | **1.286** | **1.286** |

Per-step TensorBoard/log scalars from worker 1:

| step | s/step | TFLOP/s/device | tok/s/device | loss | note |
|---:|---:|---:|---:|---:|---|
| 0 | 9.855 | 99.59 | 1,662.45 | 12.435 | warmup |
| 1 | 3.488 | 281.35 | 4,696.64 | 12.435 | warmup |
| 2 | 2.589 | 379.03 | 6,327.21 | 11.721 | warmup tail |
| 3 | 2.500 | 392.66 | 6,554.73 | 10.658 | steady |
| 4 | 2.500 | 392.54 | 6,552.86 | 9.511 | steady |
| 5 | 2.501 | 392.51 | 6,552.22 | 8.358 | steady |
| 6 | 2.501 | 392.41 | 6,550.67 | 7.327 | steady |
| 7 | 2.501 | 392.38 | 6,550.09 | 6.370 | steady |
| 8 | 2.502 | 392.35 | 6,549.63 | 5.509 | steady |
| 9 | 2.502 | 392.23 | 6,547.65 | 4.742 | steady |
| 10 | 2.610 | 376.07 | 6,277.86 | 4.060 | profile start |
| 11 | 2.505 | 391.80 | 6,540.49 | 3.466 | profiled |
| 12 | 2.505 | 391.74 | 6,539.52 | 2.963 | profiled |
| 13 | 2.506 | 391.65 | 6,537.88 | 2.543 | profiled |
| 14 | 10.141 | 96.78 | 1,615.64 | 2.196 | profile stop/write |
| 15 | 2.709 | 362.26 | 6,047.36 | 1.918 | post-profile |
| 16 | 2.504 | 391.99 | 6,543.58 | 1.697 | steady |
| 17 | 2.505 | 391.80 | 6,540.47 | 1.526 | steady |
| 18 | 2.505 | 391.78 | 6,540.04 | 1.392 | steady |
| 19 | 2.505 | 391.74 | 6,539.37 | 1.286 | steady |

Memory lines from logs and HLO:

```text
Total memory size: 22.2 GB, Output size: 11.4 GB, Temp size: 10.8 GB,
Argument size: 11.4 GB, Host temp size: 15.8 GB.
Total bytes used: 40620000732 (37.83GiB)
allocation 369: size 15.75GiB, color 5, preallocated-temp
allocation 366: size 10.60GiB, preallocated-temp
```

The run succeeds despite the 37.83 GiB aggregate memory-report total because
the compiled program uses host/offload-colored allocations; the runtime log is
the better signal for device fit.

## Profile

- `gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-recipe-v005-bs2/alekseyv-qwen3-maxtext-v005-bs2/tensorboard/plugins/profile/2026_06_05_02_30_49/`
- Files: two `.xplane.pb` and two `.trace.json.gz`, one per host.
- Total profile payload: **122.09 MiB**.

## HLO Dump

- `gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-recipe-v005-bs2/alekseyv-qwen3-maxtext-v005-bs2/xla_dump/`
- Host dirs: `gke-tpu-964065d9-9vh8/`, `gke-tpu-964065d9-c180/`
- Optimized train-step HLO: `module_0103.jit_train_step.cl_759148519.after_optimizations.txt`
- Size: **1,497,087 bytes**, 8,063 lines
- SHA256: `23c6d886d655e8315f775b089d0969851fae3a5db810d43e4fa39759780f4a0f`
- Memory report SHA256: `ffd1bf3d9ec62c4c981ea193b5bb1575c8a51e23031be0e8f8091e940cb30419`
- Text counts: all-gather 284, all-reduce 121, reduce-scatter 20,
  convolution 179, custom-call 131, copy 1,226, fusion 1,771,
  dot_general 387, splash 191, `_one_hot` 8, `log_softmax` 0.

## Verdict

Supported and a MaxText-lane improvement over v003. Batch 2 raises MaxText from
**38.5% MFU / 47,172 tok/s aggregate** to **42.7% MFU / 52,397 tok/s
aggregate** without requiring vocab tiling. This still does not reproduce the
suspected ~45% recipe result, but it closes most of the gap using the legacy
runner image.

## Next Hypotheses

- **MaxText bs3 no-vocab-tiling fit probe**: the Llama3.1 no-collective-matmul
  recipe uses `per_device_batch_size=3`; Qwen3 bs2 succeeded with host/offload
  memory. Try bs3 before investing in a refreshed MaxText image.
- **Refresh MaxText runner for vocab tiling**: local MaxText has
  `num_vocab_tiling`, but the current image does not. A full dependency-refresh
  image is needed to test bs3/bs4 with vocab tiling.
