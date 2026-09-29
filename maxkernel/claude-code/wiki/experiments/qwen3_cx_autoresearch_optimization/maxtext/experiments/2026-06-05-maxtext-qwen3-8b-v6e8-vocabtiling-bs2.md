---
title: "MaxText Qwen3-8B v6e-8 vocab-tiling bs2 probe (INVALID: image lacks vocab tiling)"
type: experiment
tags: [experiment, qwen3-cc, qwen3-8b, maxtext, v6e-8, vocab-tiling, batch-scaling]
hypothesis: maxtext-qwen3-8b-vocab-tiling-bs2
model: qwen3-cc-maxtext
created: 2026-06-05
updated: 2026-06-05
commit: runner-image-tsij-2026-06-02-14-00-13
status: invalid
---

# MaxText Qwen3-8B v6e-8 vocab-tiling bs2 probe

## Hypothesis Under Test

The local MaxText Trillium recipe likely responsible for the observed ~45% MFU
claim is `llama3_1_8b_8192`, which runs seq=8192 with
`per_device_batch_size=4`, custom remat/offload, Splash attention tiles, and the
dense v6e XLA flag stack. Qwen3-8B baseline v003 used the same recipe family but
only fit `per_device_batch_size=1`, reaching **38.5% MFU** with peak HBM already
at **29.07 GiB / 31.25 GiB**.

Qwen3 has a larger vocab than Llama3.1-8B (`151936`), so the missing adaptation
is likely MaxText's vocab-tiling cross-entropy path. This probe enables
`num_vocab_tiling=16` and raises batch to 2.

## Setup

- **Cluster**: `<your-cluster>` (`<your-region>`, project
  `<your-project>`, v6e-8, 2 hosts x 4 chips)
- **Workload**: `alekseyv-qwen3-maxtext-v004-vt-bs2`
- **Image**: `<your-registry>/alekseyv_google_com-runner:tsij-2026-06-02-14-00-13`
- **MaxText layout in image**: `/app/MaxText`
- **Model config**: `/app/MaxText/configs/models/qwen3-8b.yml`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-recipe-v004-vocabtiling-bs2`
- **Mesh**: `ici_fsdp_parallelism=-1` (v6e-8 -> fsdp=8)
- **Global tokens/step**: `2 * 8 * 8192 = 131,072`
- **Changed from v003**: `per_device_batch_size=2`, `num_vocab_tiling=16`

## Recipe Source

- `/mnt/disks/persist/maxtext-main/benchmarks/maxtext_trillium_model_configs.py`
  `llama3_1_8b_8192`: dense Trillium seq8192 recipe with batch 4.
- `/mnt/disks/persist/maxtext-main/src/maxtext/configs/base.yml`:
  `num_vocab_tiling` chunks logits along the batch-sequence axis to reduce peak
  cross-entropy memory.
- `/mnt/disks/persist/maxtext-main/src/maxtext/configs/types.py`: validates
  `(per_device_batch_size * max_target_length) % num_vocab_tiling == 0` and
  rejects vocab tiling only for `enable_nnx=True`. This MaxText lane uses the
  default Linen path.

## Command

```bash
cd /app
LIBTPU_INIT_ARGS='<dense Trillium recipe flags plus disabled collective matmul>' \
PYTHONPATH=/app python3 -u -m MaxText.train MaxText/configs/base.yml \
  base_output_directory=gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-05-qwen3-8b-v6e8-recipe-v004-vocabtiling-bs2 \
  run_name=alekseyv-qwen3-maxtext-v004-vt-bs2 \
  model_name=qwen3-8b \
  tokenizer_path=Qwen/Qwen3-8B \
  dataset_path=gs://max-datasets-rogue \
  dataset_type=synthetic \
  steps=20 \
  per_device_batch_size=2 \
  num_vocab_tiling=16 \
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

Invalid. The workload failed during MaxText config parsing on both hosts before
compilation:

```text
ValueError: Key num_vocab_tiling was passed at the command line but isn't in config.
```

Follow-up diagnostic workload `alekseyv-qwen3-maxtext-diag-vocabtiling` grepped
the runner image and found Qwen3 support in `/app/MaxText/layers/qwen3.py`, but
no `num_vocab_tiling` or `vocab_til` implementation under `/app/MaxText`.

The local MaxText mirror does include vocab tiling, but it is a newer
lowercase-package layout and does not import cleanly against this runner image's
dependency stack:

```text
ModuleNotFoundError: No module named
'orbax.checkpoint.experimental.emergency.multi_tier_checkpointing'
```

## Verdict

Invalid as a runtime config for the current runner image. Testing this
hypothesis requires either a refreshed MaxText runner image with the newer
vocab-tiling implementation and dependencies, or a focused backport into the
legacy `/app/MaxText` tree.
