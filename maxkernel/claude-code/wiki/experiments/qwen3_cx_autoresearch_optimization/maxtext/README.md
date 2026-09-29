# maxtext code - Qwen3 8B autoresearch reference lane

Reference-baseline runs of **Qwen3-8B via [MaxText](../../../codebases/maxtext.md)**
on TPU. This lane anchors the hand-tuned `jax/` and `torchax/` lanes against the
upstream AI-Hypercomputer JAX stack.

Unlike the Gemma 4 MaxText lane, this is not an architectural approximation:
MaxText ships native Qwen3 dense support, including `decoder_block: "qwen3"`,
QK norm, untied logits, and `qwen3-8b.yml`.

## Status

| Run | tok/s/chip | MFU | Date | Page |
|-----|-----------:|----:|------|------|
| MaxText `qwen3-8b`, bs=3 seq=8192 fsdp=8 v6e-8 | **6,883** | **44.9%** | 2026-06-05 | [bs3 probe](experiments/2026-06-05-maxtext-qwen3-8b-v6e8-bs3-no-vocabtiling.md) |
| MaxText `qwen3-8b`, bs=2 seq=8192 fsdp=8 v6e-8 | **6,550** | **42.7%** | 2026-06-05 | [bs2 probe](experiments/2026-06-05-maxtext-qwen3-8b-v6e8-bs2-no-vocabtiling.md) |
| MaxText `qwen3-8b`, bs=1 seq=8192 fsdp=8 v6e-8 | **5,896** | **38.5%** | 2026-06-05 | [baseline](experiments/2026-06-05-maxtext-qwen3-8b-v6e8-baseline.md) |

## Convention

- No source code in this folder. Runs use upstream MaxText source from the
  runner image and/or the external MaxText worktree under `/mnt/disks/persist/`.
- What lives here: `experiments/` pages, the lane log, and run notes needed to
  reproduce a baseline.
- Each experiment page records the MaxText commit/image, the exact command,
  tuning parameters, XProf path, and HLO path.

## Implementation Found

- Runner image path: `/app/MaxText/layers/qwen3.py`
- Model config: `/app/MaxText/configs/models/qwen3-8b.yml`
- Local source mirror: `/mnt/disks/persist/maxtext-main/src/maxtext/models/qwen3.py`
- Local model config: `/mnt/disks/persist/maxtext-main/src/maxtext/configs/models/qwen3-8b.yml`

Qwen3-8B config values match the program invariants: 36 layers, hidden 4096,
32 query heads, 8 KV heads, head dim 128, MLP 12288, vocab 151936, RoPE theta
1000000, QK norm enabled, untied logits.

## Recipe Inspection

The likely source of the suspected ~45% MFU MaxText recipe is
`/mnt/disks/persist/maxtext-main/benchmarks/maxtext_trillium_model_configs.py`
`llama3_1_8b_8192`, not a Qwen3-specific recipe. It uses seq8192,
`per_device_batch_size=4`, custom remat/offload, 2048 Splash tiles, scoped VMEM
98304, reduce-scatter layout, data-parallel overlap, async all-gather
continuation fusion, SparseCore all-reduce offload, and host-offload scheduler
flags.

The current runner image predates MaxText `num_vocab_tiling`, so the direct
Qwen3 vocab-tiling adaptation is invalid on that image. The supported bs3 probe
uses the legacy-compatible Llama3.1 `no_collective_matmul` batch-density shape
and reaches **44.9% MFU**, effectively matching the suspected ~45% recipe level.

## See Also

- [Qwen3 8B - jax](../../../models/qwen3-cc-jax.md)
- [Qwen3 8B - torchax](../../../models/qwen3-cc-torchax.md)
- [MaxText codebase](../../../codebases/maxtext.md)
