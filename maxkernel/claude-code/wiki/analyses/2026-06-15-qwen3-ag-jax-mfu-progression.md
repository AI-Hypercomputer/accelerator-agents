---
title: "Qwen3-ag jax — per-experiment MFU/TPS data"
type: analysis
created: 2026-06-15
updated: 2026-06-16
tags: [qwen3-ag, jax, mfu, tps, data]
---

# Qwen3-ag jax — per-experiment MFU/TPS data

Best-effort per-experiment extraction for the **ag** lane, in experiment order. MFU is on the causal (MaxText-comparable) basis. Frontiers are clamped to the verified model-page best. MaxText MFU targets **causal-adjusted**: 2k = 36.6%, 8k = 39.8% (the tpu-recipes-v0.1.4 figures 38.0/45.3% are non-causal; the lanes count attention causally). TPS is convention-free. `—` = not recoverable (crash / no metric / unparsed). Regenerate: `python wiki/analyses/qwen3/build_explorer.py`.

**Frontier:** 2k = 33.0% MFU / 4,515 tok/s/chip · 8k = 30.6% MFU / 5,329 tok/s/chip.

| # | Experiment | seq | verdict | MFU | tok/s/chip |
|---|---|---|---|---|---|
| 1 | 2026-06-02-qwen3-jax-v001-batch-scaling | 2k | refuted | — | — |
| 2 | 2026-06-02-qwen3-jax-v002-selective-sc-offload | 2k | invalid | — | — |
| 3 | 2026-06-02-qwen3-jax-v003-selective-remat | 2k | refuted | — | — |
| 4 | 2026-06-02-qwen3-jax-v004-splash-bs2 | 2k | — | 25.3% | 4,515 |
| 5 | 2026-06-02-qwen3-jax-v005-splash-tuned | 2k | — | 22.5% | — |
| 6 | 2026-06-02-qwen3-jax-v006-tokamax-ce | 2k | — | — | — |
| 7 | 2026-06-02-qwen3-jax-v6e8-baseline-repro | 2k | supported | 20.0% | — |
| 8 | 2026-06-02-qwen3-jax-v6e8-baseline | 2k | baseline | 20.5% | — |
| 9 | 2026-06-02-qwen3-jax-v007-tp2-bs2 | 2k | — | — | — |
| 10 | 2026-06-02-qwen3-jax-v008-splash-tp2-bs4 | 2k | — | 28.6% | — |
| 11 | 2026-06-02-qwen3-jax-v009-splash-tp2-bs8 | 2k | — | 28.6% | — |
| 12 | 2026-06-02-qwen3-jax-v010-splash-tp2-tokamax-bs8 | ? | pending | — | — |
| 13 | 2026-06-02-qwen3-jax-v011-splash-tp2-remat-dots-bs8 | 2k | — | — | — |
| 14 | 2026-06-02-qwen3-jax-v012-splash-tp4-bs8 | 2k | — | 25.4% | — |
| 15 | 2026-06-03-qwen3-jax-v013-splash-tp2-tokamax-bs4 | 2k | — | 28.6% | — |
| 16 | 2026-06-03-qwen3-jax-v014-splash-tp2-tokamax-remat-bs8 | 2k | — | — | — |
| 17 | 2026-06-03-qwen3-jax-v015-splash-tp2-tokamax-remat-fused-rope-bs8 | 2k | — | — | — |
| 18 | 2026-06-03-qwen3-jax-v016-splash-tp1-tokamax-remat-fused-bs4 | 2k | — | 30.7% | — |
| 19 | 2026-06-03-qwen3-jax-v017-splash-tp2-tokamax-remat-scan-bs8 | 2k | — | 30.8% | — |
| 20 | 2026-06-03-qwen3-jax-v018-collective-matmul-off | ? | — | 29.9% | — |
| 21 | 2026-06-03-qwen3-jax-v019-phase2-bundle | ? | — | — | — |
| 22 | 2026-06-03-qwen3-jax-v020-phase2-bundle-bs4 | ? | — | 28.1% | — |
| 23 | 2026-06-03-qwen3-jax-v021-tp1-vocab-parallel-bs4 | 2k | — | — | — |
| 24 | 2026-06-03-qwen3-jax-v022-tp1-chunked-xla-bs4 | ? | — | — | — |
| 25 | 2026-06-03-qwen3-jax-v023-tp1-chunked-xla-bs4 | 2k | — | — | — |
| 26 | 2026-06-03-qwen3-jax-v024-tp1-xla-bs4 | 2k | — | — | — |
| 27 | 2026-06-03-qwen3-jax-v025-tp1-vocab-parallel-bs2 | 2k | — | 24.3% | — |
| 28 | 2026-06-03-qwen3-jax-v026-tp1-vocab-parallel-bs4-no-outer-ckpt | ? | — | 32.2% | — |
| 29 | 2026-06-03-qwen3-jax-v027-tp1-vocab-parallel-bs8-no-outer-ckpt | ? | — | 32.8% | — |
| 30 | 2026-06-03-qwen3-jax-v028-tp1-vocab-parallel-bs16-no-outer-ckpt | 2k | — | 32.8% | — |
| 31 | 2026-06-03-qwen3-jax-v029-tp1-vocab-parallel-bs8-fix-layout | ? | — | 32.6% | — |
| 32 | 2026-06-03-qwen3-jax-v030-tp1-vocab-parallel-bs8-chunked-xla | ? | — | 32.8% | — |
| 33 | 2026-06-03-qwen3-jax-v031-tp1-vocab-parallel-bs8-mosaic-tpu | ? | — | 32.8% | — |
| 34 | 2026-06-03-qwen3-jax-v032-tp1-vocab-parallel-bs8-mosaic-tpu-patched | ? | — | 32.8% | — |
| 35 | 2026-06-03-qwen3-jax-v033-tp1-vocab-parallel-bs8-collective-overlap | ? | — | 32.8% | — |
| 36 | 2026-06-04-qwen3-jax-v034-tp1-vocab-parallel-bs8-fused-qknorm-rope | 2k | — | 13.3% | — |
| 37 | 2026-06-04-qwen3-jax-v035-tp1-vocab-parallel-bs8-xla-sdpa | 2k | — | 32.8% | — |
| 38 | 2026-06-04-qwen3-jax-v036-tp1-vocab-parallel-bs8-chunked-xla | ? | — | 32.8% | — |
| 39 | 2026-06-04-qwen3-jax-v037-tp1-vocab-parallel-bs8-chunked-xla-splash-fix | 2k | — | 32.8% | — |
| 40 | 2026-06-04-qwen3-jax-v038-tp1-vocab-parallel-bs8-fused-rope | ? | — | 32.8% | — |
| 41 | 2026-06-04-qwen3-jax-v039-tp1-vocab-parallel-bs32-seq1024 | 2k | — | 33.0% | — |
| 42 | 2026-06-04-qwen3-jax-v040-tp1-vocab-parallel-bs16-seq2048-tokamax | 2k | — | — | — |
| 43 | 2026-06-04-qwen3-jax-v041-tp1-vocab-parallel-bs16-seq2048-tokamax-shardmap | 2k | — | 24.5% | — |
| 44 | 2026-06-04-qwen3-jax-v042-tp1-vocab-parallel-bs16-seq2048-tokamax-shardmap-scan | 2k | pending | — | — |
| 45 | 2026-06-05-qwen3-jax-v042-gate-up-fusion | 2k | supported | 33.0% | — |
| 46 | 2026-06-05-qwen3-jax-v043-bs32 | 2k | refuted | — | — |
| 47 | 2026-06-05-qwen3-jax-v044-bs24 | 2k | pending | — | — |
| 48 | 2026-06-05-qwen3-jax-v045-nsa-attention | 2k | pending | — | — |
| 49 | 2026-06-05-qwen3-jax-v045-nsa | 2k | refuted | — | — |
| 50 | 2026-06-05-qwen3-jax-v046-nsa-pallas | ? | completed | — | — |
| 51 | 2026-06-12-qwen3-jax-v047-seq8k-baseline | 8k | pending | — | — |
| 52 | 2026-06-13-qwen3-jax-v047-seq8k-baseline | 8k | completed | 30.6% | 5,329 |
| 53 | 2026-06-13-qwen3-jax-v048-bs64 | 8k | completed | — | — |
| 54 | 2026-06-13-qwen3-jax-v049-splash-bs64 | 8k | completed | — | — |
| 55 | 2026-06-13-qwen3-jax-v050-splash-tokamax-bs64 | ? | completed | — | — |
| 56 | 2026-06-13-qwen3-jax-v051-tokamax-bs64 | 8k | completed | — | — |
| 57 | 2026-06-13-qwen3-jax-v052-selective-remat-bs64 | 8k | refuted | — | — |
| 58 | 2026-06-13-qwen3-jax-v053-block-remat-bs64 | ? | completed | — | — |
| 59 | 2026-06-15-qwen3-jax-v053-block-remat-bs64-b | ? | completed | — | — |
| 60 | 2026-06-15-qwen3-jax-v053-block-remat-bs64-c | ? | completed | — | — |
| 61 | 2026-06-15-qwen3-jax-v053-block-remat-bs64-d | ? | invalid | — | — |
| 62 | 2026-06-15-qwen3-jax-v053-block-remat-bs64-e | ? | completed | — | — |
| 63 | 2026-06-15-qwen3-jax-v054-fused-qknorm-rope-c | ? | completed | — | — |
| 64 | 2026-06-15-qwen3-jax-v054-fused-qknorm-rope-d | ? | completed | — | — |
| 65 | 2026-06-15-qwen3-jax-v054-fused-qknorm-rope-e | 8k | completed | — | — |
| 66 | 2026-06-15-qwen3-jax-v054-fused-qknorm-rope-f | ? | completed | — | — |
| 67 | 2026-06-15-qwen3-jax-v055-ac-selective-remat | ? | completed | — | — |
| 68 | 2026-06-15-qwen3-jax-v056-splash-vmem-tune | 8k | pending | — | — |
| 69 | 2026-06-15-qwen3-jax-v057-d-tokamax-layer-ckpt | 8k | completed | — | — |
| 70 | 2026-06-15-qwen3-jax-v057-e-tokamax-layer-ckpt-fix-splash | 8k | completed | — | — |
| 71 | 2026-06-15-qwen3-jax-v057-f-tokamax-layer-ckpt-fix-splash | 2k | completed | — | — |
| 72 | 2026-06-15-qwen3-jax-v057-g-tokamax-layer-ckpt-fix-splash | 2k | refuted | — | — |
| 73 | 2026-06-15-qwen3-jax-v057-h-tokamax-block-remat | 2k | completed | — | — |
| 74 | 2026-06-15-qwen3-jax-v057-i-tokamax-block-remat-fsdp-fix | 8k | completed | — | — |
| 75 | 2026-06-15-qwen3-jax-v057-j-tokamax-layer-ckpt-bs4 | 2k | completed | — | — |
| 76 | 2026-06-15-qwen3-jax-v057-k-tokamax-policy | 8k | completed | — | — |
| 77 | 2026-06-15-qwen3-jax-v057-l-tokamax-layer-dots | 2k | completed | — | — |
| 78 | 2026-06-15-qwen3-jax-v057-m-tokamax-layer-dots-bs8 | 2k | completed | — | — |
| 79 | 2026-06-15-qwen3-jax-v057-n-tokamax-layer-dots-bs8 | 8k | completed | — | — |
| 80 | 2026-06-15-qwen3-jax-v057-o-tokamax-layer-dots-bs4 | ? | completed | — | — |
| 81 | 2026-06-15-qwen3-jax-v057-p-block-remat-bs8 | 8k | completed | — | — |
| 82 | 2026-06-15-qwen3-jax-v057-q-block-remat-bs8-no-policy | 8k | completed | — | — |
| 83 | 2026-06-15-qwen3-jax-v057-r-nested-ckpt-bs8 | 8k | completed | — | — |
| 84 | 2026-06-15-qwen3-jax-v057-s-nested-ckpt-nothing-saveable-bs8 | 8k | completed | — | — |
| 85 | 2026-06-15-qwen3-jax-v057-t-nested-ckpt-custom-outer-policy-bs8 | 8k | invalid | — | — |
| 86 | 2026-06-15-qwen3-jax-v057-tokamax-fix-b | 8k | refuted | — | — |
| 87 | 2026-06-15-qwen3-jax-v057-tokamax-fix-c | 8k | completed | — | — |
| 88 | 2026-06-15-qwen3-jax-v057-tokamax-fix | ? | pending | — | — |
| 89 | 2026-06-15-qwen3-jax-v057-u-nested-ckpt-bs4 | 8k | completed | — | — |

## See also

- [Interactive explorer](qwen3/mfu-explorer.html) · [progression plot](qwen3/mfu-progression.png)
- [Qwen3 ag — jax model page](../models/qwen3-ag-jax.md)
