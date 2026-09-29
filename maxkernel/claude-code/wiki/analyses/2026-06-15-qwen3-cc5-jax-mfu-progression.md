---
title: "Qwen3-cc5 jax — per-experiment MFU/TPS data"
type: analysis
created: 2026-06-15
updated: 2026-06-16
tags: [qwen3-cc5, jax, mfu, tps, data]
---

# Qwen3-cc5 jax — per-experiment MFU/TPS data

Best-effort per-experiment extraction for the **cc5** lane, in experiment order. MFU is on the causal (MaxText-comparable) basis. Frontiers are clamped to the verified model-page best. MaxText MFU targets **causal-adjusted**: 2k = 36.6%, 8k = 39.8% (the tpu-recipes-v0.1.4 figures 38.0/45.3% are non-causal; the lanes count attention causally). TPS is convention-free. `—` = not recoverable (crash / no metric / unparsed). Regenerate: `python wiki/analyses/qwen3/build_explorer.py`.

**Frontier:** 2k = 40.5% MFU / 8,196 tok/s/chip · 8k = 39.9% MFU / 6,959 tok/s/chip.

| # | Experiment | seq | verdict | MFU | tok/s/chip |
|---|---|---|---|---|---|
| 1 | 2026-06-12-v001-bs4 | 2k | refuted | 20.5% | — |
| 2 | 2026-06-12-v002-splash | 2k | supported | 22.3% | — |
| 3 | 2026-06-12-v003-tkmce | 2k | refuted | — | — |
| 4 | 2026-06-12-v004-bs2 | 2k | refuted | — | — |
| 5 | 2026-06-12-v005-bs2-splash | 2k | inconclusive | — | — |
| 6 | 2026-06-12-v006-tkmce-bf16 | 2k | invalid | 23.3% | — |
| 7 | 2026-06-02-qwen3-jax-v6e8-baseline | 2k | baseline | 20.5% | — |
| 8 | 2026-06-12-v007-bs2-vmem | 8k | supported | 31.4% | — |
| 9 | 2026-06-12-v008-chunked-xla-ce | 8k | invalid | — | — |
| 10 | 2026-06-12-v009-sc-offload | 8k | refuted | 31.4% | — |
| 11 | 2026-06-12-v010-host-offload-sched | 8k | refuted | — | — |
| 12 | 2026-06-12-v011-cxla-f32x | 8k | supported | — | — |
| 13 | 2026-06-12-v012-remat | 2k | invalid | 24.9% | 6,095 |
| 14 | 2026-06-12-v013-seq8k | 8k | refuted | — | — |
| 15 | 2026-06-12-v014-bs3-ce | 2k | refuted | — | — |
| 16 | 2026-06-12-v015-save-attn-remat | 8k | supported | 31.8% | — |
| 17 | 2026-06-12-v016-splash-bkv2048 | 2k | refuted | 31.8% | — |
| 18 | 2026-06-12-v017-partial-remat | 8k | supported | 33.8% | — |
| 19 | 2026-06-12-v018-prem4 | 8k | refuted | — | — |
| 20 | 2026-06-12-v019-val50 | 8k | supported | 33.8% | — |
| 21 | 2026-06-12-v020-scan | 8k | supported | 31.8% | — |
| 22 | 2026-06-12-v021-bs2-s8k-scan | 8k | refuted | 39.9% | — |
| 23 | 2026-06-12-v022-tokamax-splash | 2k | refuted | — | — |
| 24 | 2026-06-12-v023-scan-save-ffn | 8k | refuted | 33.6% | — |
| 25 | 2026-06-12-v024-offload-bs3 | 8k | refuted | — | 5,898 |
| 26 | 2026-06-12-v025-offload-scan-bs3 | 8k | refuted | — | 5,898 |
| 27 | 2026-06-12-v026-mt-flags-bs3 | 8k | supported | — | 5,933 |
| 28 | 2026-06-12-v027-bs4-mtfl | 8k | supported | 34.6% | 6,040 |
| 29 | 2026-06-12-v028-splash-blocks-bs4 | 2k | refuted | — | 6,100 |
| 30 | 2026-06-12-v029-bs5-probe | 8k | refuted | 34.6% | 6,040 |
| 31 | 2026-06-12-v030-ag-overlap | 8k | inconclusive | — | 6,161 |
| 32 | 2026-06-12-v031-ag-pipeline-flags | 8k | refuted | — | 6,161 |
| 33 | 2026-06-12-v032-loop-unroll | 8k | inconclusive | — | — |
| 34 | 2026-06-12-v033-tp2-probe | 8k | refuted | 34.6% | 6,100 |
| 35 | 2026-06-12-v034-scan-unroll | 8k | inconclusive | — | 6,161 |
| 36 | 2026-06-12-v035-unroll-noacf | 8k | refuted | — | — |
| 37 | 2026-06-12-v037-unroll-saveattn-bs2 | 8k | refuted | 35.8% | 6,240 |
| 38 | 2026-06-12-v038-bs2-saveattn-val | 8k | supported | 35.8% | 6,240 |
| 39 | 2026-06-12-v039-bs3-saveattn | 8k | refuted | — | 6,283 |
| 40 | 2026-06-12-v041-save-qkv-bs3 | 8k | refuted | 35.7% | 6,283 |
| 41 | 2026-06-12-v043-bs2-svqkv-val | 8k | supported | 36.0% | 6,271 |
| 42 | 2026-06-12-v044-context-checkpoint | 8k | supported | 38.6% | 6,723 |
| 43 | 2026-06-12-v045-ctxval | 8k | supported | — | 6,715 |
| 44 | 2026-06-12-v046-fused-mlp | 8k | refuted | — | 6,849 |
| 45 | 2026-06-12-v047-ctx-hbm | 8k | supported | 38.6% | 6,782 |
| 46 | 2026-06-12-v048-ctxh-val | 8k | supported | 38.9% | 6,780 |
| 47 | 2026-06-12-v049-act-shard | 8k | refuted | — | 6,840 |
| 48 | 2026-06-12-v050-weight-io | 8k | supported | 38.9% | 6,840 |
| 49 | 2026-06-12-v051-wioval | 8k | supported | 39.9% | 6,959 |
| 50 | 2026-06-12-v052-2kbase | 2k | supported | 31.6% | — |
| 51 | 2026-06-12-v053-2kflags | 2k | inconclusive | — | — |
| 52 | 2026-06-12-v054-2kscanflags | 2k | inconclusive | 31.6% | — |
| 53 | 2026-06-12-v055-2kstack | 2k | refuted | 31.6% | — |
| 54 | 2026-06-12-v056-2kio | 2k | refuted | — | — |
| 55 | 2026-06-12-v057-2kbs4 | 2k | supported | 38.5% | 8,196 |
| 56 | 2026-06-12-v058-2kbs5 | 2k | supported | 40.5% | 7,874 |
| 57 | 2026-06-12-v059-2kbs6 | 2k | refuted | — | 7,953 |
| 58 | 2026-06-12-v060-2kbs5val | 2k | supported | 40.5% | 7,874 |
| 59 | 2026-06-12-v061-2kce8 | 2k | invalid | 40.5% | 7,987 |
| 60 | 2026-06-12-v062-2kce5par | 2k | refuted | 40.5% | 7,871 |
| 61 | 2026-06-13-v063-2kunroll2 | 2k | refuted | 40.5% | 7,871 |

## See also

- [Interactive explorer](qwen3/mfu-explorer.html) · [progression plot](qwen3/mfu-progression.png)
- [Qwen3 cc5 — jax model page](../models/qwen3-cc5-jax.md)
