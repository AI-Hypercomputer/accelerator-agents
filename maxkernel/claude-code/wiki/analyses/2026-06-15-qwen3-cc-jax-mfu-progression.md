---
title: "Qwen3-cc jax — per-experiment MFU/TPS data"
type: analysis
created: 2026-06-15
updated: 2026-06-16
tags: [qwen3-cc, jax, mfu, tps, data]
---

# Qwen3-cc jax — per-experiment MFU/TPS data

Best-effort per-experiment extraction for the **cc** lane, in experiment order. MFU is on the causal (MaxText-comparable) basis. Frontiers are clamped to the verified model-page best. MaxText MFU targets **causal-adjusted**: 2k = 36.6%, 8k = 39.8% (the tpu-recipes-v0.1.4 figures 38.0/45.3% are non-causal; the lanes count attention causally). TPS is convention-free. `—` = not recoverable (crash / no metric / unparsed). Regenerate: `python wiki/analyses/qwen3/build_explorer.py`.

**Frontier:** 2k = 35.8% MFU / 6,964 tok/s/chip · 8k = 34.6% MFU / 5,992 tok/s/chip.

| # | Experiment | seq | verdict | MFU | tok/s/chip |
|---|---|---|---|---|---|
| 1 | 2026-06-02-v001-batch-bs2 | 2k | invalid | 20.5% | — |
| 2 | 2026-06-02-v002-batch-bs4 | 2k | invalid | — | — |
| 3 | 2026-06-02-v003-sc-offload | 2k | refuted | 16.5% | — |
| 4 | 2026-06-02-v004-remat-bs2 | 2k | supported | 20.5% | 4,280 |
| 5 | 2026-06-02-v005-remat-bs3 | 2k | supported | 25.1% | 4,874 |
| 6 | 2026-06-02-v006-splash-bs4 | 2k | invalid | — | — |
| 7 | 2026-06-02-qwen3-jax-v6e8-baseline | 2k | baseline | 20.5% | — |
| 8 | 2026-06-02-v007-splash-s8k | 8k | invalid | 25.1% | — |
| 9 | 2026-06-02-v008-splash-vmem-bs4 | 2k | supported | 25.1% | 6,299 |
| 10 | 2026-06-02-v009-splash-s8k-vmem | 8k | supported | 30.4% | 5,305 |
| 11 | 2026-06-02-v010-ce-bs2 | 2k | invalid | 22.0% | — |
| 12 | 2026-06-02-v011-ce-mtpu-bs2 | ? | invalid | — | — |
| 13 | 2026-06-02-v012-splash-ce-bs6 | ? | invalid | — | — |
| 14 | 2026-06-02-v013-ce-bs2 | 2k | supported | 22.0% | — |
| 15 | 2026-06-02-v014-splash-ce-bs6 | 8k | refuted | 30.5% | — |
| 16 | 2026-06-02-v016-s8k-ce-bs2 | 8k | refuted | 30.4% | — |
| 17 | 2026-06-02-v017-sc-bs4 | 2k | refuted | 32.4% | — |
| 18 | 2026-06-02-v018-xla-flag-stack | 2k | supported | 32.4% | 6,964 |
| 19 | 2026-06-02-v019-xla-flags-s8k | 8k | inconclusive | 30.4% | — |
| 20 | 2026-06-02-v020-tokamax-splash-knobs | 2k | refuted | 35.8% | — |
| 21 | 2026-06-02-v021-splash-bq1024 | 2k | refuted | 35.8% | — |
| 22 | 2026-06-02-v022-splash-bq4096-s8k | 8k | inconclusive | 30.4% | — |
| 23 | 2026-06-02-v023-async-collective-fusion | 2k | refuted | 35.8% | — |
| 24 | 2026-06-02-v024-xla-flag-ablation | 2k | refuted | 35.8% | 6,203 |
| 25 | 2026-06-02-v025-offload-remat-s8k-bs2 | 8k | invalid | 30.4% | — |
| 26 | 2026-06-02-v026-offload-ce-s8k-bs3 | 8k | invalid | 30.4% | — |
| 27 | 2026-06-02-v027-named-offload-ce-s8k-bs3 | 8k | invalid | 30.4% | — |
| 28 | 2026-06-02-v028-scan-overlap-s8k-bs1 | 8k | supported | 32.3% | 5,632 |
| 29 | 2026-06-02-v029-full-stack-s8k-bs3 | 8k | invalid | — | — |
| 30 | 2026-06-02-v030-scan-offload-ce-s8k-bs3 | 8k | inconclusive | — | — |
| 31 | 2026-06-02-v031-scan-ce-bs2-nooffload-s8k | 8k | refuted | — | 5,632 |
| 32 | 2026-06-02-v032-scan-overlap-s2k-bs4 | 2k | refuted | — | 6,964 |
| 33 | 2026-06-02-v033-maxtext-ce-s8k-bs1 | 8k | supported | — | 5,632 |
| 34 | 2026-06-02-v034-maxtext-ce-s8k-bs2 | 8k | supported | — | 5,992 |
| 35 | 2026-06-02-v035-maxtext-ce-s8k-bs3 | 8k | supported | 34.6% | — |
| 36 | 2026-06-02-v036-maxtext-ce-offload-s8k-bs3 | 8k | refuted | — | — |
| 37 | 2026-06-02-v037-maxtext-ce-s2k-bs4 | 2k | refuted | — | — |
| 38 | 2026-06-02-v038-maxtext-flags-s8k-bs3 | 8k | supported | — | — |
| 39 | 2026-06-02-v039-maxtext-offload-recipe-s8k-bs3 | 8k | refuted | — | — |
| 40 | 2026-06-02-v040-splash-bkv2048-s8k-bs3 | 8k | refuted | — | — |
| 41 | 2026-06-03-v041-shard-acts-s8k-bs3 | 8k | refuted | — | — |
| 42 | 2026-06-05-v042-rmsnorm-bf16 | 8k | inconclusive | — | — |
| 43 | 2026-06-05-v043-tp2-s8k | 8k | inconclusive | — | — |
| 44 | 2026-06-05-v043b-tp2-bs6 | 8k | refuted | — | — |
| 45 | 2026-06-05-v045-save-norm-stats | 8k | supported | — | — |

## See also

- [Interactive explorer](qwen3/mfu-explorer.html) · [progression plot](qwen3/mfu-progression.png)
- [Qwen3 cc — jax model page](../models/qwen3-cc-jax.md)
