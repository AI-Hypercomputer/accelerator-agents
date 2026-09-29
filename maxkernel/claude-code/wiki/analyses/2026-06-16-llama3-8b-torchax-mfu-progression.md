---
title: "Llama3-8B torchax — per-experiment MFU/TPS data"
type: analysis
created: 2026-06-16
updated: 2026-06-16
tags: [llama3-8b, torchax, mfu, tps, data]
---

# Llama3-8B torchax — per-experiment MFU/TPS data

Best-effort per-experiment extraction for the **torchax** lane, in experiment order. MFU as reported (MaxText formula); the lane and MaxText use the same FLOP accounting. Frontiers (8k) clamped to the verified best. MaxText reference: **39.6% MFU (causal-adjusted; the tpu-recipes-v0.1.4 figure 44.6% is non-causal — current MaxText applies the causal ÷2)** / 7,069 tok/s/chip @ seq8192 (no ref for 1k/2k/4k). `—` = not recoverable (crash / no metric / unparsed). Regenerate: `python wiki/analyses/llama3/build_explorer.py`.

**Frontier:** 1k = 35.7% MFU / 5,450 tok/s/chip · 2k = 35.2% MFU / — tok/s/chip · 4k = 33.1% MFU / — tok/s/chip · 8k = 36.8% MFU / 6,559 tok/s/chip.

| # | Experiment | seq | verdict | MFU | tok/s/chip |
|---|---|---|---|---|---|
| 1 | 2026-04-25-baseline | 1k | supported | — | 5,450 |
| 2 | 2026-04-25-exp2-splash-bs2-potential | 1k | potential | 22.9% | — |
| 3 | 2026-04-25-exp3-splash-bs4-accepted | 1k | supported | 35.7% | — |
| 4 | 2026-04-25-exp7-splash-xla-bs4-rejected | 1k | refuted | 35.7% | — |
| 5 | 2026-04-25-exp8-splash-kernel-autotune-potential | 1k | potential | 35.7% | — |
| 6 | 2026-04-25-exp9-splash-autotuned-bs4-accepted | 1k | supported | — | — |
| 7 | 2026-04-25-exp13-per-layer-remat-accepted | 2k | supported | 35.2% | — |
| 8 | 2026-04-25-exp17-layer-remat-bs1-seq8k-accepted | 8k | supported | 32.2% | — |
| 9 | 2026-04-25-exp20-amp-fp32-master-seq8k-accepted | 8k | supported | 31.6% | — |
| 10 | 2026-04-25-exp23-amp-bs2-seq4k-accepted | 4k | supported | 33.1% | — |
| 11 | 2026-04-26-exp55b-fp32-master-tokamax-ce-bs3-seq8k-accepted | 8k | supported | 34.8% | 6,206 |
| 12 | 2026-04-26-exp62b-chunkedxla-ce-bs3-seq8k-accepted | 8k | supported | 35.4% | 6,313 |
| 13 | 2026-04-26-exp72a-tokamax-splash-bs3-seq8k-accepted | 8k | supported | 35.8% | 6,559 |
| 14 | README | 8k | — | 36.8% | 6,559 |

## See also

- [Interactive explorer](llama3/mfu-explorer.html)
- [Llama3-8B program](../experiments/llama3_8B_autoresearch_optimization/README.md)
