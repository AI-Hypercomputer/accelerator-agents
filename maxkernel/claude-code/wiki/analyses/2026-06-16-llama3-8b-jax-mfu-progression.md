---
title: "Llama3-8B jax — per-experiment MFU/TPS data"
type: analysis
created: 2026-06-16
updated: 2026-06-16
tags: [llama3-8b, jax, mfu, tps, data]
---

# Llama3-8B jax — per-experiment MFU/TPS data

Best-effort per-experiment extraction for the **jax** lane, in experiment order. MFU as reported (MaxText formula); the lane and MaxText use the same FLOP accounting. Frontiers (8k) clamped to the verified best. MaxText reference: **39.6% MFU (causal-adjusted; the tpu-recipes-v0.1.4 figure 44.6% is non-causal — current MaxText applies the causal ÷2)** / 7,069 tok/s/chip @ seq8192 (no ref for 1k/2k/4k). `—` = not recoverable (crash / no metric / unparsed). Regenerate: `python wiki/analyses/llama3/build_explorer.py`.

**Frontier:** 1k = 41.7% MFU / — tok/s/chip · 2k = 43.5% MFU / — tok/s/chip · 4k = 34.3% MFU / — tok/s/chip · 8k = 43.6% MFU / 7,768 tok/s/chip.

| # | Experiment | seq | verdict | MFU | tok/s/chip |
|---|---|---|---|---|---|
| 1 | 2026-04-26-jax-exp1e-baseline-port-direct-accepted | 8k | supported | 36.6% | — |
| 2 | 2026-04-26-jax-exp3-profile-bs3-baseline-accepted | 8k | supported | — | — |
| 3 | 2026-04-26-jax-exp4-vmem-default-localtmp-accepted | 8k | supported | — | — |
| 4 | 2026-04-26-jax-exp5-bs2-density-accepted | 8k | supported | 35.4% | — |
| 5 | 2026-04-26-jax-exp6-bs4-density-accepted | 8k | supported | — | — |
| 6 | 2026-04-26-jax-exp7-bs3-noscan-rejected | 8k | refuted | — | — |
| 7 | 2026-04-26-jax-exp8-bs3-remat-dotsnobatch-rejected | 8k | refuted | — | — |
| 8 | 2026-04-26-jax-exp9-bs4-frontier-pre-maxtext-xla-potential | 8k | inconclusive | — | — |
| 9 | 2026-04-26-jax-exp10-noop-default-stack-accepted | 8k | supported | — | — |
| 10 | 2026-04-26-jax-exp11-scan-remat-everything-saveable-rejected | 8k | refuted | — | — |
| 11 | 2026-04-26-jax-exp12-maxtext-xla-stack-bs3-accepted | 8k | supported | 40.7% | — |
| 12 | 2026-04-26-jax-exp12b-maxtext-xla-stack-bs4-accepted | 8k | supported | 41.5% | — |
| 13 | 2026-04-26-jax-exp13-maxtext-xla-stack-bs5-accepted | 8k | supported | 41.6% | 7,471 |
| 14 | 2026-04-26-jax-exp14-bs6-density-rejected | 8k | refuted | 39.6% | — |
| 15 | 2026-04-26-jax-exp15-exp13-profile-bs5-accepted | 8k | supported | — | — |
| 16 | 2026-04-26-jax-exp16-bs7-attempt-rejected | 8k | refuted | — | — |
| 17 | 2026-04-26-jax-exp17-bs8-attempt-rejected | 8k | refuted | — | — |
| 18 | 2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier | 8k | supported | 43.6% | 7,768 |
| 19 | 2026-04-27-jax-exp18-bkv2048-match-maxtext-bs5-accepted | 2k | supported | 41.9% | — |
| 20 | 2026-04-27-jax-exp18b-bkv2048-bs3-accepted | 2k | supported | 41.3% | — |
| 21 | 2026-04-27-jax-exp19-bkv2048-bs4-accepted | 2k | supported | 41.4% | — |
| 22 | 2026-04-27-jax-exp19b-bkv2048-bs6-rejected | 2k | refuted | — | — |
| 23 | 2026-04-27-jax-exp20-jax-experimental-splash-accepted | 8k | supported | 40.0% | — |
| 24 | 2026-04-27-jax-exp23b-bq1k-bkv1k-bs5-rejected | 1k | refuted | — | — |
| 25 | 2026-04-27-jax-exp26-frontier-profile-bs5-maxtext-accepted | 8k | supported | — | — |
| 26 | 2026-04-27-jax-exp29-vmem-131072-bs4-rejected | 8k | refuted | 42.3% | — |
| 27 | 2026-04-27-jax-exp30-bkv-2048-bs4-fullsc-rejected | 2k | refuted | 43.5% | — |
| 28 | 2026-04-27-jax-exp31-bs3-fullsc-maxtext-shape-accepted | 8k | supported | 42.4% | — |
| 29 | 2026-04-27-jax-exp32-splash-bq-4096-bs4-rejected | 4k | refuted | 34.3% | — |
| 30 | 2026-04-27-jax-exp35-save-qkv-proj-bs4-rejected | 8k | refuted | — | — |
| 31 | 2026-04-27-jax-exp36-qkv-proj-offloaded-bs4-rejected | 8k | refuted | 42.8% | — |
| 32 | 2026-04-27-jax-exp37-qkv-proj-offloaded-bs6-rejected | 8k | refuted | — | — |
| 33 | 2026-04-27-jax-exp38-qkv-proj-offloaded-bs5-rejected | 8k | refuted | 42.8% | — |
| 34 | 2026-04-27-jax-exp39-save-out-proj-bs4-rejected | 8k | refuted | — | — |
| 35 | 2026-04-27-jax-exp40-save-out-proj-bs3-potential | 8k | potential | 42.9% | — |
| 36 | 2026-04-27-jax-exp41-scan-unroll-2-bs4-rejected | 8k | refuted | 42.4% | — |
| 37 | 2026-04-27-jax-exp42-no-bundle-aware-cost-model-rejected | 8k | refuted | 42.4% | — |
| 38 | 2026-04-27-jax-exp43-enhanced-launch-barrier-rejected | 8k | refuted | 42.7% | — |
| 39 | 2026-04-27-jax-exp44-async-collective-permute-potential | 8k | inconclusive | 43.1% | — |
| 40 | 2026-04-27-jax-exp45-no-megacore-fusion-allow-ags-potential | 8k | inconclusive | 43.2% | — |
| 41 | 2026-04-27-jax-exp46-combo-acpermute-nomegafuseag-rejected | 8k | refuted | 42.7% | — |
| 42 | 2026-04-27-jax-exp47-vmem-65536-bs4-rejected | 8k | refuted | 41.4% | — |
| 43 | 2026-04-27-jax-exp48-vmem-81920-mxt-moe-level-rejected | 8k | refuted | 42.5% | — |
| 44 | 2026-04-27-jax-exp49-splash-bkv-512-bs4-rejected | 8k | refuted | 42.2% | — |
| 45 | 2026-04-27-jax-exp50-validate-exp28b-rerun-accepted | 8k | supported | 43.1% | — |
| 46 | 2026-04-27-jax-exp51-enable-collective-matmul-rejected | 8k | refuted | 36.8% | — |
| 47 | 2026-04-27-jax-exp52-splash-bkv-dkv-1024-rejected | 1k | refuted | 41.7% | — |
| 48 | 2026-04-27-jax-exp53-tokamax-ce-mosaic-tpu-rejected | 8k | refuted | 41.3% | — |
| 49 | 2026-04-27-jax-exp54-precast-bf16-weights-bs4-rejected | 8k | refuted | 42.7% | — |
| 50 | 2026-04-27-jax-exp55-precast-bf16-weights-bs5-rejected | 8k | refuted | 43.0% | — |
| 51 | 2026-04-27-jax-exp56-validate-exp28b-rerun-2-accepted | 8k | supported | 43.2% | — |
| 52 | 2026-04-27-jax-exp57-no-overlap-compute-collective-rejected | 8k | refuted | 42.7% | — |
| 53 | 2026-04-27-jax-exp58-no-aggressive-opt-barrier-potential | 8k | inconclusive | 43.1% | — |
| 54 | 2026-04-27-jax-exp59-lat-hiding-rerun-0-potential | 8k | inconclusive | 43.2% | — |
| 55 | 2026-04-27-jax-exp60-loop-inv-chain-disabled-potential | 8k | inconclusive | 42.9% | — |
| 56 | 2026-04-27-jax-exp61-val100opt-realdata-exhausted-potential | 8k | inconclusive | — | — |
| 57 | 2026-04-27-jax-exp62-val100base-realdata-exhausted-potential | 8k | inconclusive | — | — |
| 58 | 2026-04-27-jax-exp64-val100ref-realdata-exhausted-potential | 8k | inconclusive | — | — |
| 59 | 2026-04-27-jax-exp65-67-loss-validation-100steps | 8k | supported | — | — |
| 60 | 2026-04-27-jax-exp68-lr3e5-syn-bs4-accepted | 8k | supported | — | — |
| 61 | 2026-04-27-jax-exp69-lr3e5-bs3-syn-accepted | 8k | supported | 42.5% | — |
| 62 | README | 8k | — | 43.6% | 7,700 |

## See also

- [Interactive explorer](llama3/mfu-explorer.html)
- [Llama3-8B program](../experiments/llama3_8B_autoresearch_optimization/README.md)
