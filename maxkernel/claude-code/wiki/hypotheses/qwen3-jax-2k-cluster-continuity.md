---
title: "qwen3-jax: seq-2048 baseline continuity on charles-v6e"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "0% (continuity check — validates plumbing on the new cluster)"
confidence: high
effort: S
origin: human
tags: [qwen3-cc5, jax, v6e-8, seq2048, charles-v6e, baseline]
created: 2026-06-12
updated: 2026-06-12
---

# seq-2048 baseline continuity on charles-v6e

Phase-opening bring-up for the user-directed seq-2048 optimization phase on
the **charles-v6e** cluster (project `<your-project-2>`, <your-zone>, 3× v6e-8
on-demand; confirmed v6e-8 = tpu-v6e-slice 2x4).

## Statement

The historical seq-2048 best recipe
([v007](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v007-bs2-vmem.md):
bs2 + splash + `--xla_tpu_scoped_vmem_limit_kib=81920`, nothing else)
reproduces **667 ms / 48,758 TPS / 31.4% MFU within ±2%** on charles-v6e
using the current trunk image (`latest` = v050-wio2, trunk `b991d9f`).

## Rationale

Same hardware generation (v6e-8), same image, recipe is launch-flag-only —
any deviation beyond noise indicates either a cluster-environment difference
(libtpu, drivers, on-demand vs spot host variance) or a trunk regression at
the 2k op-point (trunk has gained io-layout/scan/ctx code paths since v007,
all default-off). Also validates the new-cluster plumbing end-to-end:
cross-project image pull (artifactregistry.reader confirmed), cross-region
GCS profile write (objectAdmin granted 2026-06-12), xpk dispatch with
dedicated kubeconfig.

## Proposed experiment

v052: v007 recipe verbatim on charles-v6e, 20 steps, profile 12–14, HLO
dump. PASS = step within 667 ms ± 2% and loss series matching v007's noise
band. FAIL → debug environment before any 2k optimization work.

## Risks

None semantic (recipe identical to v007). Cross-region GCS (~10× slower
writes) may slow profile upload, not the measured steps.

## Dependencies

charles-v6e quota: 2 of 3 slices held by foreign zombie Kueue reservations
(charles-rl-mh-2026-01-03-*, QuotaReserved since January) — exactly 1 slice
free; serial dispatch only.

## Outcome (2026-06-12)

**Supported** —
[v052](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v052-2kbase.md):
667.6 ms / 49,090 TPS / ≈31.6% MFU — +0.1% step vs v007, well inside ±2%.
charles-v6e is measurement-equivalent; baseline anchored.

## See also

- [Model page](../models/qwen3-cc5-jax.md) · [v007](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v007-bs2-vmem.md) · [v052](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v052-2kbase.md)
