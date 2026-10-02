---
title: "Qwen3 jax — XLA scheduler flag ablation (minimal load-bearing set)"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "0 (attribution); de-risks torchax transfer"
confidence: high
effort: S
origin: 2026-06-02-qwen3_cc-jax-retrospective-2
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 jax — XLA scheduler flag ablation

v018's +3.4 pp came from a bundle of **7** LIBTPU scheduler flags. This ablation identifies which are
load-bearing — primarily to **de-risk the torchax-lane transfer** (if torchax's libtpu rejects a flag,
we need to know whether it was a critical one) and as a reusable prior for the model-optimization-index.

## Statement

Running the v018 frontier with ONLY `xla_tpu_enable_all_experimental_scheduler_features=true` (the
umbrella flag) + the required `scoped_vmem` flag — dropping the other 6 — recovers ≥ most of the +3.4 pp
(MFU ≈ 35.8% within ~1 pp). If so, the umbrella flag is the minimal load-bearing set; if MFU falls back
toward 32.4% (v008 baseline), the other 6 flags carry material weight and need individual ablation.

## Rationale

- `xla_tpu_enable_all_experimental_scheduler_features` is an umbrella that may subsume the others
  (`scheduler_memory_pressure_tracking`, `latency_hiding_scheduler_rerun`, etc.).
- Not frontier-moving (the full bundle is already the frontier), but high-confidence informative and
  cheap (flag-only). Directly serves the high-value torchax pivot.
- `scoped_vmem=98304` is NOT dropped — splash bs4 OOMs without it (v006).

## Proposed experiment

v024: v018 frontier launch with `LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=98304
--xla_tpu_enable_all_experimental_scheduler_features=true'` (only). Compare MFU vs 35.8% (full bundle)
and 32.4% (no scheduler flags). One run; if the umbrella alone ≈ full bundle → minimal set found.

## Risks

- None material (flag-only, frontier-config base). Not expected to set a new frontier (≤ full bundle).

## Dependencies

None — flag-only on the splash image.
