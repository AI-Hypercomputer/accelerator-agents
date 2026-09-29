---
title: "Qwen3-jax: MaxText custom_vjp CE at seq2048"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "REFUTED — maxtext-CE regresses −6.9% at seq2048 (v037)"
confidence: low
effort: S
origin: 2026-06-02-v035-maxtext-ce-s8k-bs3
hlo_prefilter: n/a
tags: [qwen3-cc, jax, cross-entropy, maxtext, custom-vjp, seq2048]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-jax: MaxText custom_vjp CE at seq2048

## Statement

Applying the proven `--use_maxtext_ce` (T5X custom_vjp CE) to the seq2048 frontier (v018, 35.8% / 6,964)
is **neutral-to-slightly-positive** — it does not regress. Unlike seq8192 (where it cracked the batch
wall, v034), at seq2048 the CE-backward transient is a smaller share of the step so the gain is expected
to be small.

## Rationale

maxtext-CE was the seq8192 batch-wall lever ([v034](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v034-maxtext-ce-s8k-bs2.md)).
At seq2048 the logits are 4× smaller, so the CE-backward memory transient is proportionally less of the
step — the wall it cracked at seq8192 may not even exist at seq2048. Worth a cheap confirmation run so
maxtext-CE can become the lane's default CE everywhere (it's bit-identical math, [v033](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v033-maxtext-ce-s8k-bs1.md)).

## Proposed experiment

- maxtext-CE @ seq2048 bs4 (v018 shape) vs v018 (35.8% / 6,964). Support: ≥ v018 beyond noise. Neutral
  is the expected (and acceptable) outcome — confirms no regression, makes maxtext-CE the universal default.

## See also

- [Qwen3 8B — jax](../models/qwen3-cc-jax.md) · [maxtext-CE hypothesis](qwen3-jax-maxtext-ce.md)
