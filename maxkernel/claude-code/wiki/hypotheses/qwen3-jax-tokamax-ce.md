---
title: "Qwen3 jax — tokamax streamed cross-entropy"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "HBM unlock at seq 8192 (~2.5 GiB/chip logits saved)"
confidence: medium
effort: M
origin: 2026-06-02-qwen3-jax-v6e8-baseline
hlo_prefilter: pending
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 jax — tokamax streamed cross-entropy

*Hypothesis*: Replacing the plain softmax CE (which materializes `[B,L,V=151936]`
logits at the lm_head) with `tokamax.linear_softmax_cross_entropy_loss`
(mosaic_tpu / chunked_xla, streamed logsumexp over V via shard_map) frees
~2.5 GiB/chip at seq 8192, removing the lm_head memory wall.

*Mechanism*: Toggle `skip_lm_head` so forward returns hidden; compute loss from
`hidden` + `lm_head_weight()` inside the jit via the shard_map CE wrapper (mirror
the llama3 jax lane's `_ce_tokamax`). fp32 boundary cast required for chunked_xla.

*Falsification criterion*: no HBM reduction vs plain CE (XLA already fuses the
logit+softmax favorably — check HLO pre-filter first) → refuted `xla-already-fuses`.

## Outcome — REFUTED (2026-06-02)

CE is numerically correct ([v013](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v013-ce-bs2.md)) and *does* relieve logit-memory pressure (enabled bs2 at seq8192), but it never beat the splash-only frontier on MFU:
- [v014](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v014-splash-ce-bs6.md) seq2048 bs6: 30.5% < 32.4% (v008).
- [v016](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v016-s8k-ce-bs2.md) seq8192 bs2: 29.5% < 30.4% (v009).

**Verdict**: memory enabler, not a throughput lever, at either seq. Retired for this lane — revisit only if a future config is hard logit-memory-bound with no cheaper relief.
