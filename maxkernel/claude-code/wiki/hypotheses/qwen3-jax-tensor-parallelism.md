---
title: "Qwen3 jax — TP=2 probe at the frontier shape (the never-probed axis)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "±0–1 pp (coverage probe: restructures the FSDP AG pattern that holds 624 ms of exposed wait)"
confidence: low
effort: S
origin: 2026-06-12-qwen3_cc5-jax-retrospective-2
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — tensor parallelism probe (fsdp4 × tp2)

*Hypothesis*: `--tp_parallelism=2` (mesh fsdp4×tp2) at the frontier shape
changes the collective structure enough — TP shards the matmul weight dims,
shrinking/replacing the per-layer FSDP weight all-gathers whose exposed
wait is 624 ms/step — to beat 6,100 tok/s/chip (frontier +1%).

*Rationale*: [retrospective #2](../analyses/2026-06-12-qwen3_cc5-jax-retrospective-2.md)
direction #2 — TP has ZERO experiments in all 33 on this lane. Negative
cross-model prior: llama3-8b-jax measured TP=2 at −14% (different model,
pre-flag-stack era). v6e ICI 80 GB/s; taxonomy says TP ≤ 8 intra-island is
viable. This is a cheap axis-coverage probe, not a high-conviction bet.

*Falsification criterion*: crash/OOM, or ≤ 6,100 tok/s/chip → refuted; TP
axis closed on this lane with a measured number (replacing the borrowed
llama3 prior).

## See also

- [Retrospective #2 (origin)](../analyses/2026-06-12-qwen3_cc5-jax-retrospective-2.md)
- [Bucket diff (the AG bucket this restructures)](../analyses/2026-06-12-maxtext-vs-jax-bucket-diff.md)
