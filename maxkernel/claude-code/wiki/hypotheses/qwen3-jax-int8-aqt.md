---
title: "Qwen3 jax — int8 weight-only quantization (AQT/qwix)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: open
expected_gain: "~+2 pp MFU (halves weight HBM traffic into the 48.6% conv bucket + halves weight memory) — speculative pending parity validation"
confidence: low-medium
effort: L
origin: 2026-06-12-qwen3_cc5-jax-retrospective-2
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — int8 weight-only (AQT/qwix)

*Stub — retrospective-2 direction #3; the remaining L item after the
prefetch branch closed.*

*Hypothesis*: int8 weight-only quantization (AQT or qwix; bf16 compute,
int8 storage + dequant-on-load) cuts the conv-fusion bucket's HBM
bandwidth demand enough for ≥ +2% tok/s/chip at the v038 frontier shape,
with loss parity within the SCHEMA rule-8 bar (else `invalid`).

*Rationale*: taxonomy precision section ("AQT weight-only int8 is the
recommended starting point"; halves bandwidth → halves the critical batch);
[v038 profile](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v038-bs2-saveattn-val.md):
conv fusion 48.6% with MXU at 55.5% — partially bandwidth-bound; weight
memory savings (~8 GB) also reopen the bs3/bs4 rungs on the save_attn
substrate.

*Falsification criterion*: loss diverges beyond parity (→ invalid), or
tok/s/chip ≤ frontier +1% → refuted.

## See also

- [Retrospective #2 (origin)](../analyses/2026-06-12-qwen3_cc5-jax-retrospective-2.md)
- [v038 (the op-point it targets)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v038-bs2-saveattn-val.md)
