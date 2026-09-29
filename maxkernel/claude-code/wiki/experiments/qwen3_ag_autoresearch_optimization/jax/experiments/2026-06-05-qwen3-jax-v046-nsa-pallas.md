---
model: qwen3
lane: jax
date: 2026-06-05
version: v046
hypothesis: Pallas NSA fix for `ConcretizationTypeError`
status: complete
---

# v046

The `math.sqrt` fix worked, but the underlying Pallas kernel implementation is not fully ported to TPU yet (only CPU).
