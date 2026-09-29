---
title: "Qwen3 8B - MaxText"
type: model
architecture: qwen3-cx
lane: maxtext
sizes: ["8B"]
hardwares: ["v6e-8"]
tags: [qwen3-cx, maxtext]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 8B - MaxText

Qwen3 8B (`Qwen/Qwen3-8B`) running under **MaxText** on TPU v6e-8. This lane is
the upstream reference stack for the Qwen3 autoresearch program.

## Target Metrics

- Primary: tokens/sec/chip and MFU at seq_len=8192.
- Secondary: step time, aggregate tokens/sec, peak HBM, profile/HLO shape.

## Variant Matrix

| Size | Hardware | Status | Baseline | Current best | Open hyps | Frontier exp |
|------|----------|--------|----------|--------------|-----------|--------------|
| 8B | v6e-8 | live | 1.389 s / 47,172 TPS / 38.5% MFU @ seq8192 bs8 | 3.571 s / 55,064 TPS / 44.9% MFU @ seq8192 bs24 | 2 | [2026-06-05 bs3 probe](../experiments/qwen3_cx_autoresearch_optimization/maxtext/experiments/2026-06-05-maxtext-qwen3-8b-v6e8-bs3-no-vocabtiling.md) |

## Open Hypotheses

1. Batch 4 fit probe: local MaxText's Llama3.1 primary Trillium recipe uses
   `per_device_batch_size=4`; Qwen3 batch 3 fits and reaches 44.9% MFU on the
   legacy runner image.
2. Vocab-tiling image refresh: local MaxText supports `num_vocab_tiling`, but
   the current runner image does not. A refreshed dependency image is needed
   before testing Qwen3 bs3/bs4 with vocab tiling.

## See Also

- [Lane README](../experiments/qwen3_cx_autoresearch_optimization/maxtext/README.md)
- [Qwen3 8B - jax](qwen3-cx-jax.md)
- [Qwen3 8B - torchax](qwen3-cx-torchax.md)
