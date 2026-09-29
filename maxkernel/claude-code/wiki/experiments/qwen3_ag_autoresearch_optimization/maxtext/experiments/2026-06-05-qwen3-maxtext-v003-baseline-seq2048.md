---
model: qwen3
lane: maxtext
slug: baseline-seq2048
verdict: invalid
date: 2026-06-05
---

# v003: MaxText baseline seq=2048 bs=16

## Hypothesis
If we run MaxText with `per_device_batch_size=16` and `max_target_length=2048`, we can compare its scaling against the torchtitan JAX lane (which achieved 33.0% MFU at the same scale).

## Setup
`per_device_batch_size=16`, `max_target_length=2048`, `model_name=qwen3-8b`.

## Results
**FAILED (OOM)**. MaxText exhausted memory during compilation.
`Used 41.76G of 31.25G hbm. Exceeded hbm capacity by 10.51G.`

This shows that the torchtitan JAX lane is actually more memory-efficient than the default MaxText config for this sequence length and batch size.

## Next Steps
MaxText baselines are complete. MaxText struggles with memory overhead compared to torchtitan JAX at this scale. The torchtitan JAX lane remains our primary candidate.
