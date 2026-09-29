---
variant: 8B/v6e-8
hypothesis: MaxText baseline performance for Qwen3-8B on TPU v6e-8
commit: N/A
status: running
verdict: open
---

# v001: MaxText baseline

## Hypothesis under test
- **Mechanism**: Use MaxText's native implementation of Qwen3-8B (`qwen3-8b.yml` configuration).
- **Predicted signal**: This provides a pure JAX/XLA baseline performance ceiling to compare against the torchax implementation. MaxText typically achieves high MFU on TPUs.

## Expected outcome
High MFU on TPU v6e-8. We expect `per_device_batch_size=1` or `2` at `max_target_length=8192` to work out of the box, with high tokens/sec/chip.

## Results
- **Step time**: 1.39s (bs=1)
- **Throughput**: 47,114 tok/s (cluster), 5,889 tok/s/chip
- **MFU**: 22.4% (Calculated: 353 cluster TFLOP/s / 1576 peak TFLOP/s)

## Verdict
**CLOSED**. MaxText baseline at `bs=1` achieves 1.39s step time and 22.4% MFU. This is almost exactly 2x faster than torchax at `bs=1` (2.62s). However, torchax scaled to `bs=2` for 2.55s (33.0% MFU). We will now test MaxText at `bs=2`.
