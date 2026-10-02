---
variant: 8B/v6e-8
hypothesis: MaxText baseline performance for Qwen3-8B on TPU v6e-8
commit: N/A
status: running
verdict: open
---

# v002: MaxText baseline (bs=2)

## Hypothesis under test
- **Mechanism**: Test MaxText's native implementation at `per_device_batch_size=2` (`qwen3-8b.yml` configuration).
- **Predicted signal**: Because maxtext implements efficient memory management and pipeline/tensor parallelism options (or good default sharding), it should be able to fit `bs=2`.

## Expected outcome
Higher MFU than `bs=1` (22.4%), potentially beating Torchax's 33.0%.

## Results
- **Step Time**: ~3.70s
- **MFU**: 14.8% (29.1 TFLOP/s/device)
- **Throughput**: 4,428 tokens/s/device (35,424 tok/s cluster)

## Analysis
- **Conclusion**: FAILED to improve MFU. Increasing the batch size from 1 to 2 caused performance to degrade significantly (step time went from 1.39s to 3.70s).
- **Hypothesis Evaluation**: The hypothesis that larger batch size would improve arithmetic intensity was incorrect. The severe degradation (lower throughput and lower MFU compared to `bs=1`) strongly suggests that `bs=2` exceeds the efficient memory footprint, likely causing aggressive activation recomputation (rematerialization) or HBM spilling.
- **Next Steps**: `bs=1` is the optimal baseline for MaxText on this topology. The baseline MaxText MFU (22.4%) is lower than the optimized JAX lane baseline (33.0%). We will pause MaxText and continue optimization in the JAX lane with Pallas kernels.

## Verdict
**OPEN**. Running to collect baseline.
