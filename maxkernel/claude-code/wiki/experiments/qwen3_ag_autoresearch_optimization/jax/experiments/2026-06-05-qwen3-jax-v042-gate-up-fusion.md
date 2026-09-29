---
variant: 8B/v6e-8
hypothesis: Fuse gate_proj and up_proj into a single gate_up_proj to reduce the number of XLA operations and memory copies
commit: 2026-06-04-v042-tp1-vocab-parallel-bs16-seq2048-tokamax-shardmap-gate-up-fusion
status: filed
verdict: supported
---

# v042: Fused gate_up_proj in Qwen3MLP

## Hypothesis under test
- **Mechanism**: Fusing `gate_proj` and `up_proj` linear layers into a single `gate_up_proj` in `Qwen3MLP`.
- **Predicted signal**: MXU utilization rises, step time drops due to fewer matmuls and better data locality.

## Results
- **Step time**: 5116.1ms
- **Throughput**: 51239 tok/s (vs 49944 tok/s in v040)
- **MFU**: 33.0% (vs 32.1% in v040)

## Verdict
**SUPPORTED**.
The throughput increased from 49,944 tok/s to 51,287 tok/s (avg), adding ~1.3k tok/s and reaching 33.0% MFU. This confirms that fusing `gate_proj` and `up_proj` improves performance. In order to avoid OOMs during device parameters initialization, `sharding.py` was updated to initialize directly on-device using `jax.jit`.
