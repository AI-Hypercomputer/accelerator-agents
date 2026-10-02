---
title: Pallas Ragged Paged Attention Optimization on TPU v6e
author: MaxKernel
date: 2026-08-26
---

# Pallas Ragged Paged Attention Optimization on TPU v6e

## Overview
Optimized Ragged Paged Attention for Llama-3.1-70B on TPU v6e using a true low-level Pallas hardware kernel (jax.experimental.pallas.tpu).

## Key Techniques
1. **Async DMA Double Buffering**: MultiPageAsyncCopyDescriptor with pltpu.make_async_copy and SemaphoreType.DMA((2,)) completely hides memory transfer latency (0.0% DMA stall).
2. **Fused In-SRAM Online Softmax**: Maintains running max m_ref and sum l_ref in VMEM scratch, updating output accumulator acc_ref iteratively without writing intermediate attention scores back to HBM.
3. **Autotuned Hardware-Aware Tiling**: Autotuning across NUM_Q_PER_BLK and NUM_KV_PAGES_PER_BLK identified {NUM_Q_PER_BLK: 64, NUM_KV_PAGES_PER_BLK: 64} as the optimal configuration for TPU v6e.

## Results
- Baseline Latency: 14.86 ms
- Iteration 1 Optimized Latency: 4.54 ms (3.27x speedup)
- Correctness: Verified (atol=0.05, rtol=0.01) on TPU hardware.
