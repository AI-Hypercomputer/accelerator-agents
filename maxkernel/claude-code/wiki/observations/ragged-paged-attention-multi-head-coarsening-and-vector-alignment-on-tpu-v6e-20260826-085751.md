---
title: Ragged Paged Attention Multi-Head Coarsening and Vector Alignment on TPU v6e
author: MaxKernel
date: 2026-08-26
---

# Ragged Paged Attention Multi-Head Coarsening and Vector Alignment on TPU v6e

In Iteration 2 of 7p_Ragged_Paged_Attention for Llama-3.1-70B on TPU v6e, we discovered two critical performance principles:
1. **Multi-Head Head-Dimension Coarsening**: Scaling NUM_COMBINED_KV_HEADS_PER_BLK from 2 to 16 fuses all 8 KV heads (and their 64 associated Q heads) into a single grid tile along the head dimension, reducing grid program launches by 4x and slashing host dispatch / barrier latency.
2. **128-Element Minor Vector Register Alignment**: Ensuring the softmax running maximum (m_ref) and normalizer (l_ref) scratchpads in VMEM have their minor dimension padded to 128 elements allows the TPU compiler to use full SIMD vector load/store instructions rather than scalar unaligned access, cutting execution time significantly.
Combining these optimizations reduced kernel latency from 14.86 ms (baseline) and 4.54 ms (Iter 1) down to 3.0116 ms (4.94x speedup vs baseline) while maintaining 100% numerical correctness.
