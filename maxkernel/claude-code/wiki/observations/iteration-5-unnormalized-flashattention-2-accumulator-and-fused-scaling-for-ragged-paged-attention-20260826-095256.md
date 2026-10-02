---
title: Iteration 5: Unnormalized FlashAttention-2 Accumulator and Fused Scaling for Ragged Paged Attention
author: MaxKernel
date: 2026-08-26
---

# Iteration 5: Unnormalized FlashAttention-2 Accumulator and Fused Scaling for Ragged Paged Attention

In Iteration 5 of optimizing 7p_Ragged_Paged_Attention on TPU v6e (Trillium), we implemented an unnormalized FlashAttention-2 online softmax accumulator pipeline, hoisted Q pre-scaling (fusing scaling directly into systolic GEMM), and hoisted causal/KV mask generation across all 8 KV heads. This eliminated vector division operations from the inner loop and removed >8.3 million redundant scalar multiplications per Q block. The kernel achieved 3.0318 ms latency (vs 14.86 ms baseline, 4.90x speedup), verified 100% numerical correctness within atol=0.05, rtol=0.01, and maintained 92.46% compute efficiency with only 7.54% DMA synchronization overhead.
