---
title: Ragged Paged Attention Batched 3D GEMM and Vectorized KV Unpacking on TPU v6e
author: MaxKernel
date: 2026-08-26
---

# Ragged Paged Attention Batched 3D GEMM and Vectorized KV Unpacking on TPU v6e

In Iteration 4 of Ragged Paged Attention for Llama-3.1-70B on TPU v6e (Trillium), we developed a fully vectorized multi-head batched Pallas TPU kernel. By structuring the inner flash-attention loop as batched 3D tensor contractions (8, 256, 128) x (8, 1024, 128)^T across all 8 KV heads simultaneously and utilizing SIMD uint32 bitcast intrinsics to unpack K/V pages, the kernel eliminates 8x Python loop serialization and reduces masked VMEM stores from 24 down to 3 per KV block. Achieving 3.04 ms latency (4.88x speedup over 14.86 ms baseline) with 92.4% compute duty cycle and <7.6% DMA sync wait.
