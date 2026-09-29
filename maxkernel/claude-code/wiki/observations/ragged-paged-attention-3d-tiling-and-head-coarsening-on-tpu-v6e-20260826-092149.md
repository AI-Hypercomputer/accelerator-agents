---
title: Ragged Paged Attention 3D Tiling and Head Coarsening on TPU v6e
author: MaxKernel
date: 2026-08-26
---

# Ragged Paged Attention 3D Tiling and Head Coarsening on TPU v6e

In Ragged Paged Attention for Llama-3.1-70B on TPU v6e (Trillium), coarsening the head tiling dimension (NUM_COMBINED_KV_HEADS_PER_BLK=16) fuses all 8 KV heads into a single dispatch tile, amortizing grid barrier synchronization overhead. Combining with NUM_Q_PER_BLK=32 and NUM_KV_PAGES_PER_BLK=64 with double-buffered asynchronous DMA achieves 3.0089 ms latency (4.94x speedup over 14.86 ms baseline), with 92.2% compute ratio and DMA wait under 7.8%.
