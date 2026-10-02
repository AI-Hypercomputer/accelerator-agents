---
title: 2p_GQA_Attention Iteration 2
author: MaxKernel
date: 2026-08-26
---

# 2p_GQA_Attention Iteration 2

3D dynamic grid with fori_loop achieves 13.648 ms on TPU v6e, confirming that 2D static unrolled grid (7.769 ms) is 1.76x faster due to zero dynamic loop/slice overhead in Mosaic.
