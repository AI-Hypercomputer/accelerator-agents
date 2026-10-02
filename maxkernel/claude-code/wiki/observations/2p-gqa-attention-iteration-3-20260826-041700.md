---
title: 2p_GQA_Attention Iteration 3
author: MaxKernel
date: 2026-08-26
---

# 2p_GQA_Attention Iteration 3

Achieved 6.900 ms (4.79x speedup) on TPU v6e by combining static 2D grid, BQ=512, BK=256 diagonal tiling, pre-scaled query inputs, and 16MB internal scratch.
