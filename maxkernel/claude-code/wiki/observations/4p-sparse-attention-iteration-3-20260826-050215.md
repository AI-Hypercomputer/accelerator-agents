---
title: 4p_Sparse_Attention Iteration 3
author: MaxKernel
date: 2026-08-26
---

# 4p_Sparse_Attention Iteration 3

Pre-scaling queries outside the kernel incurs extra memory bandwidth overhead (3.756 ms vs 3.583 ms). Scaling in registers inside the Pallas block update is optimal.
