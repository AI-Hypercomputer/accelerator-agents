---
title: 3p_MLA_Attention Iteration 3
author: MaxKernel
date: 2026-08-26
---

# 3p_MLA_Attention Iteration 3

Tested BK=128 which resulted in 17.176 ms due to doubling the inner loop count. BK=256 remains the optimal block size for DeepSeek-V3 MLA FlashAttention.
