---
title: 2p_GQA_Attention Iteration 4
author: MaxKernel
date: 2026-08-26
---

# 2p_GQA_Attention Iteration 4

Tested BQ=1024, BK=1024 achieving 7.097 ms (4.67x speedup). BQ=512, BK=256 remains faster (6.900 ms, 4.79x speedup) due to tighter causal mask granularity.
