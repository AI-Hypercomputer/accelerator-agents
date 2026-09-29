---
title: 3p_MLA_Attention Iteration 2
author: MaxKernel
date: 2026-08-26
---

# 3p_MLA_Attention Iteration 2

Tested BQ=1024, BK=512 yielding 12.339 ms. BQ=512, BK=256 remains faster (12.243 ms) on S=2048 due to 4-stage causal tile pipelining.
