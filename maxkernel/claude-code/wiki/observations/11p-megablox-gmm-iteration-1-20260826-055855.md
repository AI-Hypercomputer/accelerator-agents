---
title: 11p_Megablox_GMM Iteration 1
author: MaxKernel
date: 2026-08-26
---

# 11p_Megablox_GMM Iteration 1

Vectorized 128-expert GMM into parallel batched matmul (jnp.einsum('gmk,gkn->gmn')), achieving 1.487 ms (2.20x speedup vs 3.27 ms baseline, Bit-exact match).
