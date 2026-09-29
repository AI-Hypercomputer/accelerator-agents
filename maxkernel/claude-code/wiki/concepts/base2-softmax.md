---
title: "Base-2 Softmax"
type: concept
tags: [stub, kernel, attention, optimization]
created: 2026-04-22
updated: 2026-04-22
sources: 1
---

Rewrite `exp(x) = 2^(x·log2 e)` to map softmax onto TPU's native base-2 exp unit; opt-in via `use_base2_exp` in tokamax splash attention.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Splash Attention](splash-attention.md)
- [Flash Attention](flash-attention.md)
- [Attention Block Sizes](attention-block-sizes.md)
- [Pallas Kernel](pallas-kernel.md)

## Sources

- [tokamax splash attention](../sources/2026-tokamax-splash-attention.md) — `raw/code/tokamax/docs/splash_attention.md`
