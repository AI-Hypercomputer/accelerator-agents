<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

---
title: "Splash Attention"
type: concept
tags: [stub, kernel, attention]
created: 2026-04-22
updated: 2026-04-22
sources: 4
---

TPU-native flash-family attention kernel; sparse-mask aware, supports MHA/MQA/GQA, soft-cap, separate fwd/bwd tiling. Available in `jax.experimental.pallas.ops.tpu.splash_attention` and mirrored in tokamax.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Flash Attention](flash-attention.md)
- [Attention Block Sizes](attention-block-sizes.md)
- [Base-2 Softmax](base2-softmax.md)
- [Pallas Kernel](pallas-kernel.md)
- [Mosaic Kernel](mosaic-kernel.md)

## Sources

- [tokamax splash attention](../sources/2026-tokamax-splash-attention.md) — `raw/code/tokamax/docs/splash_attention.md`
- [tokamax supported ops](../sources/2026-tokamax-supported-ops.md) — `raw/code/tokamax/docs/supported_ops.md`
- [tokamax basic usage](../sources/2026-tokamax-basic-usage.md) — `raw/code/tokamax/docs/basic_usage.md`
- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — splash-attention backward block sizes (`block_q_dkv`, `block_kv_dkv`, etc.) are the TPU analogue of FlashAttention v2's backward tiling; backward blocks not yet autotuned in tokamax
