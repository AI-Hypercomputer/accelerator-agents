<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

---
title: "Flash Attention"
type: concept
tags: [stub, kernel, attention]
created: 2026-04-22
updated: 2026-04-22
sources: 4
---

Tiled SRAM-resident attention algorithm (Algorithm 1 of Dao et al.); avoids materializing the N×N attention matrix in HBM.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Splash Attention](splash-attention.md)
- [Attention Block Sizes](attention-block-sizes.md)
- [VMEM (Vector Memory)](vmem.md)
- [Pallas Kernel](pallas-kernel.md)
- [Base-2 Softmax](base2-softmax.md)

## Sources

- [tokamax supported ops](../sources/2026-tokamax-supported-ops.md) — `raw/code/tokamax/docs/supported_ops.md`
- [tokamax basic usage](../sources/2026-tokamax-basic-usage.md) — `raw/code/tokamax/docs/basic_usage.md`
- [tokamax splash attention](../sources/2026-tokamax-splash-attention.md) — `raw/code/tokamax/docs/splash_attention.md`
- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — FlashAttention v1/v2 walkthrough; TPU analogue is splash attention
