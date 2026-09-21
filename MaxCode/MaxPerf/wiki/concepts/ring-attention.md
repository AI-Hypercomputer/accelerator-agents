<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

---
title: "Ring Attention"
type: concept
tags: [stub, kernel, attention, parallelism]
created: 2026-04-22
updated: 2026-04-22
sources: 1
---

Streams K/V chunks around a ring of devices so each device holds only `1/N` of the sequence in memory; enables sequence lengths beyond single-device HBM. Naive causal-masked Ring Attention wastes ~50% of compute on the lower triangle; **Zig-Zag Ring Attention** (Brandon et al. 2023) rebalances. TPU kernel exists at `tokamax/_src/ops/experimental/...ring_attention_kernel` but is **not reachable** from `tokamax.dot_product_attention` at this commit — hypothesis surface.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Splash Attention](splash-attention.md)
- [Flash Attention](flash-attention.md)
- [Context Parallelism](context-parallelism.md)
- [ICI (Inter-Chip Interconnect)](ici.md)

## Sources

- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — Section 5 (Context Parallelism)
