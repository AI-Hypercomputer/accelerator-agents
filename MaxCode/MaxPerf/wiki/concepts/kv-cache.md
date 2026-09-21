<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

---
title: "KV Cache"
type: concept
tags: [stub, inference]
created: 2026-04-22
updated: 2026-04-22
sources: 2
---

Per-token key/value cache consumed during autoregressive decode; streaming the full cache through HBM at each step makes decode bandwidth-bound at small batch.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Static KV Cache](static-cache.md)
- [Continuous Batching](continuous-batching.md)
- [Decode Profile Signature](decode-profile-signature.md)
- [HBM (High-Bandwidth Memory)](hbm.md)
- [Memory-Bound](memory-bound.md)

## Sources

- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [JAX HuggingFace Part 3](../sources/2026-jax-huggingface-part-3.md) — HF `DynamicCache` grows per-step (shape `(batch, heads, seq, head_dim)`); KV cache sharded on num-heads axis (`P(None, 'axis', None, None)`) post-prefill; Llama-2-7B 50-tok decode 130.9 s baseline
