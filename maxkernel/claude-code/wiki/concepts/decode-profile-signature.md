---
title: "Decode Profile Signature"
type: concept
tags: [stub, inference, diagnostics]
created: 2026-04-22
updated: 2026-04-22
sources: 2
---

Characteristic shape of a bandwidth-bound autoregressive decode profile: dynamic-slice / update-slice dominant + TPU duty cycle <60%.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [KV Cache](kv-cache.md)
- [Static KV Cache](static-cache.md)
- [Continuous Batching](continuous-batching.md)
- [TPU Duty Cycle](tpu-duty-cycle.md)
- [Memory-Bound](memory-bound.md)

## Sources

- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [JAX HuggingFace Part 3](../sources/2026-jax-huggingface-part-3.md) — DynamicCache vs StaticCache decode shapes; per-token timing at batch=1 is the memory-bandwidth regime; dynamic-slice / update-slice dominate decode under `StaticCache`
