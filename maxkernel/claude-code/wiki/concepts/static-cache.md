---
title: "Static KV Cache"
type: concept
tags: [stub, inference, optimization]
created: 2026-04-22
updated: 2026-04-22
sources: 2
---

Fixed-shape KV cache that lets `jax.jit` avoid recompilation; worth ~8.8× on Llama-2-7B decode.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [KV Cache](kv-cache.md)
- [Continuous Batching](continuous-batching.md)
- [Decode Profile Signature](decode-profile-signature.md)
- [Serving Warmup](serving-warmup.md)
- [Mark-Step Sync](mark-step-sync.md)

## Sources

- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [JAX HuggingFace Part 3](../sources/2026-jax-huggingface-part-3.md) — `StaticCache` + `jax.jit` + `torch.func.functional_call` gives 8.87× speedup on Llama-2-7B decode (130.9 s → 14.77 s); pytree registration for `StaticCache` and prefill/decode jit split
