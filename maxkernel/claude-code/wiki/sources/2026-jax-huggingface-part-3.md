---
title: "How to Run a Hugging Face Model in JAX (Part 3): StaticCache + jax.jit autoregressive decoding"
type: source
tags: [blog, torchax, jax, huggingface, llama, kv-cache, static-cache, functional-call, autoregressive-decode]
author: Han Qi (qihqi)
upstream: https://github.com/qihqi/learning_machine/blob/main/jax-huggingface/03-run-huggingface-model-in-jax.md
companion_script: jax_hg_03.py
created: 2026-04-22
updated: 2026-04-22
---

Blog post #3 of a four-part series. Opens with a deep-dive on how `torchax` actually works (it wraps `jax.Array` in a `torch.Tensor` subclass — the "trojan tensor"), then builds an autoregressive decoder that is fast enough to be useful: **50-token Llama-2-7B decode in 14.77 s** with `StaticCache` + `tx.interop.jax_jit` + `torch.func.functional_call`, down from **130.9 s** for `DynamicCache` eager and **88.4 s** for `StaticCache` eager-no-jit. An 8.9× speedup.

## Overview

The post is structured around three progressively-more-optimized decoders. Each step reveals a specific JAX+HF impedance-mismatch and fixes it:

1. **`DynamicCache` eager decode** (baseline). Works, but 130 s for 50 tokens is unusably slow.
2. **`StaticCache` eager decode.** Faster (88 s) just by virtue of pre-allocated fixed-shape buffers — no jit yet.
3. **`StaticCache` + `jax.jit`.** Two new problems surface: (a) `StaticCache` is not a JAX pytree (fixed by `register_pytree_node`); (b) implicit weight capture produces a **13.48 GB captured-constants warning** that makes compilation slow (fixed by passing weights as explicit arg via `torch.func.functional_call`). After both fixes, 14.77 s.

The opening third of the post is essential context: it explains that `torchax.Tensor` is literally a subclass of `torch.Tensor` whose `_elem` is a `jax.Array`, and that `tx.default_env()` is a dispatch context that makes torch ops on such tensors run through JAX lowerings. `tx.interop.torch_view(jax_arr)` and `torch.ones(..., device='jax')` are the two supported constructors; `tensor.apply_jax(fn, *a, **kw)` applies a JAX function to the inner array in place (used for sharding).

## Key claims

1. **`torchax.Tensor` is a trojan horse:** the PyTorch dispatcher sees `torch.Tensor`, the payload is `jax.Array`, ops go through JAX.
2. **`StaticCache` exists specifically for `torch.compile` / static-shape compilation.** Using it with `jax.jit` is the natural adaptation, but its pytree registration must match the HF-version-specific layout.
3. **Implicit weight capture is catastrophic for compile time at Llama-2-7B scale.** 13.48 GB of inlined constants slow lowering and waste graph memory. Fix: `torch.func.functional_call(model, weights_dict, args, kwargs)` to make weights an explicit jit input.
4. **Pytree flattener for `StaticCache` changed across HF versions.** The blog text shows `(cache.key_cache, cache.value_cache)`; the companion script [`jax_hg_03.py`](../codebases/jax-huggingface.md#jax_hg_03py-274-lines) uses per-layer `cache.layers[i].keys` / `.values`. **The script is the current-HF-API version; the post text will fail on new transformers.**
5. **KV cache should be sharded on the num-heads axis** (`P(None, 'axis', None, None)` for shape `(batch, heads, seq, head_dim)`), but only **after prefill** — `StaticCache` is allocated replicated.
6. **Prefill and decode need separate jitted functions** with different input signatures (prefill takes `attention_mask`, decode takes `cur_token` and `cache_position`). Same model, same weights, different compile.

## Key data points

### Llama-2-7B bfloat16, 50-token autoregressive decode

| Approach | Wall time (50 tok) | Per-token (decode) | Notes |
|---|---|---|---|
| `DynamicCache`, eager | **130.9 s** | ~2.6 s | shapes grow every step |
| `StaticCache`, eager | **88.4 s** | ~1.77 s | fixed shapes, no jit |
| `StaticCache` + `tx.interop.jax_jit` + `functional_call` | **14.77 s** | ~0.29 s | prefill + decode jitted separately |

Relative speedups:
- `StaticCache` alone (no jit): **1.48× faster** than `DynamicCache` baseline.
- Full jit path: **8.87× faster** than `DynamicCache` baseline, **5.99× faster** than `StaticCache`-eager.

### KV cache shapes (Llama-2-7B, `num_layers=32`, `num_heads=32`, `head_dim=128`, `batch=1`, prompt length 12)

| Stage | Cache key/value shape per layer |
|---|---|
| After first forward | `(1, 32, 12, 128)` |
| After second forward (+1 token) | `(1, 32, 13, 128)` |

With `StaticCache(max_cache_len=max_tokens)`, both stages have shape `(1, 32, max_cache_len, 128)` — the length axis is fixed and values are written into slots indexed by `cache_position`.

### `StaticCache` pytree registration (current HF API, per `jax_hg_03.py`)

| Part | Content |
|---|---|
| children | `([layer.keys for layer in cache.layers], [layer.values for layer in cache.layers])` |
| aux | `(cache._config, cache.max_cache_len)` |
| unflatten | `StaticCache(config=_config, max_cache_len=max_cache_len)`, then overwrite per-layer `keys`/`values` |

## Techniques referenced

- **`torch.func.functional_call(module, state_dict, args, kwargs)`** — PyTorch stdlib API for calling a module with an explicit weights dict. The single most important pattern in this post: it converts implicit weight-capture into an explicit jit input, which avoids the 13.48 GB captured-constants warning and allows donate-buffers / sharding propagation on weights.
- **`tx.interop.jax_jit(fn)`** — torchax helper; conceptually `torch.compile(mode='jax')` for arbitrary torch functions (not just modules).
- **`StaticCache`** — HF fixed-length KV cache introduced for `torch.compile` support. See <https://huggingface.co/docs/transformers/v4.44.0/en/llm_optims?static-kv=advanced+usage%3A+control+Static+Cache#static-kv-cache-and-torchcompile>.
- **Prefill/decode split** — the standard LLM-inference structural decomposition, here materialized as two separate jitted functions with different signatures.
- **Post-prefill cache resharding** — `apply_jax_(jax.device_put, NamedSharding(mesh, P(None, 'axis', None, None)))` per-layer, per-tensor.
- **`tensor.apply_jax_` (trailing underscore)** — in-place variant of `apply_jax`; the underscore matches PyTorch's in-place convention.

## Gaps & caveats

- **Hardware is unstated.** The post does not say which device was used for the 130 s / 88 s / 14.77 s numbers. Given the series context, TPU v6e is likely. Part 3 absolutely needs this disambiguation before any of its numbers can be used as a baseline.
- **`jax_hg_03.py` does sharding on weights (via `shard_weights_llama`) but the post text does not mention TP for the decode run.** If the 14.77 s number is on 8 chips, it is not comparable to a 1-chip decode.
- **No per-token latency decomposition.** The script prints per-iteration timing but the post only reports the aggregate.
- **The `StaticCache` flattener in the post text will not work on current `transformers`** — the blog and the script diverge. See claim 4 above.
- **No memory measurement.** The 13.48 GB captured-constants number is graph-side, not HBM-side. Peak HBM under each of the three decoders is not reported.
- **No quality check.** The generated text ("100% butter...") differs slightly between runs (Part 3 shows a comma-separated token list vs Part 2's space-joined sentence). Likely matmul-precision drift; not investigated.
- **Batch size is hard-coded to 1.** Per-token decode at batch=1 is the memory-bandwidth regime, so the conclusion "jit is 6× faster than eager" is specifically about this regime.

## Connections

Updates / informs:
- [codebases/jax-huggingface](../codebases/jax-huggingface.md) — canonical location for the StaticCache + functional_call pattern and the decode numbers.
- [codebases/torchax](../codebases/torchax.md) — `default_env()`, `interop.torch_view`, `interop.jax_jit`, `apply_jax` / `apply_jax_` all documented here.

Future hypothesis anchors (not filed — no `model/` page yet):
- **Per-token decode latency on TPU v6e / v5p** once the hardware disambiguation is resolved.
- **KV cache sharding strategies** — this post establishes head-axis sharding post-prefill. Alternatives (head+seq 2D, batch-aware) are not explored.
- **Prefill/decode graph compile cost amortization** — compiling two graphs vs one is a concrete cost the post glosses over.
- **tokamax splash-attention inside decode** — the current decode presumably uses whatever attention HF SDPA resolves to; swapping to a TPU Pallas kernel via [tokamax](../codebases/tokamax.md) is a direct candidate.

## See also

- [kv-cache](../concepts/kv-cache.md) — the general KV-cache concept.
- [static-cache](../concepts/static-cache.md) — the specific HF class this post uses.
- [sharding](../concepts/sharding.md) — for the post-prefill cache reshard on num-heads axis.
- [tensor-parallelism](../concepts/tensor-parallelism.md) — the weight-sharding recipe carried over from Part 2.
- [splash-attention](../concepts/splash-attention.md) — candidate kernel swap inside decode.
- [Part 1](2026-jax-huggingface-part-1.md) — first `register_pytree_node` occurrence.
- [Part 2](2026-jax-huggingface-part-2.md) — TP recipe reused in `shard_weights_llama`.
- [Part 4](2026-jax-huggingface-part-4.md) — the `torchax.compile` approach for non-`nn.Module` pipelines.

## Sources

- `raw/code/learning-machine/jax-huggingface/03-run-huggingface-model-in-jax.md`
- `raw/code/learning-machine/jax-huggingface/jax_hg_03.py`
- `raw/code/learning-machine/jax-huggingface/torchax-demo.py` (the torchax-env demo referenced in the post opener)
- `raw/code/learning-machine/jax-huggingface/image-trojan.png` (trojan-tensor diagram)
- `raw/code/learning-machine/jax-huggingface/llm-predict.png` (decode shapes diagram)
- Upstream: <https://github.com/qihqi/learning_machine/blob/main/jax-huggingface/03-run-huggingface-model-in-jax.md>
