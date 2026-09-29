---
title: "How to Run a Hugging Face Model in JAX (Part 1): single-device forward + jax.jit"
type: source
tags: [blog, torchax, jax, huggingface, llama, jit, pytree, tpu-v6e]
author: Han Qi (qihqi)
upstream: https://github.com/qihqi/learning_machine/blob/main/jax-huggingface/01-run-huggingface-model-in-jax.md
companion_script: jax_hg_01.py
created: 2026-04-22
updated: 2026-04-22
---

Blog post #1 of a four-part series. Establishes the baseline: run `meta-llama/Llama-2-7b-hf` in bfloat16 single-device on TPU v6e via `torchax.extract_jax`, resolve pytree registration and static-arg errors, and JIT-compile the forward pass. Headline measured result: **first JIT call 4.365 s, cached calls 13 ms** on TPU v6e.

## Overview

The post frames itself as a stress test for [torchax](../codebases/torchax.md) — the interop library that makes PyTorch HuggingFace models callable as JAX functions. It walks through three failure modes you hit in order when you naively try to wrap an HF model in `jax.jit`, and resolves each: (1) the HF output type `CausalLMOutputWithPast` is not a JAX pytree; (2) the KV-cache type `DynamicCache` is not a JAX pytree either; (3) the `use_cache` kwarg triggers `ConcretizationTypeError` because HF branches on its boolean value during tracing. Fixes: `jax.tree_util.register_pytree_node` for (1) and (2); a closure that bakes `use_cache=False` for (3).

## Key claims

1. `torchax.extract_jax(model)` returns `(weights, func)` where `weights` is a pytree of JAX arrays (effectively `model.state_dict()` lifted to JAX) and `func(weights, args_tuple, kwargs_dict)` is a pure JAX callable that wraps `model.forward`.
2. HF output types must be registered with `register_pytree_node` before they can cross a `jax.jit` boundary. The post gives working registrations for `CausalLMOutputWithPast` (via `.to_tuple()`) and `DynamicCache` (via `(key_cache, value_cache)`).
3. When a function branches on a runtime Python value (like `use_cache`), either pass it as a static arg via `jax.jit(..., static_argnums=...)` or bake it in with a closure. The post uses the closure approach.
4. JIT compile-once / run-many-times produces the expected latency profile: the first invocation pays full compile cost; subsequent invocations are 100–300× faster on TPU v6e.
5. There is no semantic change from the closure/static-arg fix — the jitted output matches eager output element-wise (shown by copy-pasting both outputs).

## Key data points

### Llama-2-7B bfloat16, forward pass, single TPU v6e chip

| Iteration | Wall time | Notes |
|---|---|---|
| 0 (first jit call) | 4.365 s | includes compilation |
| 1 | 13.4 ms | cached |
| 2 | 13.0 ms | cached |

| Quantity | Value | Notes |
|---|---|---|
| Model | `meta-llama/Llama-2-7b-hf` | loaded in bfloat16 on CPU, then moved |
| Input shape | `(1, 12)` tokens | "The secret to baking a good cake is " |
| Hardware | Google Cloud TPU v6e | single chip |
| Speedup JIT vs eager | not measured directly | but cached 13ms is "milliseconds" vs "4s" ≈ 300× |

### Pytree registration signatures (verbatim)

| HF type | Flattener returns | Unflattener builds |
|---|---|---|
| `CausalLMOutputWithPast` | `(v.to_tuple(), None)` | `CausalLMOutputWithPast(*children)` |
| `DynamicCache` | `((dc.key_cache, dc.value_cache), None)` | `DynamicCache()`, then assign children |

## Techniques referenced

- **`torchax.extract_jax`** — single-entry PyTorch→JAX callable conversion. The returned signature `func(weights, args, kwargs)` enforces pure-function semantics.
- **`jax.tree_util.register_pytree_node`** — standard JAX API for teaching the pytree system about custom container types. The generalized pattern: write `flatten(v) -> (children, aux)` and `unflatten(aux, children) -> v`.
- **Closure-as-static-arg** — Python closures capturing concrete values bake those values into the compiled graph, avoiding `static_argnums` bookkeeping.
- **`jax.block_until_ready`** — required for correct wall-clock timing on TPU; JAX dispatches are asynchronous otherwise.

## Gaps & caveats

- **Hardware scope is narrow.** Only single-chip TPU v6e. No comparison with CPU, GPU, or other TPU generations.
- **No MFU or tokens/sec.** The 13 ms cached number is forward-pass wall time on 12 input tokens — not a decoding benchmark, not a per-token rate.
- **No memory measurement.** HBM usage is not reported. Llama-2-7B in bfloat16 is ~13.5 GB of weights; fits on v6e comfortably but leaves no headroom signal.
- **`use_cache=False` is forced.** The forward pass does not populate a KV cache in this part — decoding with cache is deferred to Part 3. The measurement is therefore prefill-only.
- **Claim that 13.48 GB of "captured constants" happens in jit is not from this part** — that specific number appears in Part 3. Part 1's jit call passes weights explicitly as first arg, so it does not hit the constant-inline path.

## Connections

Updates / informs:
- [codebases/jax-huggingface](../codebases/jax-huggingface.md) — canonical location for the `extract_jax` + pytree pattern and the v6e baseline numbers.
- [codebases/torchax](../codebases/torchax.md) — the `extract_jax` API entry point on the torchax parent page.

Anchors future hypotheses for:
- Any model page that uses Llama-2-7B on TPU v6e as a baseline — this post's 13 ms single-chip number is the reference floor for a single-chip prefill comparison.

## See also

- [kv-cache](../concepts/kv-cache.md) — `DynamicCache` is registered here but not exercised until Part 3.
- [Part 2](2026-jax-huggingface-part-2.md) — adds TP on top of this baseline.
- [Part 3](2026-jax-huggingface-part-3.md) — extends to autoregressive decode with `StaticCache`.

## Sources

- `raw/code/learning-machine/jax-huggingface/01-run-huggingface-model-in-jax.md`
- `raw/code/learning-machine/jax-huggingface/jax_hg_01.py`
- Upstream: <https://github.com/qihqi/learning_machine/blob/main/jax-huggingface/01-run-huggingface-model-in-jax.md>
