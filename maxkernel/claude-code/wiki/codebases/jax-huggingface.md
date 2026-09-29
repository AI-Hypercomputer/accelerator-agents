---
title: "jax-huggingface (learning_machine subproject)"
type: codebase
tags: [torchax, jax, huggingface, llama, stable-diffusion, tensor-parallelism, kv-cache, pytree]
repo: qihqi/learning_machine
path: jax-huggingface/
commit: 93328b25d70d8a362a22f19d75f26d6d24c5ff1e
author: Han Qi (google)
created: 2026-04-22
updated: 2026-04-22
---

> **Grounded front door → [learning-machine overview](learning-machine/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

Four-part tutorial plus matching runnable scripts showing how to execute HuggingFace PyTorch models — Llama-2-7B (transformers) and Stable Diffusion 2-base (diffusers) — under JAX via [torchax](torchax.md). The series walks from eager `torchax.extract_jax` → `jax.jit` with pytree registration → 8-way tensor parallelism → `StaticCache` + `torch.func.functional_call` for fast autoregressive decoding → `torchax.compile` for diffusion pipelines. The scripts are the canonical reference implementations the blog text narrates.

## Overview

This subproject is the working repo for a four-part blog series by torchax contributor Han Qi (qihqi). Its thesis: you can keep using PyTorch HuggingFace models as-is under JAX by "trojan-horsing" `jax.Array` inside objects that duck-type as `torch.Tensor`. The four parts progressively harden the path from toy single-device forward pass to production-shaped tensor-parallel decoding and diffusion image generation.

Because it is a subfolder of the larger `learning_machine` self-study repo, the ingestion is scoped to `jax-huggingface/` only — the sibling directories (`llama_ref/`, `spmd_sharding/`, `torch_pallas/`, etc.) are independent mini-projects and out of scope for this page.

## Architecture

```
jax-huggingface/
  01-run-huggingface-model-in-jax.md       ← Part 1: single-device forward + jax.jit
  02-run-huggingface-model-distributed.md  ← Part 2: 8-way tensor parallelism
  03-run-huggingface-model-in-jax.md       ← Part 3: torchax env deep-dive + StaticCache decode
  04-run-hugging-face-model-in-jax.md      ← Part 4: Stable Diffusion via torchax.compile
  jax_hg_01.py   jax_hg_02.py   jax_hg_03.py   jax_hg_04.py   ← matching runnable scripts
  script.py                                ← alternative minimal-jit entry (variant of Part 1)
  torchax-demo.py                          ← minimal torchax env demo (Part 3 opener)
  *.png                                    ← figures referenced from the blog posts
```

Four conceptual boundaries run through the series:

1. **PyTorch→JAX interop boundary.** Three APIs with increasingly explicit control:
   - `torchax.extract_jax(model)` — returns `(weights_as_jax_pytree, jax_callable)`; Parts 1–2.
   - `torchax.compile(module, CompileOptions(...))` — wraps a `torch.nn.Module`, replacing forward (or a named method) with a `jax.jit`ed equivalent; returns a `torch.nn.Module` so it stays drop-in inside pipelines; Part 4.
   - `tx.interop.jax_jit(fn)` — jits a torch-taking/torch-returning function through a wrap/unwrap shim; Part 3.
2. **Array wrapping boundary.** A `torchax.Tensor` is a `torch.Tensor` subclass whose payload is a `jax.Array`. `tx.interop.torch_view(jax_arr)` and `tensor.to('jax')` are the two constructors; `tensor.apply_jax(fn, *a, **kw)` / `apply_jax_` applies a JAX function to the inner array (used to call `jax.device_put` with a `NamedSharding`).
3. **gSPMD sharding boundary.** A single `jax.make_mesh((N,), ('axis',))` mesh plus per-weight `NamedSharding(mesh, P('axis', None))` / `P(None, 'axis')` specs — no per-device code. Parts 2–3.
4. **HF-semantic-types ↔ JAX-pytree boundary.** Custom HF classes (`CausalLMOutputWithPast`, `DynamicCache`, `StaticCache`, `BaseModelOutputWithPooling`) must be registered with `jax.tree_util.register_pytree_node` before they can cross a `jax.jit` boundary.

## Key abstractions

| Abstraction | What it is | Script(s) | Notes |
|---|---|---|---|
| `torchax.extract_jax(model)` | One-shot converter: returns `(weights, func)` where `func(weights, args_tuple, kwargs_dict)` is a pure JAX callable | `jax_hg_01.py`, `jax_hg_02.py`, `script.py` | Weights become a dict-pytree of jax arrays; semantically a functional-call view of `model.forward` |
| `tx.default_env()` / `env:` ctx | Dispatch env that makes torch ops on `torchax.Tensor` run via JAX lowerings | `torchax-demo.py`, `jax_hg_03.py`, `jax_hg_04.py` | Required around any code that constructs or operates on torchax tensors |
| `model.to('jax')` | Moves an `nn.Module`'s parameters to torchax tensors in-place | `jax_hg_03.py`, `jax_hg_04.py` | Only works on `nn.Module`s — non-modules (e.g. `PNDMScheduler`) need manual tensor walk |
| `torchax.compile(m, CompileOptions(...))` | Returns an `nn.Module` whose selected methods are `jax.jit`-compiled | `jax_hg_04.py` | `methods_to_compile=['decode']` needed for VAE (default is only `forward`); `jax_jit_kwargs` forwarded to `jax.jit` |
| `tx.interop.jax_jit(fn)` | Jits a torch-in/torch-out function by unwrapping→jitting→rewrapping around `jax.jit` | `jax_hg_03.py` | Use when the function is not a module method |
| `torch.func.functional_call` | PyTorch stdlib: call a module with an explicit `state_dict`-shaped weights arg | `jax_hg_03.py` | Essential pattern for making weights an explicit jit input rather than captured constants |
| `register_pytree_node` (HF types) | Teach JAX how to flatten/unflatten HF output + cache classes | all four scripts | See Part 1 / Part 3 / Part 4 pytree cookbook — StaticCache flattener differs across HF versions |

## Entry points

All scripts are self-contained and meant to be run from this directory:

```bash
cd raw/code/learning-machine/jax-huggingface
python jax_hg_01.py       # Part 1: single-device Llama2-7B forward + jit timing loop
python jax_hg_02.py       # Part 2: 8-way TP, jax.profiler.trace to /tmp/jax-trace
python jax_hg_03.py       # Part 3: StaticCache + jax_jit autoregressive decode (50 tokens)
python jax_hg_04.py       # Part 4: Stable Diffusion 2-base, 20 inference steps, saves astronaut.png
python torchax-demo.py    # minimal torchax env demo (no HF model)
python script.py          # pared-down Part 1 variant
```

Hardware the posts were measured on:
- **Parts 1 & 2:** Google Cloud TPU v6e (8 chips per host).
- **Part 3:** hardware unspecified in the post text; script imports TPU-relevant code but the 14.77s number for 50 decode tokens is consistent with TPU v6e.
- **Part 4:** **A100 GPU** — not TPU. The `torchax.compile` pattern applies to TPU but the reported 5.9s→1.07s/image numbers do not transfer directly.

## Dependencies

```bash
pip install torchax
# or, per Part 3, latest dev (Part 3 requires bugfixes after initial torchax release):
pip install git+https://github.com/pytorch/xla.git#subdirectory=torchax

pip install -U transformers datasets evaluate accelerate timm flax
pip install diffusers          # Part 4 only
pip install jax[tpu]            # or jax[cuda12] for GPU
huggingface-cli login           # Llama-2 is gated
```

Python 3.10 (Parts 1–3) or 3.13 (Part 4's traceback path) both work.

## Notable files

### `jax_hg_01.py` (84 lines)
Part-1 reference. Loads `meta-llama/Llama-2-7b-hf` in bfloat16 on CPU, calls `torchax.extract_jax`, wraps in a `func_with_constant` closure that bakes `{'use_cache': False}` (the closure trick for static args), then times 3 jit iterations. Also registers pytree nodes for `CausalLMOutputWithPast` and `DynamicCache`. No sharding; single device.

### `script.py` (56 lines)
Minimal variant of `jax_hg_01.py` — same pytree registrations, same `func_with_constant`, but **no timing loop**. Useful as the shortest runnable demo of the eager→jit path.

### `jax_hg_02.py` (118 lines)
Part-2 reference. Adds:
- `jax.make_mesh((jax.device_count(),), ('axis',))` — the single-axis mesh.
- `shard_weights_llama(mesh, weights)` — the Llama tensor-parallel recipe:
  - `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj` → `P('axis', None)` (column-sharded).
  - `o_proj`, `down_proj`, `lm_head.weight`, `embed_tokens` → `P(None, 'axis')` (row-sharded).
  - everything else → `P()` (replicated).
- Input `input_ids` replicated via `NamedSharding(mesh, P())`.
- `jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=False)` wrapping the 3-iter timing loop.

The commented-out block at bottom (lines 104–117) sketches a path to `model.generate` via `JittableModule` — unused but gives hints for future KV-cache integration with the generate loop.

### `jax_hg_03.py` (274 lines)
Part-3 reference and the most important script in the series. Key structural differences from the blog text:
- **Uses `torchax.default_env()` directly** (not `extract_jax`). `model.to('jax')` moves weights in place, `load_state_dict(weights, assign=True, strict=False)` puts sharded weights back into the module.
- **`StaticCache` pytree flattener** (lines 54–78) iterates `cache.layers` with per-layer `.keys` / `.values` attributes, **not** `cache.key_cache` / `cache.value_cache` as the Part-3 post shows. This reflects the newer HF cache API. Hypothesis-writers: **the post's flattener will not work on current `transformers`; use the script's version.**
- **Prefill/decode split.** `prefill_fn(model_weights, input_ids, attention_mask, cache_position, past_key_values)` runs once for the prompt; `decode_one_tokens(model_weights, cur_token, input_pos, cache_position, past_key_values)` runs per generated token. Both are jitted via `tx.interop.jax_jit`.
- **Weights are an explicit jit input.** Both functions use `torch.func.functional_call` to avoid the 13.48 GB "captured constants" warning JAX emits if weights are closed-over implicitly.
- **KV cache sharded on num-heads axis after prefill.** Lines 228–231: `P(None, 'axis', None, None)` over cache shape `(batch, num_heads, seq, head_dim)`. This is the TP-style head-split sharding and must happen after prefill because `StaticCache` was constructed replicated.
- Per-iteration timing is printed (`Iteration i took ...`) alongside the aggregate 50-token wall time.

### `jax_hg_04.py` (66 lines)
Part-4 reference for `stabilityai/stable-diffusion-2-base`:
- `jax.config.update('jax_default_matmul_precision', 'high')` at module load — diffusion image quality is sensitive to matmul precision.
- `BaseModelOutputWithPooling` pytree registration (the diffusion text encoder's output type).
- `move_scheduler(pipe.scheduler)` — walks `scheduler.__dict__` and calls `.to('jax')` on any tensor. Required because `PNDMScheduler` is not an `nn.Module` and `pipe.to('jax')` skips it; without this, `alphas_cumprod[timestep]` raises `AttributeError: 'Tensor' object has no attribute '_env'` deep inside the scheduler's step function.
- `torchax.compile` applied to `pipe.unet` (with `static_argnames=('return_dict',)`), `pipe.vae` (with `methods_to_compile=['decode']` and same static arg), and `pipe.text_encoder` (defaults). The `methods_to_compile` override is load-bearing because the pipeline calls `vae.decode`, not `vae.forward`.
- 3 iterations of `pipe(prompt, num_inference_steps=20)`, printing each iteration's wall time, then saves `astronaut.png`.

### `torchax-demo.py` (35 lines)
Minimal demonstration of `tx.interop.torch_view(jax_array)` and `torch.ones((4,4), device='jax')` — both produce a `torch.Tensor` backed by a `jax.Array`. Runs `torch.matmul`, `torch.sin`, `torch.exp` inside `with env:`. Useful as the **hello-world for the "trojan tensor" pattern** described in Part 3.

### The four markdown posts
Each post is ingested as a separate `source/` page (see Connections); they contain the measured numbers and the narrative context the scripts don't. Figures:
- `tensor-parallelism.png` — NeMo-Megatron column/row sharding diagram (Part 2).
- `llm-predict.png` — shape diagram of autoregressive decoding (Part 3).
- `image-trojan.png` — the torchax "trojan tensor" illustration (Part 3).
- `image.png`, `hf_drop_jax.png` — TensorBoard/xprof screenshots.
- `astronaut.png` — Part 4 output image.

## Performance-relevant surfaces

These are the knobs, patterns, and gotchas that future hypotheses on TPU-hosted HF models will touch. All are grounded in specific script lines.

### 1. Sharding recipe for Llama-family models
`jax_hg_02.py:61-78` and `jax_hg_03.py:84-104`. The column/row assignment is mechanical:
- **Column-sharded (`P('axis', None)`):** Q, K, V, gate, up projections.
- **Row-sharded (`P(None, 'axis')`):** O, down projections, `lm_head.weight`, `embed_tokens`.
- **Replicated (`P()`):** everything else (layernorm weights, rotary freqs, biases).

For non-Llama architectures, the pattern transfers but the substring match (`'q_proj' in k` etc.) does not. Candidate hypothesis: a generic name-based sharder keyed on HF attention/MLP naming conventions.

### 2. KV cache sharding after prefill
`jax_hg_03.py:228-231`. `StaticCache` is allocated replicated, then cache tensors are moved to `NamedSharding(mesh, P(None, 'axis', None, None))` — shard on the num-heads axis — **after the prefill step**. This is the correct split because decode is dominated by cache attention and the num-heads axis is the only head-parallel axis. Doing it before prefill is wrong (prefill writes the cache).

### 3. `torch.func.functional_call` to escape captured-constants warning
`jax_hg_03.py:167-177`. If weights are closed-over (implicit in the decode function), `jax.jit` lowers them as constants in the HLO graph — the post reports 13.48 GB of captured constants. `functional_call(model, weights, args, kwargs)` makes weights an explicit jit input, shrinking the graph and (more importantly) letting donate-buffers / sharding propagation work on weights. Any future torchax hypothesis that jits around a whole model must follow this pattern.

### 4. `torchax.compile` method selection
`jax_hg_04.py:44-55`. `CompileOptions(methods_to_compile=['decode'])` is required for `vae` because the pipeline calls `pipe.vae.decode`, not `pipe.vae.forward`. Default compile-only-forward silently does nothing useful here, which is why the Part-4 post reports "we actually didn't run the compiled version of VAE" before fixing this. General lesson: for any HF/diffusers module used outside its `forward`, verify which method the caller actually invokes before claiming compile coverage.

### 5. `static_argnames` via `CompileOptions.jax_jit_kwargs`
`jax_hg_04.py:45-47`. `jax_jit_kwargs={'static_argnames': ('return_dict',)}` is the escape hatch for HF models that branch on `return_dict` (and similar kwargs `use_cache`, `output_attentions`) during tracing. Equivalent to `jax.jit`'s `static_argnames` but routed through torchax. Missing this produces `ConcretizationTypeError` that is easy to misdiagnose.

### 6. Scheduler tensor-move gotcha
`jax_hg_04.py:33-42`. Diffusion schedulers (`PNDMScheduler`, `DDIMScheduler`, ...) are not `nn.Module`s, so `pipe.to('jax')` leaves their tensor attributes (`alphas_cumprod`, `betas`, etc.) on CPU. The symptom is a crash deep inside `scheduler.step` with `'Tensor' object has no attribute '_env'`. The 3-line `move_scheduler` walk of `__dict__` fixes it but is not generalized — any new scheduler with nested tensor containers would need a deeper walk.

### 7. Pytree registration cookbook
Parts 1, 3, 4 each re-register the HF classes they need. Known working registrations:
- `CausalLMOutputWithPast` — `to_tuple()` / `__init__(*children)`.
- `DynamicCache` — `(key_cache, value_cache)`; `__init__()` then assign children.
- `StaticCache` (new API) — per-layer `keys` / `values`, aux is `(config, max_cache_len)`; see `jax_hg_03.py:54-78`.
- `BaseModelOutputWithPooling` — `(last_hidden_state, pooler_output, hidden_states, attentions)` / `__init__(*children)`.

Candidate concept: an "HF pytree registry" helper that pre-registers all common output/cache types would remove this boilerplate from every torchax+HF application.

### 8. Static-arg strategy: closure vs `static_argnums`
`jax_hg_01.py:53-55` (closure baking) vs `jax_hg_04.py` (`static_argnames`). The post recommends the closure approach for "constants you control" and static-argnames for "caller-provided flags you can't close over". Both compile equivalently.

### 9. `jax.default_matmul_precision = 'high'`
`jax_hg_04.py:12`. Sets the default precision for all matmuls in the diffusion pipeline. Diffusion model quality degrades visibly at `'default'` (bfloat16 matmul) on TPU. For Llama inference the post does not set this, which suggests bfloat16 matmul was acceptable. Candidate observation: diffusion ≠ LLM inference in precision tolerance.

### 10. Profile capture idiom
`jax_hg_02.py:96-102`. `jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=False)` wrapping the timing loop, viewed in TensorBoard via the xprof plugin. The profile for this exact snippet is the reference artifact for "is TP actually using all 8 chips". Matches the [xprof](xprof.md) / [xprof-mcp](xprof-mcp.md) profile-capture surface.

## Connections

Source pages ingested alongside this codebase:
- [Part 1 — single-device forward + jit](../sources/2026-jax-huggingface-part-1.md)
- [Part 2 — 8-way tensor parallelism](../sources/2026-jax-huggingface-part-2.md)
- [Part 3 — StaticCache + jax_jit decode](../sources/2026-jax-huggingface-part-3.md)
- [Part 4 — Stable Diffusion via torchax.compile](../sources/2026-jax-huggingface-part-4.md)

Related codebases:
- [torchax](torchax.md) — the interop library this entire subproject exercises; `torchax.extract_jax`, `torchax.compile`, `torchax.default_env()`, `tx.interop.jax_jit`, `tx.interop.torch_view` are all documented on the parent page.
- [xprof](xprof.md) / [xprof-mcp](xprof-mcp.md) — profile-capture and analysis for the `jax.profiler.trace` artifacts produced by `jax_hg_02.py`.

Future hypothesis anchors (not filed — no `model/` page yet):
- Llama-2-7B tensor-parallel forward: script `jax_hg_02.py` is a ready-made baseline.
- Llama-2-7B autoregressive decode with StaticCache: `jax_hg_03.py` is a ready-made baseline; target metric is per-token decode latency after the first.
- HF SDPA → tokamax splash-attention swap inside any of the above scripts (would touch [tokamax](tokamax.md)).
- The StaticCache flattener discrepancy between Part-3 post text and `jax_hg_03.py` is an "HF API drift" note worth a standalone observation once a model page exists.

## See also

- [tensor-parallelism](../concepts/tensor-parallelism.md) — NeMo-Megatron scheme used in Part 2.
- [sharding](../concepts/sharding.md) — gSPMD sharding primitives the scripts use.
- [kv-cache](../concepts/kv-cache.md) — Part 3 foundation.
- [static-cache](../concepts/static-cache.md) — the HF cache class used for jit-friendly decode.
- [all-reduce](../concepts/all-reduce.md) — the one collective inserted per attention/MLP block under TP.
- [profile-capture](../concepts/profile-capture.md) — `jax.profiler.trace` idiom used in `jax_hg_02.py`.
- [jax-trace](../concepts/jax-trace.md) — format of the traces viewed in xprof.
- [splash-attention](../concepts/splash-attention.md) — candidate kernel to swap in for HF SDPA in these pipelines.

## Sources

- `raw/code/learning-machine/jax-huggingface/01-run-huggingface-model-in-jax.md`
- `raw/code/learning-machine/jax-huggingface/02-run-huggingface-model-distributed.md`
- `raw/code/learning-machine/jax-huggingface/03-run-huggingface-model-in-jax.md`
- `raw/code/learning-machine/jax-huggingface/04-run-hugging-face-model-in-jax.md`
- `raw/code/learning-machine/jax-huggingface/jax_hg_01.py`
- `raw/code/learning-machine/jax-huggingface/jax_hg_02.py`
- `raw/code/learning-machine/jax-huggingface/jax_hg_03.py`
- `raw/code/learning-machine/jax-huggingface/jax_hg_04.py`
- `raw/code/learning-machine/jax-huggingface/script.py`
- `raw/code/learning-machine/jax-huggingface/torchax-demo.py`
- Upstream: <https://github.com/qihqi/learning_machine/tree/main/jax-huggingface>
