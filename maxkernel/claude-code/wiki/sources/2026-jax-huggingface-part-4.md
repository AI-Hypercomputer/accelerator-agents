---
title: "How to Run a Hugging Face Model in JAX (Part 4): Stable Diffusion via torchax.compile"
type: source
tags: [blog, torchax, jax, diffusers, stable-diffusion, vae, unet, compile-options, a100]
author: Han Qi (qihqi)
upstream: https://github.com/qihqi/learning_machine/blob/main/jax-huggingface/04-run-hugging-face-model-in-jax.md
companion_script: jax_hg_04.py
created: 2026-04-22
updated: 2026-04-22
---

Blog post #4 of a four-part series. Applies the pattern to an image-generation pipeline: `stabilityai/stable-diffusion-2-base` via `diffusers.StableDiffusionPipeline`. Shows `torchax.compile` (instead of `extract_jax`) because the pipeline is not a single `nn.Module`. Headline measured result: **5.9 s → 1.07 s per image on A100 GPU** after correctly compiling the VAE's `decode` method. Hardware here is **A100 GPU, not TPU** — numbers do not transfer, but the patterns do.

## Overview

`StableDiffusionPipeline` is an orchestrator, not a `torch.nn.Module` — so `torchax.extract_jax` does not apply. The post instead uses `torchax.compile`, which wraps individual `nn.Module`s (`pipe.unet`, `pipe.vae`, `pipe.text_encoder`) with a forward that's `jax.jit`-compiled but preserves the `nn.Module` interface so the pipeline continues to work unmodified.

Four problems surface while making this run, each a concrete `CompileOptions` / API detail:
1. HF's `BaseModelOutputWithPooling` is not a JAX pytree → `register_pytree_node`.
2. HF modules branch on `return_dict` → `CompileOptions(jax_jit_kwargs={'static_argnames': ('return_dict',)})`.
3. `pipe.scheduler` is not an `nn.Module`, so `pipe.to('jax')` does not move its tensor attributes (`alphas_cumprod`, etc.) → manual `move_scheduler` walk of `__dict__`.
4. The pipeline calls `vae.decode`, not `vae.forward` — so default `torchax.compile` (which only wraps `forward`) silently fails to compile VAE → `CompileOptions(methods_to_compile=['decode'])`.

Fixing (1)–(3) gets the pipeline running; fixing (4) is what actually produces the 5.9× speedup.

## Key claims

1. **`torchax.compile` is the right API when you cannot isolate a single `nn.Module`** — it drops in as a replacement for `torch.compile` and returns a `torch.nn.Module` so it composes with existing pipelines.
2. **`CompileOptions.jax_jit_kwargs` forwards arbitrary kwargs to the underlying `jax.jit`** — the general escape hatch when HF models branch on kwargs like `return_dict`, `use_cache`, `output_attentions`.
3. **`CompileOptions.methods_to_compile` controls which methods are jitted.** Default is `['forward']`. For `AutoencoderKL`, the pipeline invokes `decode`, so `methods_to_compile=['decode']` is required for actual speedup.
4. **Non-`nn.Module` pipeline components (schedulers, tokenizers, feature extractors) are not covered by `pipe.to('jax')`.** A manual `__dict__` walk is needed for any tensor attribute.
5. **Diffusion quality is precision-sensitive.** The script sets `jax.config.update('jax_default_matmul_precision', 'high')` at module load — this is **not** present in the Llama scripts, suggesting bfloat16 matmul was acceptable for LLM inference but not for SD2.
6. **The compile-time cost is amortized aggressively.** First iteration 53.9 s (includes all three compiles), second and third 1.07 s — compile-cost crossover is after just one image.

## Key data points

### Stable Diffusion 2-base, 20 inference steps, batch=1, **A100 GPU (not TPU)**

| Iteration | Wall time | Notes |
|---|---|---|
| 0 (first) | 53.95 s | includes unet + vae.decode + text_encoder compile |
| 1 | 1.07 s | cached, all three modules compiled |
| 2 | 1.07 s | cached, steady-state |

| Quantity | Value | Notes |
|---|---|---|
| Before VAE-decode fix | 5.9 s / image | VAE running uncompiled (default `methods_to_compile=['forward']`) |
| After VAE-decode fix | 1.07 s / image | ~5.5× faster per image |
| Iteration rate (cached) | 19.7–19.8 it/s | 20 UNet steps complete in ~1 s |

### `CompileOptions` usage summary (verbatim from script)

| Module | `methods_to_compile` | `static_argnames` |
|---|---|---|
| `pipe.unet` | default (`['forward']`) | `('return_dict',)` |
| `pipe.vae` | **`['decode']`** | `('return_dict',)` |
| `pipe.text_encoder` | default (`['forward']`) | default (none) |

### `BaseModelOutputWithPooling` pytree registration

| Part | Content |
|---|---|
| children | `(v.last_hidden_state, v.pooler_output, v.hidden_states, v.attentions)` |
| aux | `None` |
| unflatten | `BaseModelOutputWithPooling(*children)` |

### `move_scheduler` pattern (verbatim)

```python
def move_scheduler(scheduler):
  for k, v in scheduler.__dict__.items():
    if isinstance(v, torch.Tensor):
      setattr(scheduler, k, v.to('jax'))
```

Applied after `pipe.to('jax')`. Fixes the crash `AttributeError: 'Tensor' object has no attribute '_env'` inside `scheduler.step` when `alphas_cumprod[timestep]` is indexed.

## Techniques referenced

- **`torchax.compile(module, CompileOptions(...))`** — the primary API for this post; returns a drop-in `nn.Module` with jitted methods.
- **`CompileOptions.methods_to_compile`** — whitelist of methods (not just `forward`) to jit.
- **`CompileOptions.jax_jit_kwargs`** — forwarded to `jax.jit`; most commonly used for `static_argnames`.
- **`jax.config.update('jax_default_matmul_precision', 'high')`** — module-global matmul precision override for image-quality-sensitive workloads.
- **`register_pytree_node` for HF diffusers output types** — same mechanical pattern as Parts 1 and 3.
- **Python-stdlib `__dict__` walk** — used here as a crude but effective way to move scheduler tensors when `nn.Module.to` does not apply.

## Gaps & caveats

- **Not TPU.** Hardware is A100 GPU. The 5.9 s → 1.07 s number does not translate to v6e/v5p. The series prefix "Hugging Face in JAX" does apply to TPU, but this specific measurement is GPU-only.
- **No MFU, no bandwidth utilization.** Just wall-clock per-image.
- **`num_inference_steps=20` is a quality-tradeoff choice** — production image quality usually runs 30–50 steps. The post does not discuss how compile cost scales with step count (it shouldn't — UNet compile is step-count-invariant — but this is not stated).
- **VAE decode is the only `methods_to_compile` fix shown** — the post does not audit whether `text_encoder.forward` vs `text_encoder.__call__` is actually what diffusers calls, nor does it discuss `encode` vs `decode` of VAE for img2img pipelines.
- **No profile artifact.** Part 2 showed an xprof screenshot; Part 4 does not — so claims about where time is spent (unet vs vae vs text_encoder) are inferred from the VAE-fix speedup, not measured.
- **No mention of CFG (classifier-free guidance) batching.** SD pipelines typically double-batch for CFG; how that interacts with static shapes is not addressed.
- **Scheduler walk is specific to PNDMScheduler's flat `__dict__`.** Schedulers with nested containers (lists/dicts of tensors) would need a deeper walk.

## Connections

Updates / informs:
- [codebases/jax-huggingface](../codebases/jax-huggingface.md) — canonical location for `torchax.compile` + `CompileOptions` patterns, the scheduler-move gotcha, and the VAE `methods_to_compile` gotcha.
- [codebases/torchax](../codebases/torchax.md) — the `compile` / `CompileOptions` API.

Future hypothesis anchors (not filed — no `model/` page yet; and this part is GPU not TPU):
- **Stable Diffusion 2-base on TPU v6e with the same `torchax.compile` setup** — directly porting this script to TPU would produce the first in-wiki TPU baseline for SD. Likely substantially different numbers.
- **VAE-decode vs UNet time split** — the 5.9 → 1.07 s gap implies VAE.decode was dominant when uncompiled; confirming this on TPU with `jax.profiler.trace` is a quick win.
- **`jax_default_matmul_precision` sweep for diffusion on TPU** — Part 4 sets `'high'` without quantification; the step between `'default'` (bf16) and `'high'` (fp32-approximation) is a concrete TPU perf lever.

## See also

- [dtype-strategy](../concepts/dtype-strategy.md) — the broader precision discussion (`jax_default_matmul_precision` lives here).
- [mxu](../concepts/mxu.md) — matmul precision is ultimately an MXU configuration.
- [Part 1](2026-jax-huggingface-part-1.md) — first `register_pytree_node` pattern, reused here for `BaseModelOutputWithPooling`.
- [Part 3](2026-jax-huggingface-part-3.md) — the `tx.interop.jax_jit` approach `torchax.compile` wraps for you.

## Sources

- `raw/code/learning-machine/jax-huggingface/04-run-hugging-face-model-in-jax.md`
- `raw/code/learning-machine/jax-huggingface/jax_hg_04.py`
- `raw/code/learning-machine/jax-huggingface/astronaut.png` (output)
- Upstream: <https://github.com/qihqi/learning_machine/blob/main/jax-huggingface/04-run-hugging-face-model-in-jax.md>
