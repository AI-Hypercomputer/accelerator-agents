---
title: marin (levanter) — overview
type: overview
updated: 2026-07-03
---
# marin (levanter) — what it is and how it fits together

## In one paragraph

This grounded catalog covers `lib/levanter` — the JAX training/serving library vendored inside the
`marin` monorepo (marin's own `lib/marin` package, plus data-processing/evaluation/experiment-tracking
code elsewhere in the repo, are out of scope here; see `config/marin.md` for the exact index_shards).
Levanter's central idea is one shared `Attention` layer that dispatches across four TPU/GPU attention
backends (vanilla, pure-JAX flash, TPU Pallas Splash, transformer-engine flash) behind one structured
`AttentionMask` abstraction, and one `LmHeadModel`/`LmConfig` registry so every concrete architecture
(Llama, OLMo 2, GPT-2, Mistral, ...) plugs into the same training loop (`Trainer`), checkpointing
(`save_checkpoint`), and continuous-batching inference engine (`InferenceEngine`) without
architecture-specific glue code.

## Core architecture

```mermaid
flowchart TD
  LMCONFIG["LmConfig registry\n(discover_packages_path='levanter.models')"] -.-> LLAMA["LlamaConfig / Olmo2Config / ..."]
  LLAMA -->|attention_config()| ATTN["Attention (root.md)\nbackend-dispatching, paged-decode-capable"]
  ATTN --> MASK["AttentionMask\n(causal/sliding-window/segment/explicit)"]
  ATTN --> ROPE["RotaryEmbeddingsConfig\n(Default/Llama3/Yarn)"]
  LLAMA --> LMHEAD["LmHeadModel.compute_next_token_loss"]
  LMHEAD --> TRAINER["Trainer.train_step\n(_compute_gradients_microbatched -> take_step)"]
  TRAINER --> OPT["OptimizerConfig/LrSchedule registry"]
  TRAINER --> CKPT["save_checkpoint (TensorStore/OCDBT)"]
  LMHEAD --> ENGINE["InferenceEngine.generate\n(prefill + _run_generation_loop)"]
  ENGINE --> SCHED["DecodeState/SequenceTable/PageTable\n(jit_scheduler)"]
  SCHED --> ATTN
```

## Main concepts

**One `Attention` layer dispatches across four compute backends behind one structured mask.**
Vanilla, pure-JAX flash, TPU Splash (Pallas), and transformer-engine flash all consume the same
`AttentionMask`; Splash additionally lowers that mask into its own mask-object algebra and derives
per-shard block sizes. See [root](concepts/root.md) and
[lib-levanter-src-levanter-layers-attention_mask](concepts/lib-levanter-src-levanter-layers-attention_mask.md).

**`AttentionMask` is one flat class combined by implicit conjunction, not an inheritance hierarchy** —
an explicit, self-acknowledged trade-off for `jit`-friendliness. See
[lib-levanter-src-levanter-layers-attention_mask](concepts/lib-levanter-src-levanter-layers-attention_mask.md).

**RoPE frequency schedules are a pluggable registry (Default/Llama3/YaRN), decoupled from the
attention layer itself.** See [lib-levanter-src-levanter-layers-rotary](concepts/lib-levanter-src-levanter-layers-rotary.md).

**Continuous-batching inference is a fully-jitted scheduler (`DecodeState`/`SequenceTable`/`PageTable`)
driving a paged KV cache, with a separate generation-loop engine on top.** Every scheduling mutation —
slot assignment, page allocation, token commit, page freeing — is itself JIT-compiled, not host-side
Python. See [lib-levanter-src-levanter-inference-jit_scheduler](concepts/lib-levanter-src-levanter-inference-jit_scheduler.md)
and [lib-levanter-src-levanter-inference-engine](concepts/lib-levanter-src-levanter-inference-engine.md).

**Every model architecture converts its own config into one shared `AttentionConfig` and round-trips
with HuggingFace via `from_hf_config`/`to_hf_config`.** See
[lib-levanter-src-levanter-models-llama](concepts/lib-levanter-src-levanter-models-llama.md) and
[lib-levanter-src-levanter-models-olmo](concepts/lib-levanter-src-levanter-models-olmo.md).

**`LmConfig`/`LmHeadModel` form one registry + one abstract interface every architecture satisfies**,
so the training loop, loss computation, and inference engine are architecture-agnostic. See
[lib-levanter-src-levanter-models-lm_model](concepts/lib-levanter-src-levanter-models-lm_model.md).

**The training loop splits hook execution (checkpointing, tracking) from the JIT-compiled step
itself**, and optimizer/LR-schedule selection is its own pluggable registry. See
[lib-levanter-src-levanter-trainer](concepts/lib-levanter-src-levanter-trainer.md),
[lib-levanter-src-levanter-checkpoint](concepts/lib-levanter-src-levanter-checkpoint.md), and
[lib-levanter-src-levanter-config](concepts/lib-levanter-src-levanter-config.md).

**Expert-parallel transport has a GPU/CUDA-specific backend (DeepEP) built via a runtime JIT
compile-and-cache pipeline**, one of several interchangeable transport strategies under `kernels/`.
See [lib-levanter-src-levanter-kernels-deepep-transport_ffi](concepts/lib-levanter-src-levanter-kernels-deepep-transport_ffi.md).

## How a request flows

**Training:** `LmConfig` resolves an architecture → `LlamaConfig.attention_config()` builds the shared
`AttentionConfig` → `LlamaDecoderLayer.init` assembles the model → `Trainer.training_steps` iterates
batches, each `train_step` computing microbatched gradients
(`LmHeadModel.compute_next_token_loss` → `activations` → `dot_product_attention`) and applying the
optimizer update → periodic `save_checkpoint` hooks.

**Serving:** `InferenceEngine.generate` admits requests via `_prefill_batch`/`_prefill_kernel`
(allocating pages via the `jit_scheduler`, running the model's `paged_decode`), then
`_run_generation_loop` repeats allocate→forward→`update_tokens` each round until every sequence
finishes.

## Map of the wiki

- "How does attention backend dispatch (vanilla/flash/Splash/TE) work?" → [root](concepts/root.md).
- "What is `AttentionMask` and how is it combined/materialized?" →
  [lib-levanter-src-levanter-layers-attention_mask](concepts/lib-levanter-src-levanter-layers-attention_mask.md).
- "How do RoPE variants work?" → [lib-levanter-src-levanter-layers-rotary](concepts/lib-levanter-src-levanter-layers-rotary.md).
- "How does continuous-batching page/slot scheduling work?" →
  [lib-levanter-src-levanter-inference-jit_scheduler](concepts/lib-levanter-src-levanter-inference-jit_scheduler.md).
- "How does the prefill/decode generation loop work?" →
  [lib-levanter-src-levanter-inference-engine](concepts/lib-levanter-src-levanter-inference-engine.md).
- "How does a model architecture plug in (config, HF interop)?" →
  [lib-levanter-src-levanter-models-llama](concepts/lib-levanter-src-levanter-models-llama.md),
  [lib-levanter-src-levanter-models-olmo](concepts/lib-levanter-src-levanter-models-olmo.md),
  [lib-levanter-src-levanter-models-lm_model](concepts/lib-levanter-src-levanter-models-lm_model.md).
- "How does the training loop/optimizer/checkpoint work?" →
  [lib-levanter-src-levanter-trainer](concepts/lib-levanter-src-levanter-trainer.md),
  [lib-levanter-src-levanter-config](concepts/lib-levanter-src-levanter-config.md),
  [lib-levanter-src-levanter-checkpoint](concepts/lib-levanter-src-levanter-checkpoint.md).
- "How does DeepEP expert-parallel transport get built?" →
  [lib-levanter-src-levanter-kernels-deepep-transport_ffi](concepts/lib-levanter-src-levanter-kernels-deepep-transport_ffi.md).
- For the exhaustive per-symbol index, see `catalog/`; for the ranked concept list, see `index.md`.
