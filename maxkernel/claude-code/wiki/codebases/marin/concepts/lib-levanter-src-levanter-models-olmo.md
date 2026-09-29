---
title: levanter.models.olmo — Olmo2Config, the same HF-interop pattern as Llama
type: concept
provenance: mixed
concept: lib-levanter-src-levanter-models-olmo
updated: 2026-07-03
status: fresh
---
# levanter.models.olmo — Olmo2Config, the same HF-interop pattern as Llama

## Overview

[`Olmo2Config`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2Config) follows the identical
structural pattern as
[`LlamaConfig`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaConfig) (see
[lib-levanter-src-levanter-models-llama](lib-levanter-src-levanter-models-llama.md)) —
[`attention_config`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2Config.attention_config)
converts to the shared `AttentionConfig`,
[`from_hf_config`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2Config.from_hf_config)/
[`to_hf_config`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2Config.to_hf_config) round-trip
with HuggingFace — but constructs its own
[`Olmo2Attention`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2Attention.init) and
[`Olmo2DecoderLayer`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2DecoderLayer.init) types
rather than reusing Llama's, reflecting OLMo 2's own norm-placement conventions.

## Diagram

```mermaid
flowchart TD
  HF["HF Olmo2Config dict"] -->|from_hf_config| OC["Olmo2Config\n(hidden_dim, intermediate_dim, num_heads,\nnum_kv_heads, rope, tie_word_embeddings)"]
  OC -->|to_hf_config| HF
  OC -->|attention_config()| AC["AttentionConfig (shared, see root.md)"]
  OC -->|Olmo2Attention.init| OA["Olmo2Attention"]
  OC -->|Olmo2DecoderLayer.init| ODL["Olmo2DecoderLayer\n(Olmo2Attention + MLP + mk_LayerNorm)"]
```

## Design rationale (why it's built this way)

**`Olmo2Config.attention_config` shares the exact same field-mapping shape as `LlamaConfig`'s
version** (`num_heads`/`num_kv_heads`/`qk_norm`/`rope`/`attn_backend`/`flash_attention_block_size`/
`upcast_attn`/`use_bias`), confirming `AttentionConfig` is genuinely architecture-agnostic — the
difference between OLMo 2 and Llama attention lives entirely in
[`Olmo2Attention.init`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2Attention.init) (which
builds its own `Olmo2Attention` instance) and in `Olmo2DecoderLayer`'s own norm wiring, not in the
shared config shape.

**`Olmo2Config` additionally tracks `tie_word_embeddings` and `intermediate_dim` explicitly through
both HF-conversion directions**, per
[`from_hf_config`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2Config.from_hf_config)
and
[`to_hf_config`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2Config.to_hf_config) both
explicitly reading/writing these fields (`LlamaConfig`'s own conversion methods, per the sibling
packet, do not cite these two fields) — a genuine per-architecture difference in what needs
round-tripping.

## Entry points

- [`Olmo2Config.from_hf_config`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2Config.from_hf_config) /
  [`to_hf_config`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2Config.to_hf_config) — the
  HF-checkpoint-compatibility boundary.
- [`Olmo2Config.attention_config`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2Config.attention_config) —
  called once per `Olmo2Attention.init`/`Olmo2DecoderLayer.init`.
- [`Olmo2DecoderLayer.init`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2DecoderLayer.init) —
  the per-layer constructor.

## Mechanism (step-by-step)

1. **[`from_hf_config`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2Config.from_hf_config)
   resolves `activation_function`, `rope`, `tie_word_embeddings`, and
   `intermediate_dim`** from the HF dict, alongside the scalar fields shared with every other
   architecture.
2. **[`Olmo2Attention.init`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2Attention.init)
   builds an attention instance from `config.attention_config()`**, reading
   `HeadSize`/`Heads`/`KVHeads`/`QHeadsPerGroup` off the resulting `AttentionConfig`.
3. **[`Olmo2DecoderLayer.init`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2DecoderLayer.init)
   assembles one `Olmo2Attention` plus an MLP and layer norms via
   `mk_LayerNorm`**, mirroring `LlamaDecoderLayer.init`'s structure but with OLMo 2's own attention
   type.
4. **[`to_hf_config`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2Config.to_hf_config)
   reverses the process**, re-deriving the HF dict's fields (including
   `tie_word_embeddings`) from `Olmo2Config`'s own fields.

## Key data structures

- **[`Olmo2Config`](../catalog/lib/levanter/src/levanter/models/olmo.md#Olmo2Config)** — `hidden_dim`,
  `intermediate_dim`, `num_heads`, `num_kv_heads`, `num_layers`, `rope`, `tie_word_embeddings`,
  `activation_function`, `use_bias`, `attn_backend`, `flash_attention_block_size`, `upcast_attn`.
- **`Olmo2Attention`**/**`Olmo2DecoderLayer`** — OLMo-2-specific attention and decoder-layer types,
  structurally parallel to Llama's but distinct classes.

## Dynamics (design intent)
Not addressable beyond the config-conversion/assembly pipeline from this packet's subgraph.

## Edge cases
None directly visible beyond the `tie_word_embeddings` round-trip requirement.

## Open questions
- What OLMo 2's specific norm-placement convention (pre-norm vs. post-norm per sub-layer) actually is
  isn't resolved by the symbols in this packet's subgraph — only that `mk_LayerNorm` is called from
  `Olmo2DecoderLayer.init`, not the placement details.

## See also
- [lib-levanter-src-levanter-models-llama](lib-levanter-src-levanter-models-llama.md) — the sibling
  architecture sharing the same `attention_config`/`from_hf_config`/`to_hf_config` pattern.
- [root](root.md) — the shared `Attention`/`AttentionConfig` both architectures target.
