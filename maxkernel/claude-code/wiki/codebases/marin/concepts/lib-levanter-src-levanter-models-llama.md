---
title: levanter.models.llama — LlamaConfig, HF interop, and decoder-layer assembly
type: concept
provenance: mixed
concept: lib-levanter-src-levanter-models-llama
updated: 2026-07-03
status: fresh
---
# levanter.models.llama — LlamaConfig, HF interop, and decoder-layer assembly

## Overview

[`LlamaConfig`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaConfig) is a
`LmConfig.register_subclass("llama")`-registered, `HFCompatConfig`-implementing frozen dataclass:
Llama's own architecture parameters plus two conversion methods
([`from_hf_config`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaConfig.from_hf_config)/
[`to_hf_config`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaConfig.to_hf_config)) that
make it bit-compatible with HuggingFace's `LlamaConfig`. It does not implement its own attention —
[`attention_config`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaConfig.attention_config)
converts `LlamaConfig` into the shared
[`AttentionConfig`](../catalog/lib/levanter/src/levanter/layers/attention.md) that levanter's one
generic `Attention` layer (see [root](root.md)) consumes — the same pattern
[`MistralConfig`](../catalog/lib/levanter/src/levanter/models/mistral.md#MistralConfig.attention_config)
follows.

## Diagram

```mermaid
flowchart TD
  HF["HuggingFace LlamaConfig dict"] -->|from_hf_config| LC["LlamaConfig\n(hidden_dim, num_heads, num_kv_heads,\nrope, norm_config, ...)"]
  LC -->|to_hf_config| HF
  LC -->|attention_config()| AC["AttentionConfig\n(shared by every architecture)"]
  LC -->|norm_config| LN["LayerNormConfigBase.build()"]
  LC -->|LlamaDecoderLayer.init| LDL["LlamaDecoderLayer\n(Attention + LlamaMlp + 2-4 LayerNorms)"]
  AC -->|Attention.init| ATTN["Attention (see root.md)"]
```

## Design rationale (why it's built this way)

**`attention_config` is a pure conversion function, not attention logic itself — architecture-specific
config translates into one shared `AttentionConfig` shape rather than each architecture reimplementing
attention.** [`LlamaConfig.attention_config`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaConfig.attention_config)'s
doc — "Convert this LlamaConfig to an AttentionConfig for use with Attention" — reads off
`num_heads`/`num_kv_heads`/`rope`/`qk_norm`/`attn_backend`/`flash_attention_block_size`/`upcast_attn`/`use_bias`
and packages them; the near-identical
[`MistralConfig.attention_config`](../catalog/lib/levanter/src/levanter/models/mistral.md#MistralConfig.attention_config)
does the same field mapping for Mistral, confirming this is a shared cross-architecture contract, not
a Llama-specific mechanism.

**HF round-tripping (`from_hf_config`/`to_hf_config`) is symmetric but not trivially invertible — both
directions independently re-derive `rope`/`activation_function` rather than caching the original HF
dict.** [`LlamaConfig.from_hf_config`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaConfig.from_hf_config)
builds `rope` via `RotaryEmbeddingsConfig.from_hf_config(hf_rope_config, ...)` and
`activation_function` via `ActivationFunctionEnum`; `to_hf_config` goes the other way,
re-deriving `hidden_dim`/`num_heads`/`num_kv_heads`/`num_layers`/`rope` into HF's dict shape via
`hf_config_from_kwargs` — round-tripping through levanter's own `LlamaConfig` in between rather than
storing the HF dict verbatim.

**Layer normalization is chosen polymorphically per-config via `norm_config`, not hardcoded to
RMSNorm**, even though Llama conventionally uses RMSNorm.
[`LlamaConfig.norm_config`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaConfig.norm_config)
returns a [`LayerNormConfigBase`](../catalog/lib/levanter/src/levanter/layers/normalization.md) whose
concrete `build()` is what actually picks the norm implementation — allowing a norm-flavor swap
(e.g. for ablations) without touching `LlamaDecoderLayer.init`'s own code.

## Entry points

- [`LlamaConfig.from_hf_config`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaConfig.from_hf_config) /
  [`to_hf_config`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaConfig.to_hf_config) —
  the HF-checkpoint-compatibility boundary; called on checkpoint load/save
  (see [lib-levanter-src-levanter-checkpoint](lib-levanter-src-levanter-checkpoint.md)-adjacent
  compat code, outside this packet's own subgraph).
- [`LlamaConfig.attention_config`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaConfig.attention_config) —
  called once per `LlamaDecoderLayer.init` to build the shared `AttentionConfig`.
- [`LlamaDecoderLayer.init`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaDecoderLayer.init) —
  the per-layer constructor; called once per decoder layer at model initialization.

## Mechanism (step-by-step)

1. **A HuggingFace config dict is converted into `LlamaConfig` via
   [`from_hf_config`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaConfig.from_hf_config)**,
   which resolves `activation_function` (an
   [`ActivationFunctionEnum`](../catalog/lib/levanter/src/levanter/utils/activation.md#ActivationFunctionEnum))
   and `rope` (a [`RotaryEmbeddingsConfig`](../catalog/lib/levanter/src/levanter/layers/rotary.md#RotaryEmbeddingsConfig),
   see [lib-levanter-src-levanter-layers-rotary](lib-levanter-src-levanter-layers-rotary.md)) alongside
   the plain scalar fields.
2. **[`LlamaDecoderLayer.init`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaDecoderLayer.init)
   splits its PRNG key and builds an `Attention` from
   `config.attention_config()`, an `LlamaMlp`, and two-to-four layer norms** (an extra pair via
   `config.hybrid_norm` for post-attention/post-MLP normalization, gated by that flag).
3. **[`LlamaConfig.to_hf_config`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaConfig.to_hf_config)
   reverses the process for checkpoint export**, re-deriving the HF
   dict's fields from `LlamaConfig`'s own fields rather than replaying a stored original dict.

## Key data structures

- **[`LlamaConfig`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaConfig)** — `hidden_dim`,
  `num_heads`, `num_kv_heads`, `num_layers`, `rope`, `norm_config`-adjacent fields,
  `activation_function`, `use_bias`, `hybrid_norm`, `attn_backend`, `flash_attention_block_size`,
  `upcast_attn`.
- **`LlamaDecoderLayer`** — one `Attention`, one `LlamaMlp`, and 2 or 4 layer-norm instances (depending
  on `hybrid_norm`), assembled by
  [`init`](../catalog/lib/levanter/src/levanter/models/llama.md#LlamaDecoderLayer.init).

## Dynamics (design intent)
Not addressable beyond the config-conversion/assembly pipeline described above from this packet's
subgraph.

## Edge cases
None directly visible in this packet's subgraph beyond the `hybrid_norm`-gated extra norm layers.

## Open questions
- Whether `to_hf_config`'s independent re-derivation (rather than caching the original HF dict) can
  silently drop HF-config fields levanter doesn't itself model isn't resolved by this packet's
  subgraph.

## See also
- [root](root.md) — the shared `Attention` layer `attention_config()` targets.
- [lib-levanter-src-levanter-layers-rotary](lib-levanter-src-levanter-layers-rotary.md) — `rope`'s
  concrete type.
- [lib-levanter-src-levanter-models-olmo](lib-levanter-src-levanter-models-olmo.md),
  [lib-levanter-src-levanter-models-gpt2](lib-levanter-src-levanter-models-gpt2.md) — sibling
  architectures following the same `attention_config`/`from_hf_config` pattern.
