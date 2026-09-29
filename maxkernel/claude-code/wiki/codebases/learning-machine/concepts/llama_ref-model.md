---
title: llama_ref/model — single-device Llama3 port (torch_xla2 baseline)
type: concept
provenance: mixed
concept: llama_ref-model
updated: 2026-07-03
status: fresh
---
# llama_ref/model — single-device Llama3 port (torch_xla2 baseline)
A stripped-down port of [Meta's reference Llama-3 implementation](llama_ref-model_original.md) with every `fairscale` model-parallel layer replaced by plain `nn.Linear`/`nn.Embedding`, the manual softmax replaced by `torch.nn.functional.scaled_dot_product_attention`, and the KV cache removed — the training-oriented baseline that [`llama_ref.run`](../catalog/llama_ref/run.md#main)'s driver loads under the `model_impl='orig'` option, and the direct ancestor both [model_with_scan](llama_ref-model_with_scan.md) and [model_with_collectives](llama_ref-model_with_collectives.md) are derived from.

## Entry points
- [`Attention.forward`](../catalog/llama_ref/model.md#Attention.forward) — projects through [`wq`](../catalog/llama_ref/model.md#Attention.wq)/[`wk`](../catalog/llama_ref/model.md#Attention.wk)/[`wv`](../catalog/llama_ref/model.md#Attention.wv), reshapes using [`head_dim`](../catalog/llama_ref/model.md#Attention.head_dim), applies rotary embeddings, repeats KV heads by [`n_rep`](../catalog/llama_ref/model.md#Attention.n_rep) via [`repeat_kv`](../catalog/llama_ref/model.md#repeat_kv), and calls fused SDPA — no cache read/write, since this port targets full-sequence training rather than incremental decode.
- [`main`](../catalog/llama_ref/run.md#main) (in `llama_ref.run`) — the top-level driver that instantiates this file's `Transformer` when invoked with `model_impl='orig'`, builds an `(fsdp, tp)` JAX device mesh sized from [`n_heads`](../catalog/llama_ref/model.md#ModelArgs.n_heads)/[`dim`](../catalog/llama_ref/model.md#ModelArgs.dim)-derived model configs (`"8B"`/`"70B"`/`"405B"` presets), and hands the constructed model to a training loop.

## Mechanism (step-by-step)
1. [`ModelArgs.__init__`](../catalog/llama_ref/model.md#ModelArgs.__init__) accepts arbitrary `**kwargs`, only overwriting attributes that already exist on the dataclass, then derives `n_kv_heads` from `n_heads` if unset and asserts `n_heads % n_kv_heads == 0` and `dim % n_heads == 0` — the same validation guard reused verbatim across all three JAX-facing model variants.
2. [`Attention.head_dim`](../catalog/llama_ref/model.md#Attention.head_dim) is fixed at construction to `dim / n_heads` (using module-level [`dim`](../catalog/llama_ref/model.md#ModelArgs.dim) and [`n_heads`](../catalog/llama_ref/model.md#ModelArgs.n_heads) from `ModelArgs`), and every projection weight — [`wq`](../catalog/llama_ref/model.md#Attention.wq), [`wk`](../catalog/llama_ref/model.md#Attention.wk), [`wv`](../catalog/llama_ref/model.md#Attention.wv), [`wo`](../catalog/llama_ref/model.md#Attention.wo) — is sized off it; unlike `model_original.py`'s fairscale layers, none of these are sharded at construction time — sharding here is applied externally (by `run.py`'s mesh + sharding-map machinery), not baked into the layer classes.
3. [`Attention.forward`](../catalog/llama_ref/model.md#Attention.forward) replaces the hand-rolled `matmul→mask→softmax→matmul` sequence from `model_original.py` with a single `torch.nn.functional.scaled_dot_product_attention(xq, keys, values, is_causal=(mask is not None))` call — this is also the exact call site that `run.py`'s `register_attention`/`custom_attention` override intercepts to substitute a JAX Pallas flash-attention kernel wrapped in `shard_map`, so this file's use of the *standard* SDPA op (rather than a bespoke attention function) is what makes that op-level substitution possible.
4. Each layer owns its own [`feed_forward`](../catalog/llama_ref/model.md#TransformerBlock.feed_forward) submodule instance, consistent with `Transformer.__init__` building a plain `torch.nn.ModuleList` of separate `TransformerBlock` objects (one Python object per layer, no stacking — visible in the surrounding source though not itself part of this packet's cited subgraph). This per-layer-object structure is exactly what [model_with_scan](llama_ref-model_with_scan.md) and [model_with_collectives](llama_ref-model_with_collectives.md) replace with a single `ScanLayer` wrapping one shared submodule plus a stacked-weights tensor, to get `jax.lax.scan` compilation reuse across layers.
5. [`Attention.wk`](../catalog/llama_ref/model.md#Attention.wk)/[`wv`](../catalog/llama_ref/model.md#Attention.wv) are sized by `n_kv_heads * head_dim` rather than `n_heads * head_dim` — grouped-query attention is present in this port (unlike a plain multi-head baseline) even though the KV cache and multi-device K/V handling of the original are gone.

## Key data structures
- [`ModelArgs.dim`](../catalog/llama_ref/model.md#ModelArgs.dim) / [`ModelArgs.n_heads`](../catalog/llama_ref/model.md#ModelArgs.n_heads) — the two config fields this packet's subgraph centers on; every derived per-layer dimension ([`head_dim`](../catalog/llama_ref/model.md#Attention.head_dim), the `wq`/`wk`/`wv`/`wo` shapes, [`TransformerBlock.dim`](../catalog/llama_ref/model.md#TransformerBlock.dim) is not itself in this packet but mirrors `ModelArgs.dim`) traces back to these two fields plus `n_kv_heads`.
- [`Transformer.tok_embeddings`](../catalog/llama_ref/model.md#Transformer.tok_embeddings) / [`Transformer.norm`](../catalog/llama_ref/model.md#Transformer.norm) / [`Transformer.output`](../catalog/llama_ref/model.md#Transformer.output) — plain `nn.Embedding`/`RMSNorm`/`nn.Linear`, replacing `model_original.py`'s `VocabParallelEmbedding`/`ColumnParallelLinear` — the sharding these previously provided automatically must now come from an externally-applied `NamedSharding`/`with_sharding_constraint`.

## Dynamics (design intent)
> [!inferred] `main`'s signature (`batch_size`, `model_type`, `tp`, `seqlen`, `model_impl`, `use_custom_mesh`, `use_custom_offload`, `internal_override_layers`, `unroll_layers`) shows this file is one of three interchangeable `Transformer` implementations selected by `model_impl` — `run.py`'s own subgraph shows `model_impl='orig'` maps to this file, `'scan'` to [model_with_scan](llama_ref-model_with_scan.md), `'scan_manual'` to [model_with_collectives](llama_ref-model_with_collectives.md) — all three sharing this file's `ModelArgs`/`RMSNorm`/rotary-embedding helpers (this file is the one whose helpers the sibling files still import structurally, if not literally).

## Edge cases
- [`ModelArgs.n_kv_heads`](../catalog/llama_ref/model.md#ModelArgs.n_kv_heads) defaults to `None` and is only resolved to a concrete value inside [`__init__`](../catalog/llama_ref/model.md#ModelArgs.__init__) — code reading `n_kv_heads` before construction completes (e.g. a dataclass field default elsewhere) would see `None`, not the resolved head count.

## Open questions
> [!inferred] Whether removing the KV cache means this file is training-only (no incremental decode/serving path at all), or whether decode is handled by re-running the full forward pass per token, isn't resolvable from this packet's cited subgraph — the surrounding `run.py`/`train.py` orchestration (partially visible) is oriented entirely around a training loop, consistent with the former.

## See also
- [llama_ref/model_original](llama_ref-model_original.md) — the fairscale-parallel, KV-cached ancestor this file strips down from.
- [llama_ref/model_with_scan](llama_ref-model_with_scan.md) — adds `jax.lax.scan`-over-layers and `with_sharding_constraint` GSPMD hints on top of this file's `Attention`/`TransformerBlock`.
- [llama_ref/model_with_collectives](llama_ref-model_with_collectives.md) — adds explicit JAX `all_gather`/`psum` collectives and per-layer FSDP weight-gathering inside the scan body.
