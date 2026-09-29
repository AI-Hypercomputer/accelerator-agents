---
title: axlearn — overview
type: overview
updated: 2026-07-03
---
# axlearn — what it is and how it fits together

## In one paragraph

AXLearn is a JAX training/modeling framework (Google's large-scale LLM/multi-modal training library)
built on a `Configurable`/`Module` config-tree pattern: every layer is instantiated from a `Config` via
`instantiate()`, and forward computation is invoked purely-functionally (`functional`) so it composes
with JAX transforms. This catalog covers `axlearn/common/`'s perf-relevant core: the
`MultiheadAttention` layer's training/decode-unifying `ForwardMode` dispatch, the shared
`BaseAttentionBias` mask representation, the `FlashAttention` kernel-backed attention layer and its
`BaseFlashAttention` capability-negotiation contract, the `BaseKVCache`/`KVState` cache abstraction,
and `SpmdTrainer`'s compiled train-step loop. The much larger `cloud/`, `experiments/`, `vision/`,
`audio/` subtrees (deployment infra, published-experiment configs, and domain-specific model
architectures) are out of scope for this perf-focused catalog.

## Core architecture

```mermaid
flowchart TD
  CFG["Configurable/Module config-tree\n(default_config -> instantiate)"] --> MHA["MultiheadAttention\n(ForwardMode: FORWARD/INIT_STATES/EXTEND_STEP)"]
  MHA -->|_mask_tpl| BIAS["BaseAttentionBias / CausalAttentionBias\n(implicit boolean mask)"]
  MHA -->|_compute_attention override| FA["FlashAttention (GroupedQueryAttention)"]
  FA -->|is_supported| BFA["BaseFlashAttention capability check\n(TPU Pallas / GPU cuDNN backends)"]
  MHA -->|kv_state: KVState| KV["BaseKVCache / KVCache / PagedKVCache"]
  TRAINER["SpmdTrainer.run"] -->|functional(module, ...)| MHA
  TRAINER -->|_get_compiled_train_step_fn| COMPILED["compiled train step (reused every step)"]
```

## Main concepts

**One attention layer, one `ForwardMode`-gated method, handles training and both decode phases.**
`MultiheadAttention._forward_for_mode` dispatches on `ForwardMode` (training forward / decode-init /
decode-step) rather than three separate methods. See [axlearn-common-attention](concepts/axlearn-common-attention.md).

**Masks are implicit boolean functions, not materialized arrays.** `BaseAttentionBias`/
`MaskFnAttentionBias`/`CausalAttentionBias` are shared across every attention backend (base, Flash,
ring-attention). See [axlearn-common-attention_bias](concepts/axlearn-common-attention_bias.md).

**Flash Attention backends negotiate capability explicitly before being called.**
`BaseFlashAttention.is_supported` lets a dispatcher check shape/dtype/cache-type compatibility ahead
of time rather than catching a runtime failure. See
[axlearn-common-flash_attention-common](concepts/axlearn-common-flash_attention-common.md).

**`FlashAttention` overrides exactly one method of the base attention layer.** Everything else
(projections, masking, `ForwardMode` dispatch) is inherited unchanged; only `_compute_attention`
routes through a kernel backend. See
[axlearn-common-flash_attention-layer](concepts/axlearn-common-flash_attention-layer.md).

**KV cache state and cache management are separated.** `KVState` is a plain `(k_proj, v_proj)`
`NamedTuple` every attention backend consumes identically, regardless of which `BaseKVCache`
implementation (plain buffer vs. paged) produced it. See
[axlearn-common-kv_cache-base_kv_cache](concepts/axlearn-common-kv_cache-base_kv_cache.md).

**The training loop compiles once and threads state functionally.** `SpmdTrainer` builds its compiled
step function once and reuses it; every `Module` invocation inside that compiled step goes through
`functional`, threading state/PRNG key explicitly rather than via captured Python state. See
[axlearn-common-trainer](concepts/axlearn-common-trainer.md).

## How a request flows

**Training:** `SpmdTrainer.run` iterates batches, calling a once-compiled step function; internally,
`functional` invokes the model's forward method against the current `TrainerState`, which calls into
`MultiheadAttention`/`FlashAttention` layers (`ForwardMode.FORWARD`) using `BaseAttentionBias` masks.

**Serving/decode:** the same attention layers are called with `ForwardMode.INIT_STATES` then
repeated `ForwardMode.EXTEND_STEP` calls, reading/writing `KVState` via a `BaseKVCache` implementation
each step.

## Map of the wiki

- "How does one attention layer handle both training and decode?" →
  [axlearn-common-attention](concepts/axlearn-common-attention.md).
- "How are attention masks represented?" →
  [axlearn-common-attention_bias](concepts/axlearn-common-attention_bias.md).
- "How does Flash Attention backend selection/capability-checking work?" →
  [axlearn-common-flash_attention-common](concepts/axlearn-common-flash_attention-common.md).
- "How does the FlashAttention layer integrate with the base attention layer?" →
  [axlearn-common-flash_attention-layer](concepts/axlearn-common-flash_attention-layer.md).
- "How is the KV cache abstracted?" →
  [axlearn-common-kv_cache-base_kv_cache](concepts/axlearn-common-kv_cache-base_kv_cache.md).
- "How does the compiled training loop work?" →
  [axlearn-common-trainer](concepts/axlearn-common-trainer.md).
- For the exhaustive per-symbol index, see `catalog/`; for the ranked concept list, see `index.md`.
