---
title: 'Module: python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.attention.hybrid_linear_attn_backend`/
symbols:
  attn_backend_wrapper: attn_backend_wrapper().
  LinearRecurrentAttnBackend.get_forward_metadata: LinearRecurrentAttnBackend#get_forward_metadata().
  HybridLinearAttnBackend: HybridLinearAttnBackend#
  HybridLinearAttnBackend.get_forward_metadata: HybridLinearAttnBackend#get_forward_metadata().
  LinearRecurrentAttnBackend: LinearRecurrentAttnBackend#
  LinearRecurrentAttnBackend.mesh: LinearRecurrentAttnBackend#mesh.
  LinearRecurrentAttnBackend.forward_metadata: LinearRecurrentAttnBackend#forward_metadata.
  HybridLinearAttnBackend.forward_metadata: HybridLinearAttnBackend#forward_metadata().
  HybridLinearAttnBackend.__call__: HybridLinearAttnBackend#__call__().
  LinearRecurrentAttnBackendMetadata.tree_flatten: LinearRecurrentAttnBackendMetadata#tree_flatten().
  LinearRecurrentAttnBackendMetadata.recurrent_indices: LinearRecurrentAttnBackendMetadata#recurrent_indices.
  LinearRecurrentAttnBackendMetadata.has_initial_state: LinearRecurrentAttnBackendMetadata#has_initial_state.
  HybridLinearAttnBackendMetadata.full_attn_metadata: HybridLinearAttnBackendMetadata#full_attn_metadata.
  HybridLinearAttnBackendMetadata.linear_attn_metadata: HybridLinearAttnBackendMetadata#linear_attn_metadata.
  LinearRecurrentAttnBackendMetadata.cu_q_lens: LinearRecurrentAttnBackendMetadata#cu_q_lens.
  LinearRecurrentAttnBackend.get_layer_cache: LinearRecurrentAttnBackend#get_layer_cache().
  HybridLinearAttnBackendMetadata.tree_flatten: HybridLinearAttnBackendMetadata#tree_flatten().
  HybridLinearAttnBackend.__init__: HybridLinearAttnBackend#__init__().
  HybridLinearAttnBackend.get_max_running_reqests: HybridLinearAttnBackend#get_max_running_reqests().
  HybridLinearAttnBackend._forward_metadata: HybridLinearAttnBackend#_forward_metadata.
  LinearRecurrentAttnBackendMetadata: LinearRecurrentAttnBackendMetadata#
  HybridLinearAttnBackend.full_attn_backend: HybridLinearAttnBackend#full_attn_backend.
  LinearRecurrentAttnBackend.__init__: LinearRecurrentAttnBackend#__init__().
  HybridLinearAttnBackendMetadata: HybridLinearAttnBackendMetadata#
  HybridLinearAttnBackend.linear_attn_backend: HybridLinearAttnBackend#linear_attn_backend.
  LinearRecurrentAttnBackend.tree_flatten: LinearRecurrentAttnBackend#tree_flatten().
  LinearRecurrentAttnBackend.tree_unflatten: LinearRecurrentAttnBackend#tree_unflatten().
  HybridLinearAttnBackend.full_attn_layers: HybridLinearAttnBackend#full_attn_layers.
  LinearRecurrentAttnBackendMetadata.tree_unflatten: LinearRecurrentAttnBackendMetadata#tree_unflatten().
  LinearRecurrentAttnBackend.get_max_running_reqests: LinearRecurrentAttnBackend#get_max_running_reqests().
  HybridLinearAttnBackendMetadata.tree_unflatten: HybridLinearAttnBackendMetadata#tree_unflatten().
---
# Module: [`python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py)

## Classes
### `HybridLinearAttnBackend`  ·  implements/extends AttentionBackend
- def: [`python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py:147`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L147)
- doc: Routes by layer.layer_id to a full or linear sub-backend.
- signature: `class HybridLinearAttnBackend(AttentionBackend):`
- members:
  - `__call__(self, q: jax.Array, k: jax.Array, v: jax.Array, layer, forward_batch: ForwardBatch, pool, mixed_qkv: jax.Array | None = None, a: jax.Array | None = None, b: jax.Array | None = None, **kwargs)` — [`L185`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L185) — Dispatch by layer.layer_id.
  - `forward_metadata(self)` — [`L176`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L176)
  - `get_forward_metadata(self, batch)` — [`L169`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L169)
  - `get_max_running_reqests(self, max_context_len, page_size)` — [`L227`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L227)
  - `full_attn_backend` — [`L162`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L162)
  - `full_attn_layers` — [`L166`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L166)
  - `linear_attn_backend` — [`L163`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L163)
- protocol/private: `__init__`[`L156`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L156), `_forward_metadata`[`L167`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L167)
- uses (calls/refs, reference-scoped): [`ForwardBatch`](../../model_executor/forward_batch_info.md#ForwardBatch), [`get_forward_metadata`](hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.get_forward_metadata), [`AttentionBackend`](base_attn_backend.md#AttentionBackend), [`get_forward_metadata`](base_attn_backend.md#AttentionBackend.get_forward_metadata), [`LinearRecurrentAttnBackend`](hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend), [`get_max_running_reqests`](base_attn_backend.md#AttentionBackend.get_max_running_reqests), [`forward_metadata`](hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.forward_metadata), [`full_attn_metadata`](hybrid_linear_attn_backend.md#HybridLinearAttnBackendMetadata.full_attn_metadata), [`linear_attn_metadata`](hybrid_linear_attn_backend.md#HybridLinearAttnBackendMetadata.linear_attn_metadata), [`HybridLinearAttnBackendMetadata`](hybrid_linear_attn_backend.md#HybridLinearAttnBackendMetadata)
- used by: [`padding_for_decode`](../../speculative/eagle_draft_worker.md#EagleDraftWorker.padding_for_decode), [`draft_extend_for_prefill`](../../speculative/multi_layer_draft_worker.md#MultiLayerDraftWorker.draft_extend_for_prefill), [`forward_batch_generation`](../../managers/tp_worker.md#ModelWorker.forward_batch_generation), [`draft_forward`](../../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_forward), [`draft_extend_for_prefill`](../../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_prefill), [`attn_backend_wrapper`](hybrid_linear_attn_backend.md#attn_backend_wrapper), [`AttentionBackend`](base_attn_backend.md#AttentionBackend), [`get_forward_metadata`](base_attn_backend.md#AttentionBackend.get_forward_metadata), [`get_max_running_reqests`](base_attn_backend.md#AttentionBackend.get_max_running_reqests), [`set_forward_metadata`](../../managers/tp_worker.md#ModelWorker.set_forward_metadata)

### `HybridLinearAttnBackendMetadata`
- def: [`python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py:128`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L128)
- doc: Aggregate metadata returned by HybridLinearAttnBackend.get_forward_metadata.
- signature: `class HybridLinearAttnBackendMetadata:`
- members:
  - `tree_flatten(self)` — [`L138`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L138)
  - `tree_unflatten(cls, aux_data, children)` — [`L142`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L142)
  - `full_attn_metadata` — [`L135`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L135)
  - `linear_attn_metadata` — [`L136`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L136)
- uses (calls/refs, reference-scoped): [`LinearRecurrentAttnBackendMetadata`](hybrid_linear_attn_backend.md#LinearRecurrentAttnBackendMetadata), [`AttentionBackendMetadata`](base_attn_backend.md#AttentionBackendMetadata)
- used by: [`HybridLinearAttnBackend`](hybrid_linear_attn_backend.md#HybridLinearAttnBackend), [`get_forward_metadata`](hybrid_linear_attn_backend.md#HybridLinearAttnBackend.get_forward_metadata), [`_forward_metadata`](hybrid_linear_attn_backend.md#HybridLinearAttnBackend._forward_metadata)

### `LinearRecurrentAttnBackend`  ·  implements/extends AttentionBackend
- def: [`python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py:51`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L51)
- doc: Base class for linear recurrent attention backends (KDA, GDN, Mamba2).
- signature: `class LinearRecurrentAttnBackend(AttentionBackend):`
- members:
  - `get_forward_metadata(self, batch: ModelWorkerBatch)` — [`L65`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L65) — Return the metadata for a forward pass.
  - `get_layer_cache(recurrent_state_pool, layer_id: int)` — [`L113`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L113) — Returns (recurrent_cache, conv_cache) for the given layer.
  - `get_max_running_reqests(max_context_len: int, page_size: int)` — [`L118`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L118)
  - `tree_flatten(self)` — [`L101`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L101)
  - `tree_unflatten(cls, aux_data, children)` — [`L107`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L107)
  - `forward_metadata` — [`L63`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L63)
  - `mesh` — [`L62`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L62)
- protocol/private: `__init__`[`L58`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L58)
- uses (calls/refs, reference-scoped): [`ForwardMode`](../../model_executor/forward_batch_info.md#ForwardMode), `ModelWorkerBatch`, `forward_mode`, `dp_size`, [`device_array`](../../utils/jax_utils.md#device_array), `per_dp_bs_size`, [`EXTEND`](../../model_executor/forward_batch_info.md#ForwardMode.EXTEND), [`AttentionBackend`](base_attn_backend.md#AttentionBackend), `extend_seq_lens`, [`DECODE`](../../model_executor/forward_batch_info.md#ForwardMode.DECODE), [`has_initial_state`](hybrid_linear_attn_backend.md#LinearRecurrentAttnBackendMetadata.has_initial_state), [`recurrent_indices`](hybrid_linear_attn_backend.md#LinearRecurrentAttnBackendMetadata.recurrent_indices), `recurrent_indices`, [`GDNAttnBackend`](linear/gdn_backend.md#GDNAttnBackend), [`KDAAttnBackend`](linear/kda_backend.md#KDAAttnBackend), [`LightningAttnBackend`](linear/lightning_backend.md#LightningAttnBackend), [`cu_q_lens`](hybrid_linear_attn_backend.md#LinearRecurrentAttnBackendMetadata.cu_q_lens), [`LinearRecurrentAttnBackendMetadata`](hybrid_linear_attn_backend.md#LinearRecurrentAttnBackendMetadata), `has_initial_state`
- used by: [`__call__`](linear/kda_backend.md#KDAAttnBackend.__call__), [`__call__`](linear/gdn_backend.md#GDNAttnBackend.__call__), [`__call__`](linear/lightning_backend.md#LightningAttnBackend.__call__), [`forward_extend`](linear/gdn_backend.md#GDNAttnBackend.forward_extend), [`forward_decode`](linear/gdn_backend.md#GDNAttnBackend.forward_decode), [`AttentionBackend`](base_attn_backend.md#AttentionBackend), [`HybridLinearAttnBackend`](hybrid_linear_attn_backend.md#HybridLinearAttnBackend), [`get_forward_metadata`](base_attn_backend.md#AttentionBackend.get_forward_metadata), [`_forward_extend`](linear/kda_backend.md#KDAAttnBackend._forward_extend), [`_forward_extend`](linear/lightning_backend.md#LightningAttnBackend._forward_extend), [`get_forward_metadata`](hybrid_linear_attn_backend.md#HybridLinearAttnBackend.get_forward_metadata), [`get_max_running_reqests`](base_attn_backend.md#AttentionBackend.get_max_running_reqests), [`_forward_decode`](linear/kda_backend.md#KDAAttnBackend._forward_decode), [`_forward_decode`](linear/lightning_backend.md#LightningAttnBackend._forward_decode), [`get_state`](linear/kda_backend.md#KDAAttnBackend.get_state), [`set_conv_state`](linear/kda_backend.md#KDAAttnBackend.set_conv_state), [`set_ssm_state`](linear/kda_backend.md#KDAAttnBackend.set_ssm_state), [`_short_conv_extend`](linear/kda_backend.md#KDAAttnBackend._short_conv_extend), [`GDNAttnBackend`](linear/gdn_backend.md#GDNAttnBackend), [`KDAAttnBackend`](linear/kda_backend.md#KDAAttnBackend), [`LightningAttnBackend`](linear/lightning_backend.md#LightningAttnBackend), [`__init__`](hybrid_linear_attn_backend.md#HybridLinearAttnBackend.__init__), [`__init__`](linear/lightning_backend.md#LightningAttnBackend.__init__), [`__init__`](linear/gdn_backend.md#GDNAttnBackend.__init__), [`__init__`](linear/kda_backend.md#KDAAttnBackend.__init__)

### `LinearRecurrentAttnBackendMetadata`
- def: [`python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py:31`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L31)
- signature: `class LinearRecurrentAttnBackendMetadata:`
- members:
  - `tree_flatten(self)` — [`L36`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L36)
  - `tree_unflatten(cls, aux_data, children)` — [`L42`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L42)
  - `cu_q_lens` — [`L32`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L32)
  - `has_initial_state` — [`L34`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L34)
  - `recurrent_indices` — [`L33`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L33)
- used by: [`__call__`](linear/kda_backend.md#KDAAttnBackend.__call__), [`get_forward_metadata`](hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.get_forward_metadata), [`__call__`](linear/lightning_backend.md#LightningAttnBackend.__call__), [`forward_extend`](linear/gdn_backend.md#GDNAttnBackend.forward_extend), [`forward_decode`](linear/gdn_backend.md#GDNAttnBackend.forward_decode), [`_forward_extend`](linear/lightning_backend.md#LightningAttnBackend._forward_extend), [`forward_metadata`](hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.forward_metadata), [`get_state`](linear/kda_backend.md#KDAAttnBackend.get_state), [`linear_attn_metadata`](hybrid_linear_attn_backend.md#HybridLinearAttnBackendMetadata.linear_attn_metadata)

## Functions
- `attn_backend_wrapper(runner: ModelRunner, full_attn_backend: AttentionBackend)` — [`L231`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/hybrid_linear_attn_backend.py#L231) — Wrap full_attn_backend in HybridLinearAttnBackend for hybrid models.

