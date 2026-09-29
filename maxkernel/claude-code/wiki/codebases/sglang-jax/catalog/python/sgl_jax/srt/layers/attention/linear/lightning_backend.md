---
title: 'Module: python/sgl_jax/srt/layers/attention/linear/lightning_backend.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/attention/linear/lightning_backend.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.attention.linear.lightning_backend`/
symbols:
  LightningAttnBackend.__call__: LightningAttnBackend#__call__().
  LightningAttnBackend._forward_extend: LightningAttnBackend#_forward_extend().
  LightningAttnBackend._forward_decode: LightningAttnBackend#_forward_decode().
  logger: logger.
  LightningAttnBackend.tp_slope: LightningAttnBackend#tp_slope.
  _build_alibi_base_slopes: _build_alibi_base_slopes().
  LightningAttnBackend: LightningAttnBackend#
  LightningAttnBackend.__init__: LightningAttnBackend#__init__().
  _compute_layer_slope: _compute_layer_slope().
  LightningAttnBackend._decode_fn: LightningAttnBackend#_decode_fn().
  LightningAttnBackend._prefill_fn: LightningAttnBackend#_prefill_fn().
  _build_alibi_base_slopes.get_slopes_power_of_2: _build_alibi_base_slopes().get_slopes_power_of_2().
  _CHUNK_SIZE: _CHUNK_SIZE.
  LightningAttnBackend.chunk_size: LightningAttnBackend#chunk_size.
  __all__: __all__.
---
# Module: [`python/sgl_jax/srt/layers/attention/linear/lightning_backend.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py)

## Classes
### `LightningAttnBackend`  ·  implements/extends LinearRecurrentAttnBackend
- def: [`python/sgl_jax/srt/layers/attention/linear/lightning_backend.py:86`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py#L86)
- doc: Attention backend for GLA (Gated Linear Attention) used by BailingMoeV2.5.
- signature: `class LightningAttnBackend(LinearRecurrentAttnBackend):`
- members:
  - `__init__(self, mesh: jax.sharding.Mesh = None, chunk_size: int = _CHUNK_SIZE, linear_recurrent_layer_ids: list[int] | None = None, num_hidden_layers: int | None = None, num_heads: int | None = None)` — [`L94`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py#L94) — Construct a LightningAttnBackend.
  - `_forward_decode(self, q: jax.Array, k: jax.Array, v: jax.Array, recurrent_buffer: jax.Array, recurrent_indices: jax.Array, has_initial_state: jax.Array, slope: jnp.ndarray)` — [`L182`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py#L182) — Decode forward via fused Pallas kernel with in-kernel state DMA.
  - `_forward_extend(self, q: jax.Array, k: jax.Array, v: jax.Array, recurrent_buffer: jax.Array, recurrent_indices: jax.Array, has_initial_state: jax.Array, slope: jnp.ndarray)` — [`L228`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py#L228) — Extend forward via baseline simple_gla_fwd + JAX gather/scatter.
  - `chunk_size` — [`L116`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py#L116)
  - `tp_slope` — [`L122`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py#L122)
- protocol/private: `__call__`[`L131`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py#L131), `_decode_fn`[`L196`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py#L196), `_prefill_fn`[`L245`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py#L245)
- uses (calls/refs, reference-scoped): [`ForwardMode`](../../../model_executor/forward_batch_info.md#ForwardMode), [`ForwardBatch`](../../../model_executor/forward_batch_info.md#ForwardBatch), [`named_scope`](../../../utils/profiling_utils.md#named_scope), [`EXTEND`](../../../model_executor/forward_batch_info.md#ForwardMode.EXTEND), [`forward_mode`](../../../model_executor/forward_batch_info.md#ForwardBatch.forward_mode), [`DECODE`](../../../model_executor/forward_batch_info.md#ForwardMode.DECODE), [`LinearRecurrentAttnBackend`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend), [`mesh`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.mesh), [`forward_metadata`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.forward_metadata), [`simple_gla_fwd`](../../../kernels/simple_gla/simple_gla.md#simple_gla_fwd), [`decode_simple_gla_fused`](../../../kernels/simple_gla/simple_gla_fused.md#decode_simple_gla_fused), [`has_initial_state`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackendMetadata.has_initial_state), [`recurrent_indices`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackendMetadata.recurrent_indices), [`cu_q_lens`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackendMetadata.cu_q_lens), [`get_layer_cache`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.get_layer_cache), [`_compute_layer_slope`](lightning_backend.md#_compute_layer_slope), [`__init__`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend.__init__), [`layer_id`](../../radix_lightning_attention.md#RadixLightningAttention.layer_id), [`RadixLightningAttention`](../../radix_lightning_attention.md#RadixLightningAttention), [`_CHUNK_SIZE`](lightning_backend.md#_CHUNK_SIZE)
- used by: [`attn_backend_wrapper`](../hybrid_linear_attn_backend.md#attn_backend_wrapper), [`LinearRecurrentAttnBackend`](../hybrid_linear_attn_backend.md#LinearRecurrentAttnBackend)

## Functions
- `_build_alibi_base_slopes(num_heads: int)` — [`L49`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py#L49) — ALiBi base slopes matching the HF BailingMoeV2.5 reference.
- `_compute_layer_slope(layer_id: int, num_hidden_layers: int, num_heads: int, mesh: jax.sharding.Mesh | None = None)` — [`L66`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py#L66) — Per-layer slope decay used as ``g_gamma`` by the simple_gla kernels.
- `get_slopes_power_of_2(n: int)` — [`L52`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py#L52)

## Module values
- `_CHUNK_SIZE` — [`L46`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py#L46)
- `__all__` — [`L288`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py#L288)
- `logger` — [`L30`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/lightning_backend.py#L30)

