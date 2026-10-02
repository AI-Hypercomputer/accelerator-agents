---
title: 'Module: tpu_inference/layers/vllm/backends/flash_attn.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/backends/flash_attn.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.backends.flash_attn`/
symbols:
  logger: logger.
  TPU_HEAD_SIZE_ALIGNMENT: TPU_HEAD_SIZE_ALIGNMENT.
  PallasAttentionMetadataBuilder: PallasAttentionMetadataBuilder#
  PallasAttentionMetadataBuilder.__init__: PallasAttentionMetadataBuilder#__init__().
  PallasAttentionMetadataBuilder.build: PallasAttentionMetadataBuilder#build().
  PallasAttentionBackend: PallasAttentionBackend#
  PallasAttentionBackend.get_name: PallasAttentionBackend#get_name().
  PallasAttentionBackend.get_builder_cls: PallasAttentionBackend#get_builder_cls().
  PallasAttentionBackend.get_impl_cls: PallasAttentionBackend#get_impl_cls().
  PallasAttentionBackend.supports_attn_type: PallasAttentionBackend#supports_attn_type().
  PallasAttentionBackend.get_kv_cache_shape: PallasAttentionBackend#get_kv_cache_shape().
  PallasAttentionBackend.swap_blocks: PallasAttentionBackend#swap_blocks().
  PallasAttentionBackend.get_min_page_size: PallasAttentionBackend#get_min_page_size().
  PallasAttentionBackend.get_max_num_seqs: PallasAttentionBackend#get_max_num_seqs().
  PallasAttentionBackend.get_page_size: PallasAttentionBackend#get_page_size().
  PallasAttentionBackendImpl: PallasAttentionBackendImpl#
  PallasAttentionBackendImpl.__init__: PallasAttentionBackendImpl#__init__().
  PallasAttentionBackendImpl.process_weights_after_loading: PallasAttentionBackendImpl#process_weights_after_loading().
  PallasAttentionBackendImpl.forward: PallasAttentionBackendImpl#forward().
  _prepare_qkv_layout: _prepare_qkv_layout().
  _format_attention_output: _format_attention_output().
  _jax_attn_func: _jax_attn_func().
  _jax_encoder_only_attn_func: _jax_encoder_only_attn_func().
---
# Module: [`tpu_inference/layers/vllm/backends/flash_attn.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py)

## Classes
### `PallasAttentionBackend`
- def: [`tpu_inference/layers/vllm/backends/flash_attn.py:65`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L65)
- signature: `class PallasAttentionBackend(AttentionBackend):`
- members:
  - `get_builder_cls()` — [`L72`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L72)
  - `get_impl_cls()` — [`L76`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L76)
  - `get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int, cache_dtype_str: str='auto')` — [`L87`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L87)
  - `get_max_num_seqs(model_len: int, page_size: int)` — [`L120`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L120)
  - `get_min_page_size(vllm_config: VllmConfig)` — [`L111`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L111)
  - `get_name()` — [`L68`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L68)
  - `get_page_size(vllm_config: VllmConfig)` — [`L129`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L129)
  - `supports_attn_type(cls, attn_type: str)` — [`L80`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L80)
  - `swap_blocks(src_kv_cache: torch.Tensor, dst_kv_cache: torch.Tensor, src_to_dst: torch.Tensor)` — [`L99`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L99)

### `PallasAttentionBackendImpl`
- def: [`tpu_inference/layers/vllm/backends/flash_attn.py:145`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L145)
- signature: `class PallasAttentionBackendImpl(AttentionImpl):`
- members:
  - `forward(self, layer: AttentionLayer, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor, attn_metadata: AttentionMetadata, output: Optional[torch.Tensor]=None, output_scale: Optional[torch.Tensor]=None, output_block_scale: Optional[torch.Tensor]=None)` — [`L198`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L198)
  - `process_weights_after_loading(self, act_dtype: torch.dtype)` — [`L191`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L191)
- protocol/private: `__init__`[`L147`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L147)

### `PallasAttentionMetadataBuilder`
- def: [`tpu_inference/layers/vllm/backends/flash_attn.py:36`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L36)
- signature: `class PallasAttentionMetadataBuilder(AttentionMetadataBuilder[AttentionMetadata]):`
- members:
  - `build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool=False)` — [`L48`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L48)
- protocol/private: `__init__`[`L39`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L39)

## Functions
- `_format_attention_output(outputs: jax.Array, q_len: int, num_heads: int, head_size: int, dtype: jnp.dtype)` — [`L309`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L309)
- `_jax_attn_func(kv_cache: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, sinks: jax.Array | None, attention_metadata: AttentionMetadata, mesh: Mesh, scale: float, head_size: int, num_heads: int, num_kv_heads: int, q_scale: float | None=None, k_scale: float | None=None, v_scale: float | None=None, sliding_window: int | None=None)` — [`L336`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L336)
- `_jax_encoder_only_attn_func(q: jax.Array, k: jax.Array, v: jax.Array, attention_metadata: AttentionMetadata, mesh: Mesh, scale: float, head_size: int, num_heads: int, num_kv_heads: int, sliding_window: int | None=None)` — [`L384`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L384)
- `_prepare_qkv_layout(q: jax.Array, k: jax.Array, v: jax.Array, num_heads: int, num_kv_heads: int, head_size: int)` — [`L292`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L292)

## Module values
- `TPU_HEAD_SIZE_ALIGNMENT` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L33)
- `logger` — [`L30`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn.py#L30)

