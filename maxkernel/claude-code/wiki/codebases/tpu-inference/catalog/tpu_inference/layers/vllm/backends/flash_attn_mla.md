---
title: 'Module: tpu_inference/layers/vllm/backends/flash_attn_mla.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/backends/flash_attn_mla.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.backends.flash_attn_mla`/PallasMLAttentionBackend
symbols:
  PallasMLAttentionBackend: '#'
  PallasMLAttentionBackend.get_name: '#get_name().'
  PallasMLAttentionBackend.get_impl_cls: '#get_impl_cls().'
  PallasMLAttentionBackend.get_page_size: '#get_page_size().'
  PallasMLAttentionBackendImpl: Impl#
  PallasMLAttentionBackendImpl.__init__: Impl#__init__().
  PallasMLAttentionBackendImpl.forward_mha: Impl#forward_mha().
  PallasMLAttentionBackendImpl.forward_mqa: Impl#forward_mqa().
  PallasMLAttentionBackendImpl.forward: Impl#forward().
---
# Module: [`tpu_inference/layers/vllm/backends/flash_attn_mla.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn_mla.py)

## Classes
### `PallasMLAttentionBackend`
- def: [`tpu_inference/layers/vllm/backends/flash_attn_mla.py:35`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn_mla.py#L35)
- signature: `class PallasMLAttentionBackend(AttentionBackend):`
- members:
  - `get_impl_cls()` — [`L42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn_mla.py#L42)
  - `get_name()` — [`L38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn_mla.py#L38)
  - `get_page_size(vllm_config: VllmConfig)` — [`L46`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn_mla.py#L46)

### `PallasMLAttentionBackendImpl`
- def: [`tpu_inference/layers/vllm/backends/flash_attn_mla.py:50`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn_mla.py#L50)
- signature: `class PallasMLAttentionBackendImpl(MLAAttentionImpl):`
- members:
  - `forward(self, q: tuple[torch.Tensor, torch.Tensor], kv_c_normed: torch.Tensor, k_pe: torch.Tensor, kv_cache: jnp.ndarray, attn_metadata: AttentionMetadata, mesh: Mesh, layer: MLAAttention, output: torch.Tensor | None=None, **kwargs)` — [`L112`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn_mla.py#L112) — Runs the fundamental MLA forward pass.
  - `forward_mha(self, q: torch.Tensor, kv_c_normed: torch.Tensor, k_pe: torch.Tensor, kv_c_and_k_pe_cache: torch.Tensor, attn_metadata: AttentionMetadata, k_scale: torch.Tensor, output: torch.Tensor)` — [`L85`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn_mla.py#L85) — Needed because this is abstract in the base class but we don't use it (instead, favoring a single `forward`).
  - `forward_mqa(self, q: torch.Tensor | tuple[torch.Tensor, torch.Tensor], kv_c_and_k_pe_cache: torch.Tensor, attn_metadata: AttentionMetadata, layer: AttentionLayer)` — [`L100`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn_mla.py#L100) — Needed because this is abstract in the base class but we don't use it (instead, favoring a single `forward`).
- protocol/private: `__init__`[`L52`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/backends/flash_attn_mla.py#L52)

