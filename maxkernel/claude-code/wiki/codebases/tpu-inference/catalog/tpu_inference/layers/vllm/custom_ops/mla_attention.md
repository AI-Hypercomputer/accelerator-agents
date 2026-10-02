---
title: 'Module: tpu_inference/layers/vllm/custom_ops/mla_attention.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/custom_ops/mla_attention.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.custom_ops.mla_attention`/
symbols:
  DummyMLAPrefillBackend: DummyMLAPrefillBackend#
  DummyMLAPrefillBackend.__init__: DummyMLAPrefillBackend#__init__().
  DummyMLAPrefillBackend.forward: DummyMLAPrefillBackend#forward().
  VllmMLAAttention: VllmMLAAttention#
  VllmMLAAttention.__init__: VllmMLAAttention#__init__().
  VllmMLAAttention.process_weights_after_loading: VllmMLAAttention#process_weights_after_loading().
  VllmMLAAttention.forward: VllmMLAAttention#forward().
  VllmMultiHeadLatentAttentionWrapper: VllmMultiHeadLatentAttentionWrapper#
  VllmMultiHeadLatentAttentionWrapper.__init__: VllmMultiHeadLatentAttentionWrapper#__init__().
  VllmMultiHeadLatentAttentionWrapper.forward: VllmMultiHeadLatentAttentionWrapper#forward().
---
# Module: [`tpu_inference/layers/vllm/custom_ops/mla_attention.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mla_attention.py)

## Classes
### `DummyMLAPrefillBackend`
- def: [`tpu_inference/layers/vllm/custom_ops/mla_attention.py:44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mla_attention.py#L44)
- signature: `class DummyMLAPrefillBackend:`
- members:
  - `forward(self, *args, **kwargs)` — [`L49`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mla_attention.py#L49)
- protocol/private: `__init__`[`L46`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mla_attention.py#L46)

### `VllmMLAAttention`
- def: [`tpu_inference/layers/vllm/custom_ops/mla_attention.py:53`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mla_attention.py#L53)
- signature: `class VllmMLAAttention(MLAAttention):`
- members:
  - `forward(self, q: tuple[torch.Tensor, torch.Tensor], kv_c_normed: torch.Tensor, k_pe: torch.Tensor, output: torch.Tensor | None=None, **kwargs)` — [`L164`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mla_attention.py#L164)
  - `process_weights_after_loading(self, act_dtype: torch.dtype)` — [`L101`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mla_attention.py#L101)
- protocol/private: `__init__`[`L55`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mla_attention.py#L55)

### `VllmMultiHeadLatentAttentionWrapper`
- def: [`tpu_inference/layers/vllm/custom_ops/mla_attention.py:207`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mla_attention.py#L207)
- signature: `class VllmMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):`
- members:
  - `forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, llama_4_scaling: torch.Tensor | None=None)` — [`L272`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mla_attention.py#L272)
- protocol/private: `__init__`[`L209`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mla_attention.py#L209)

