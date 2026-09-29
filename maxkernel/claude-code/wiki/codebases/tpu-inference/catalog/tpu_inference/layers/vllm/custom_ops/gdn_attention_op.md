---
title: 'Module: tpu_inference/layers/vllm/custom_ops/gdn_attention_op.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/custom_ops/gdn_attention_op.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.custom_ops.gdn_attention_op`/
symbols:
  logger: logger.
  gdn_attention_core_tpu: gdn_attention_core_tpu().
  VllmGatedDeltaNetAttention: VllmGatedDeltaNetAttention#
  VllmGatedDeltaNetAttention.forward: VllmGatedDeltaNetAttention#forward().
---
# Module: [`tpu_inference/layers/vllm/custom_ops/gdn_attention_op.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/gdn_attention_op.py)

## Classes
### `VllmGatedDeltaNetAttention`
- def: [`tpu_inference/layers/vllm/custom_ops/gdn_attention_op.py:174`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/gdn_attention_op.py#L174)
- signature: `class VllmGatedDeltaNetAttention(QwenGatedDeltaNetAttention):`
- members:
  - `forward(self, hidden_states: torch.Tensor, output: torch.Tensor)` — [`L176`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/gdn_attention_op.py#L176) — Implements the exact same logic as in vLLM (https://github.com/vllm-project/vllm/blob/9c81f35/vllm/model_executor/layers/mamba/gdn_linear_attn.py#L508)

## Functions
- `gdn_attention_core_tpu(mixed_qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor, core_attn_out: torch.Tensor, layer_name: str, mesh: jax.sharding.Mesh)` — [`L38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/gdn_attention_op.py#L38) — This acts as main bridge between PyTorch and JAX for the GDN core attention.

## Module values
- `logger` — [`L35`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/gdn_attention_op.py#L35)

