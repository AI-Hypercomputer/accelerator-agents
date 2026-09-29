---
title: 'Module: tpu_inference/layers/vllm/custom_ops/rope.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/custom_ops/rope.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.custom_ops.rope`/
symbols:
  rotate_gptj_tpu: rotate_gptj_tpu().
  VllmDeepseekScalingRotaryEmbedding: VllmDeepseekScalingRotaryEmbedding#
  VllmDeepseekScalingRotaryEmbedding.__init__: VllmDeepseekScalingRotaryEmbedding#__init__().
  VllmDeepseekScalingRotaryEmbedding.forward_native: VllmDeepseekScalingRotaryEmbedding#forward_native().
---
# Module: [`tpu_inference/layers/vllm/custom_ops/rope.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/rope.py)

## Classes
### `VllmDeepseekScalingRotaryEmbedding`
- def: [`tpu_inference/layers/vllm/custom_ops/rope.py:33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/rope.py#L33)
- doc: TPU-friendly modifications for efficient RoPE usage.
- signature: `class VllmDeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbedding):`
- members:
  - `forward_native(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor | None=None, offsets: torch.Tensor | None=None)` — [`L41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/rope.py#L41) — Intercepts PyTorch-native implementation with TPU friendlier RoPE rotation.
- protocol/private: `__init__`[`L38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/rope.py#L38)

## Functions
- `rotate_gptj_tpu(x: torch.Tensor)` — [`L20`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/rope.py#L20)

