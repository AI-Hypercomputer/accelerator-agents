---
title: 'Module: tpu_inference/layers/vllm/custom_ops/mhc.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/custom_ops/mhc.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.custom_ops.mhc`/
symbols:
  logger: logger.
  VllmMHCPreOp: VllmMHCPreOp#
  VllmMHCPreOp.enabled: VllmMHCPreOp#enabled().
  VllmMHCPreOp.forward_tpu: VllmMHCPreOp#forward_tpu().
  VllmMHCPostOp: VllmMHCPostOp#
  VllmMHCPostOp.enabled: VllmMHCPostOp#enabled().
  VllmMHCPostOp.forward_tpu: VllmMHCPostOp#forward_tpu().
  VllmHCHeadOp: VllmHCHeadOp#
  VllmHCHeadOp.enabled: VllmHCHeadOp#enabled().
  VllmHCHeadOp.forward_tpu: VllmHCHeadOp#forward_tpu().
  VllmMHCFusedPostPreOp: VllmMHCFusedPostPreOp#
  VllmMHCFusedPostPreOp.enabled: VllmMHCFusedPostPreOp#enabled().
  VllmMHCFusedPostPreOp.forward_tpu: VllmMHCFusedPostPreOp#forward_tpu().
---
# Module: [`tpu_inference/layers/vllm/custom_ops/mhc.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mhc.py)

## Classes
### `VllmHCHeadOp`
- def: [`tpu_inference/layers/vllm/custom_ops/mhc.py:84`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mhc.py#L84)
- doc: TPU implementation of HCHeadOp.
- signature: `class VllmHCHeadOp(HCHeadOp):`
- members:
  - `enabled(cls)` — [`L88`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mhc.py#L88) — Returns whether this operation is enabled.
  - `forward_tpu(self, hidden_states: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, rms_norm_eps: float, hc_eps: float)` — [`L92`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mhc.py#L92) — Applies the TPU forward pass for the op.

### `VllmMHCFusedPostPreOp`
- def: [`tpu_inference/layers/vllm/custom_ops/mhc.py:123`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mhc.py#L123)
- signature: `class VllmMHCFusedPostPreOp(MHCFusedPostPreOp):`
- members:
  - `enabled(cls)` — [`L126`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mhc.py#L126)
  - `forward_tpu(self, *args, **kwargs)` — [`L129`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mhc.py#L129)

### `VllmMHCPostOp`
- def: [`tpu_inference/layers/vllm/custom_ops/mhc.py:62`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mhc.py#L62)
- signature: `class VllmMHCPostOp(MHCPostOp):`
- members:
  - `enabled(cls)` — [`L65`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mhc.py#L65)
  - `forward_tpu(self, x: torch.Tensor, residual: torch.Tensor, post_layer_mix: torch.Tensor, comb_res_mix: torch.Tensor)` — [`L68`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mhc.py#L68)

### `VllmMHCPreOp`
- def: [`tpu_inference/layers/vllm/custom_ops/mhc.py:27`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mhc.py#L27)
- signature: `class VllmMHCPreOp(MHCPreOp):`
- members:
  - `enabled(cls)` — [`L30`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mhc.py#L30)
  - `forward_tpu(self, residual: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float, hc_post_mult_value: float, sinkhorn_repeat: int, n_splits: int=1, norm_weight: torch.Tensor | None=None, norm_eps: float=0.0)` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mhc.py#L33)

## Module values
- `logger` — [`L23`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/custom_ops/mhc.py#L23)

