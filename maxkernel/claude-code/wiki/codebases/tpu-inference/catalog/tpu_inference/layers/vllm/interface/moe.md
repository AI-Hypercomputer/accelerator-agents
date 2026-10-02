---
title: 'Module: tpu_inference/layers/vllm/interface/moe.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/vllm/interface/moe.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.vllm.interface.moe`/
symbols:
  vllm_moe_apply: vllm_moe_apply().
  select_moe_backend_from_fused_moe_config: select_moe_backend_from_fused_moe_config().
  logger: logger.
---
# Module: [`tpu_inference/layers/vllm/interface/moe.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/interface/moe.py)

## Functions
- `select_moe_backend_from_fused_moe_config(moe: FusedMoEConfig)` — [`L31`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/interface/moe.py#L31) — Select the MoE backend based on the FusedMoEConfig.
- `vllm_moe_apply(layer: RoutedExperts, weights: FusedMoEWeights, quant_method_instance: FusedMoEMethodBase, x: torch.Tensor, router_logits: torch.Tensor, input_ids: torch.Tensor | None = None)` — [`L63`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/interface/moe.py#L63) — Shared function for applying a FusedMoE layer for the TorchAX/vLLM backend.

## Module values
- `logger` — [`L28`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/vllm/interface/moe.py#L28)

