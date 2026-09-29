---
title: 'Module: tpu_inference/models/vllm/experimental/qwen3_vl_patcher.py'
type: catalog
provenance: extracted
module: tpu_inference/models/vllm/experimental/qwen3_vl_patcher.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.vllm.experimental.qwen3_vl_patcher`/
symbols:
  apply_qwen3_vl_patches: apply_qwen3_vl_patches().
  maybe_apply_qwen3_vl_patches: maybe_apply_qwen3_vl_patches().
  _patched_flatten_embeddings: _patched_flatten_embeddings().
  _patched_forward: _patched_forward().
  _patched_get_deepstack: _patched_get_deepstack().
  logger: logger.
  _patched_set_deepstack: _patched_set_deepstack().
  is_qwen3_vl: is_qwen3_vl().
  _convert_to_torchax_tensor: _convert_to_torchax_tensor().
  _patched_embed_input_ids: _patched_embed_input_ids().
---
# Module: [`tpu_inference/models/vllm/experimental/qwen3_vl_patcher.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_vl_patcher.py)

## Functions
- `_convert_to_torchax_tensor(v)` — [`L92`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_vl_patcher.py#L92) — Converts a PyTorch tensor to a Torchax tensor, ensuring JAX compatibility.
- `_patched_embed_input_ids(vllm_model, orig_embed_input_ids, *args, **kwargs)` — [`L131`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_vl_patcher.py#L131) — Appends Deepstack features to the main text embeddings.
- `_patched_flatten_embeddings(embeddings: NestedTensors)` — [`L222`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_vl_patcher.py#L222) — Patched version of vLLM's `_flatten_embeddings` to prevent JAX/Torchax ZeroDivisionError.
- `_patched_forward(vllm_model, orig_forward, input_ids, positions, intermediate_tensors, inputs_embeds=None, **kwargs)` — [`L173`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_vl_patcher.py#L173) — Unpacks vision features from combined embeddings and restores them to model state.
- `_patched_get_deepstack(vllm_model, orig_get_deepstack, num_tokens: int)` — [`L107`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_vl_patcher.py#L107) — Retrieves Deepstack embeddings, preferring JAX-compatible cached tensors.
- `_patched_set_deepstack(vllm_model, deepstack_input_embeds)` — [`L60`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_vl_patcher.py#L60) — Intercepts Deepstack embeddings to store them in a JAX-friendly cached tensors (`_deepstack_tensors`).
- `apply_qwen3_vl_patches(vllm_model)` — [`L263`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_vl_patcher.py#L263) — Apply Qwen3-VL specific patches for stateless Deepstack support.
- `is_qwen3_vl(vllm_model)` — [`L300`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_vl_patcher.py#L300) — Check if the given vLLM model is of architecture Qwen3VLForConditionalGeneration.
- `maybe_apply_qwen3_vl_patches(vllm_model: nn.Module)` — [`L305`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_vl_patcher.py#L305)

## Module values
- `logger` — [`L57`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/qwen3_vl_patcher.py#L57)

