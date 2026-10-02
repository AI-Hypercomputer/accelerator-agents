---
title: 'Module: tpu_inference/models/vllm/experimental/model_patcher.py'
type: catalog
provenance: extracted
module: tpu_inference/models/vllm/experimental/model_patcher.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.vllm.experimental.model_patcher`/
symbols:
  patch_mm_model: patch_mm_model().
  logger: logger.
  apply_model_specific_patches: apply_model_specific_patches().
  patch_mm_model._jax_compatible_merge_multimodal_embeddings: patch_mm_model()._jax_compatible_merge_multimodal_embeddings().
  patch_mm_model._flatten_model_output: patch_mm_model()._flatten_model_output().
  patch_mm_model._unflatten_model_output: patch_mm_model()._unflatten_model_output().
---
# Module: [`tpu_inference/models/vllm/experimental/model_patcher.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/model_patcher.py)

## Functions
- `_flatten_model_output(obj)` — [`L136`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/model_patcher.py#L136)
- `_jax_compatible_merge_multimodal_embeddings(inputs_embeds, multimodal_embeddings, is_multimodal)` — [`L85`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/model_patcher.py#L85)
- `_unflatten_model_output(aux, children, obj_type)` — [`L139`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/model_patcher.py#L139)
- `apply_model_specific_patches(vllm_model)` — [`L210`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/model_patcher.py#L210) — A consolidated entrypoint to apply model-specific JIT patches.
- `patch_mm_model(model: _VllmRunner, params_and_buffers: dict[str, torchax.torch.Tensor], *, jitted_mm_module_keys: Sequence[str], register_mm_module_custom_pytree_classes: Sequence[str])` — [`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/model_patcher.py#L44) — Jit some modules in the multimodal.

## Module values
- `logger` — [`L41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/model_patcher.py#L41)

