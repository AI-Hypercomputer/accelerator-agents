---
title: 'Module: tpu_inference/models/vllm/experimental/vision_tower_jit.py'
type: catalog
provenance: extracted
module: tpu_inference/models/vllm/experimental/vision_tower_jit.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.vllm.experimental.vision_tower_jit`/
symbols:
  maybe_precompile_vision_encoder_fn: maybe_precompile_vision_encoder_fn().
  is_jittable_architecture: is_jittable_architecture().
  has_jittable_vision: has_jittable_vision().
  maybe_prepare_for_jit: maybe_prepare_for_jit().
  get_vision_config: get_vision_config().
  logger: logger.
  maybe_jit_embed_multimodal_func: maybe_jit_embed_multimodal_func().
  GridTHW._nested_to_tuple: GridTHW#_nested_to_tuple().
  maybe_precompile_vision_encoder_fn.precompile_fn: maybe_precompile_vision_encoder_fn().precompile_fn().
  GridTHW.__new__: GridTHW#__new__().
  GridTHW: GridTHW#
  JITTABLE_ARCHS: JITTABLE_ARCHS.
  GridTHW.ndim: GridTHW#ndim().
  GridTHW.shape: GridTHW#shape().
  GridTHW.tolist: GridTHW#tolist().
  GridTHW.prod: GridTHW#prod().
  GridTHW.__repr__: GridTHW#__repr__().
---
# Module: [`tpu_inference/models/vllm/experimental/vision_tower_jit.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py)

## Classes
### `GridTHW`  ·  implements/extends tuple
- def: [`tpu_inference/models/vllm/experimental/vision_tower_jit.py:85`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L85)
- doc: Tensor-like wrapper for image/video grid_thw arguments.
- signature: `class GridTHW(tuple):`
- members:
  - `ndim(self)` — [`L109`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L109)
  - `prod(self, dim=-1)` — [`L119`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L119)
  - `shape(self)` — [`L113`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L113)
  - `tolist(self)` — [`L116`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L116)
- protocol/private: `__new__`[`L96`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L96), `__repr__`[`L124`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L124), `_nested_to_tuple`[`L98`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L98)
- used by: [`maybe_prepare_for_jit`](vision_tower_jit.md#maybe_prepare_for_jit), [`precompile_fn`](vision_tower_jit.md#maybe_precompile_vision_encoder_fn.precompile_fn)

## Functions
- `get_vision_config(hf_config: Any)` — [`L61`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L61) — Extract vision configuration from hf_config, supporting nested/thinker wrappers.
- `has_jittable_vision(vllm_model)` — [`L54`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L54) — Check if the model has any JIT-compiled vision component (either whole or submodule).
- `is_jittable_architecture(vllm_model)` — [`L41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L41) — Check if the given vLLM model is of an architecture that supports JIT compilation.
- `maybe_jit_embed_multimodal_func(embed_multimodal_func_jax: Callable, vllm_model)` — [`L69`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L69) — Conditionally wrap `embed_multimodal_func_jax` with jax.jit based on the VllmConfig.
- `maybe_precompile_vision_encoder_fn(params: Any, embed_multimodal_fn: Optional[Callable], vllm_model, vllm_config: VllmConfig)` — [`L128`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L128) — Return a precompile function for jittable vision encoders, or None.
- `maybe_prepare_for_jit(kwargs: dict, vllm_model)` — [`L186`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L186) — Convert certain kwargs to JIT-friendly formats, if needed.
- `precompile_fn(run_compilation_fn: Callable)` — [`L160`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L160)

## Module values
- `JITTABLE_ARCHS` — [`L36`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L36)
- `logger` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py#L33)

