---
title: 'Module: tpu_inference/runner/mm_encoder_jit_manager.py'
type: catalog
provenance: extracted
module: tpu_inference/runner/mm_encoder_jit_manager.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.runner.mm_encoder_jit_manager`/
symbols:
  MMEncoderJITManager.budget_templates: MMEncoderJITManager#budget_templates.
  _TorchaxEncoderModelAdapter.run_budget_forward: _TorchaxEncoderModelAdapter#run_budget_forward().
  _TorchaxEncoderModelAdapter.encoder_eager_forward: _TorchaxEncoderModelAdapter#encoder_eager_forward().
  maybe_create_mm_encoder_jit_manager: maybe_create_mm_encoder_jit_manager().
  _TorchaxEncoderModelAdapter._build_forward_fn: _TorchaxEncoderModelAdapter#_build_forward_fn().
  MMEncoderJITManager.__init__: MMEncoderJITManager#__init__().
  MMEncoderJITManager._run_budget_graph: MMEncoderJITManager#_run_budget_graph().
  MMEncoderJITManager.precompile_vision_encoder: MMEncoderJITManager#precompile_vision_encoder().
  JaxEncoderModelAdapter._model: JaxEncoderModelAdapter#_model.
  MMEncoderJITManager: MMEncoderJITManager#
  logger: logger.
  _TorchaxEncoderModelAdapter._jit_forward: _TorchaxEncoderModelAdapter#_jit_forward.
  MMEncoderJITManager._pad_to_template: MMEncoderJITManager#_pad_to_template().
  MMEncoderJITManager._prepare_padded_torch: MMEncoderJITManager#_prepare_padded_torch().
  MMEncoderJITManager._capture_budget_graph: MMEncoderJITManager#_capture_budget_graph().
  _TorchaxEncoderModelAdapter.__getattr__: _TorchaxEncoderModelAdapter#__getattr__().
  JaxEncoderModelAdapter.run_budget_forward: JaxEncoderModelAdapter#run_budget_forward().
  JaxEncoderModelAdapter.encoder_eager_forward: JaxEncoderModelAdapter#encoder_eager_forward().
  JaxEncoderModelAdapter.postprocess_encoder_output: JaxEncoderModelAdapter#postprocess_encoder_output().
  JaxEncoderModelAdapter.__getattr__: JaxEncoderModelAdapter#__getattr__().
  _TorchaxEncoderModelAdapter._runner: _TorchaxEncoderModelAdapter#_runner.
  _TorchaxEncoderModelAdapter._params: _TorchaxEncoderModelAdapter#_params.
  MMEncoderJITManager.execute: MMEncoderJITManager#execute().
  _TorchaxEncoderModelAdapter: _TorchaxEncoderModelAdapter#
  _TorchaxEncoderModelAdapter._model: _TorchaxEncoderModelAdapter#_model.
  _TorchaxEncoderModelAdapter._forward: _TorchaxEncoderModelAdapter#_forward().
  _TorchaxEncoderModelAdapter._t2j_if_tensor: _TorchaxEncoderModelAdapter#_t2j_if_tensor().
  _TorchaxEncoderModelAdapter._torchax_view_if_torch: _TorchaxEncoderModelAdapter#_torchax_view_if_torch().
  JaxEncoderModelAdapter: JaxEncoderModelAdapter#
  _TorchaxEncoderModelAdapter.__init__: _TorchaxEncoderModelAdapter#__init__().
  _TorchaxEncoderModelAdapter.postprocess_encoder_output: _TorchaxEncoderModelAdapter#postprocess_encoder_output().
  JaxEncoderModelAdapter.__init__: JaxEncoderModelAdapter#__init__().
---
# Module: [`tpu_inference/runner/mm_encoder_jit_manager.py`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py)

## Classes
### `JaxEncoderModelAdapter`
- def: [`tpu_inference/runner/mm_encoder_jit_manager.py:200`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L200)
- doc: Wrap a JAX/flax SupportsEncoderCudaGraph model for MMEncoderJITManager.
- signature: `class JaxEncoderModelAdapter:`
- members:
  - `encoder_eager_forward(self, mm_kwargs: dict[str, Any])` — [`L222`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L222)
  - `postprocess_encoder_output(self, output: jax.Array, indices: list[int], per_item_out_tokens: list[int], dest, clone: bool = False, batch_mm_kwargs=None)` — [`L225`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L225)
  - `run_budget_forward(self, padded_torch: dict[str, Any])` — [`L209`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L209)
- protocol/private: `__getattr__`[`L236`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L236), `__init__`[`L206`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L206), `_model`[`L207`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L207)
- used by: [`__init__`](mm_encoder_jit_manager.md#MMEncoderJITManager.__init__)

### `MMEncoderJITManager`
- def: [`tpu_inference/runner/mm_encoder_jit_manager.py:240`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L240)
- doc: Per-budget XLA-cache manager for the vision encoder forward.
- signature: `class MMEncoderJITManager(EncoderCudaGraphManager):`
- members:
  - `__init__(self, vllm_config: VllmConfig, vllm_runner: torch.nn.Module | None, vllm_model: Any, params_and_buffers: Any)` — [`L243`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L243) — Args:
  - `_capture_budget_graph(self, token_budget: int)` — [`L377`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L377) — XLA-cache analog of ``torch.cuda.graph`` capture.
  - `_pad_to_template(self, replay_values: dict[str, torch.Tensor], budget: int)` — [`L312`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L312) — Zero-and-copy each replay tensor into a template-shaped buffer.
  - `_prepare_padded_torch(self, mm_kwargs: dict[str, Any], token_budget: int)` — [`L352`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L352) — Build the full padded (plain-torch) buffer dict for one batch.
  - `_run_budget_graph(self, mm_kwargs: dict[str, Any], token_budget: int)` — [`L392`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L392) — XLA-cache analog of CUDA-graph replay.
  - `execute(self, mm_kwargs: dict[str, Any])` — [`L418`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L418) — Run the encoder on one MM batch and return per-item outputs.
  - `precompile_vision_encoder(self, run_compilation: Callable)` — [`L433`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L433)
  - `budget_templates` — [`L297`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L297)
- uses (calls/refs, reference-scoped): [`logger`](mm_encoder_jit_manager.md#logger), [`JaxEncoderModelAdapter`](mm_encoder_jit_manager.md#JaxEncoderModelAdapter), [`_TorchaxEncoderModelAdapter`](mm_encoder_jit_manager.md#_TorchaxEncoderModelAdapter)
- used by: [`get_flax_model`](../models/common/model_loader.md#get_flax_model), [`embed_multimodal_func_torch`](../models/vllm/vllm_model_wrapper.md#VllmModelWrapper.embed_multimodal_func_torch), [`wrap_precompile_vision_encoder_fn`](../models/vllm/vllm_model_wrapper.md#VllmModelWrapper.wrap_precompile_vision_encoder_fn), [`_mm_encoder_jit_manager`](../models/vllm/vllm_model_wrapper.md#VllmModelWrapper._mm_encoder_jit_manager), [`embed_multimodal_fn`](../models/common/model_loader.md#get_flax_model.embed_multimodal_fn), [`maybe_create_mm_encoder_jit_manager`](mm_encoder_jit_manager.md#maybe_create_mm_encoder_jit_manager)

### `_TorchaxEncoderModelAdapter`
- def: [`tpu_inference/runner/mm_encoder_jit_manager.py:75`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L75)
- doc: Wrap the vLLM model so the inherited `_execute_local` eager
- signature: `class _TorchaxEncoderModelAdapter:`
- members:
  - `_build_forward_fn(self)` — [`L99`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L99) — Build the closure that gets ``jax.jit``-wrapped exactly once.
  - `_t2j_if_tensor(v: torch.Tensor | jax.Array)` — [`L182`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L182) — Tree-map helper — convert leaf torch.Tensors to jax.Array.
  - `_torchax_view_if_torch(v)` — [`L189`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L189) — Tree-map helper for the eager path.
  - `encoder_eager_forward(self, mm_kwargs: dict[str, Any])` — [`L140`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L140)
  - `postprocess_encoder_output(self, output: jax.Array, indices: list[int], per_item_out_tokens: list[int], dest, clone: bool = False, batch_mm_kwargs=None)` — [`L164`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L164)
  - `run_budget_forward(self, padded_torch: dict[str, Any])` — [`L131`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L131)
- protocol/private: `__getattr__`[`L195`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L195), `__init__`[`L90`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L90), `_forward`[`L111`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L111), `_jit_forward`[`L95`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L95), `_model`[`L92`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L92), `_params`[`L94`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L94), `_runner`[`L93`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L93)
- used by: [`__init__`](mm_encoder_jit_manager.md#MMEncoderJITManager.__init__)

## Functions
- `maybe_create_mm_encoder_jit_manager(vllm_config: VllmConfig, vllm_model: Any, vllm_runner: torch.nn.Module | None, params_and_buffers: Any)` — [`L443`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L443)

## Module values
- `logger` — [`L72`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/mm_encoder_jit_manager.py#L72)

