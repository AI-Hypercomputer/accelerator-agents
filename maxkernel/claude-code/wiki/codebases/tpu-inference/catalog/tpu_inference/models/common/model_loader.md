---
title: 'Module: tpu_inference/models/common/model_loader.py'
type: catalog
provenance: extracted
module: tpu_inference/models/common/model_loader.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.common.model_loader`/
symbols:
  get_flax_model: get_flax_model().
  get_vllm_model: get_vllm_model().
  _get_model_architecture: _get_model_architecture().
  get_model: get_model().
  _get_nnx_model: _get_nnx_model().
  _MODEL_REGISTRY: _MODEL_REGISTRY.
  register_model: register_model().
  resolve_model_architecture: resolve_model_architecture().
  logger: logger.
  get_flax_model.embed_multimodal_fn: get_flax_model().embed_multimodal_fn().
  get_flax_model.run_embed_input_ids: get_flax_model().run_embed_input_ids().
  _get_nnx_model.create_jit_model: _get_nnx_model().create_jit_model().
  _get_nnx_model.create_abstract_model: _get_nnx_model().create_abstract_model().
  _get_nnx_model.create_sharded_model: _get_nnx_model().create_sharded_model().
  UnsupportedArchitectureError: UnsupportedArchitectureError#
  get_flax_model.run_embed_multimodal: get_flax_model().run_embed_multimodal().
  _not_support: _not_support().
  get_flax_model.run_model: get_flax_model().run_model().
  get_flax_model.run_draft_model: get_flax_model().run_draft_model().
  get_flax_model.run_compute_logits: get_flax_model().run_compute_logits().
  get_flax_model.jitted_embed_input_ids: get_flax_model().jitted_embed_input_ids().
  get_flax_model.combine_hidden_states: get_flax_model().combine_hidden_states().
  get_flax_model.wrapped_model_fn: get_flax_model().wrapped_model_fn().
  get_flax_model.compute_pooler_output: get_flax_model().compute_pooler_output().
  _validate_model_interface: _validate_model_interface().
  register_model.unimplemented_forward: register_model().unimplemented_forward().
  register_model.unimplemented_embed_input_ids: register_model().unimplemented_embed_input_ids().
  register_model.wrapper_init: register_model().wrapper_init().
  _VLLM_PREFERRED_ARCHITECTURES._VLLM_PREFERRED_ARCHITECTURES: _VLLM_PREFERRED_ARCHITECTURES._VLLM_PREFERRED_ARCHITECTURES.
  _PP_DISABLED_MODELS._PP_DISABLED_MODELS: _PP_DISABLED_MODELS._PP_DISABLED_MODELS.
---
# Module: [`tpu_inference/models/common/model_loader.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py)

## Classes
### `UnsupportedArchitectureError`  ·  implements/extends ValueError
- def: [`tpu_inference/models/common/model_loader.py:66`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L66)
- doc: Raised when a model architecture is not supported in the registry.
- signature: `class UnsupportedArchitectureError(ValueError):`
- used by: [`_get_model_architecture`](model_loader.md#_get_model_architecture), [`get_model`](model_loader.md#get_model), [`resolve_model_architecture`](model_loader.md#resolve_model_architecture)

## Functions
- `_get_model_architecture(config: PretrainedConfig)` — [`L71`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L71)
- `_get_nnx_model(model_class: Any, vllm_config: VllmConfig, rng: jax.Array, mesh: Mesh, pooler: Optional[Any] = None, is_draft_model: bool = False)` — [`L115`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L115) — Instantiate the nnx JAX model and optionally pass the embedding/pooling layer.
- `_not_support(*args, **kwargs)` — [`L306`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L306)
- `_validate_model_interface(model: Any)` — [`L713`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L713) — Validates that the model class has the required methods and signatures.
- `combine_hidden_states(state_leaves, hidden_states)` — [`L446`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L446)
- `compute_pooler_output(hidden_states: jax.Array, pooling_metadata: PoolingMetadata, seq_lens: np.ndarray, num_scheduled_tokens: Optional[np.ndarray] = None)` — [`L508`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L508)
- `create_abstract_model()` — [`L139`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L139) — Helper class to create an abstract model for `nnx.eval_shape`.
- `create_jit_model(model: nnx.Module, use_qwix_on_abstract_model: bool = False)` — [`L150`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L150) — Create a jit model.
- `create_sharded_model()` — [`L223`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L223)
- `embed_multimodal_fn(state_leaves, modality=None, **kwargs)` — [`L478`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L478)
- `get_flax_model(vllm_config: VllmConfig, rng: jax.Array, mesh: Mesh, is_draft_model: bool = False)` — [`L311`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L311)
- `get_model(vllm_config: VllmConfig, rng: jax.Array, mesh: Mesh, is_draft_model: bool = False, shared_params: Optional[dict[str, jax.Array]] = None)` — [`L607`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L607)
- `get_vllm_model(vllm_config: VllmConfig, rng: jax.Array, mesh: Mesh, is_draft_model: bool = False, shared_params: Optional[dict[str, jax.Array]] = None)` — [`L552`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L552)
- `jitted_embed_input_ids(state_leaves, input_ids, mm_embeds, is_multimodal=None)` — [`L423`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L423)
- `register_model(arch: str, model: Any)` — [`L764`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L764) — Registers a model class for a given architecture name.
- `resolve_model_architecture(vllm_config: VllmConfig, is_draft_model: bool)` — [`L654`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L654) — Resolves the model implementation type.
- `run_compute_logits(state_leaves, *args)` — [`L407`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L407)
- `run_draft_model(state_leaves, *args)` — [`L397`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L397)
- `run_embed_input_ids(state_leaves, input_ids, mm_embeds=None, is_multimodal=None)` — [`L433`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L433)
- `run_embed_multimodal(state_leaves, modality=None, **kwargs)` — [`L415`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L415)
- `run_model(state_leaves, *args)` — [`L382`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L382)
- `unimplemented_embed_input_ids(self, input_ids: torch.Tensor, positions: torch.Tensor, inputs_embeds: Optional[torch.Tensor] = None)` — [`L797`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L797)
- `unimplemented_forward(self, input_ids: torch.Tensor, positions: torch.Tensor, intermediate_tensors: Optional[Any] = None, inputs_embeds: Optional[torch.Tensor] = None)` — [`L785`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L785)
- `wrapped_model_fn(*args, **kwargs)` — [`L461`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L461)
- `wrapper_init(self, *args, **kwargs)` — [`L809`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L809)

## Module values
- `_MODEL_REGISTRY` — [`L50`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L50)
- `_PP_DISABLED_MODELS` — [`L62`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L62)
- `_VLLM_PREFERRED_ARCHITECTURES` — [`L54`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L54)
- `logger` — [`L48`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/model_loader.py#L48)

