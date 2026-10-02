---
title: 'Module: tpu_inference/models/common/pathways_dummy_loader.py'
type: catalog
provenance: extracted
module: tpu_inference/models/common/pathways_dummy_loader.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.common.pathways_dummy_loader`/
symbols:
  _process_weights_after_loading_jax: _process_weights_after_loading_jax().
  load_dummy_weights_jax: load_dummy_weights_jax().
  create_dummy_weights_on_tpu: create_dummy_weights_on_tpu().
  create_dummy_weights_on_tpu._generate: create_dummy_weights_on_tpu()._generate().
  PathwaysDummyModelLoader.load_weights: PathwaysDummyModelLoader#load_weights().
  is_pathways_dummy_load: is_pathways_dummy_load().
  logger: logger.
  _LOW: _LOW.
  _HIGH: _HIGH.
  _SEED: _SEED.
  PathwaysDummyModelLoader: PathwaysDummyModelLoader#
  PathwaysDummyModelLoader.__init__: PathwaysDummyModelLoader#__init__().
  PathwaysDummyModelLoader.download_model: PathwaysDummyModelLoader#download_model().
  PathwaysDummyModelLoader.load_model: PathwaysDummyModelLoader#load_model().
---
# Module: [`tpu_inference/models/common/pathways_dummy_loader.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/pathways_dummy_loader.py)

## Classes
### `PathwaysDummyModelLoader`
- def: [`tpu_inference/models/common/pathways_dummy_loader.py:144`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/pathways_dummy_loader.py#L144)
- doc: Model loader that creates dummy weights directly on the TPU mesh.
- signature: `class PathwaysDummyModelLoader(BaseModelLoader):`
- members:
  - `download_model(self, model_config: ModelConfig)` — [`L170`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/pathways_dummy_loader.py#L170)
  - `load_model(self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = "")` — [`L196`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/pathways_dummy_loader.py#L196) — Initialise a vLLM (torchax) model without loading real weights.
  - `load_weights(self, model, model_config: ModelConfig)` — [`L173`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/pathways_dummy_loader.py#L173) — Load dummy weights into *model*.
- protocol/private: `__init__`[`L167`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/pathways_dummy_loader.py#L167)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`load_dummy_weights_jax`](pathways_dummy_loader.md#load_dummy_weights_jax)

## Functions
- `_generate(key)` — [`L67`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/pathways_dummy_loader.py#L67)
- `_process_weights_after_loading_jax(module)` — [`L126`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/pathways_dummy_loader.py#L126) — Recursively call ``process_weights_after_loading`` if available.
- `create_dummy_weights_on_tpu(sharding: NamedSharding, weight_shape: tuple[int, ...], weight_dtype: jnp.dtype)` — [`L55`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/pathways_dummy_loader.py#L55) — Create small random dummy weights directly on the TPU mesh.
- `is_pathways_dummy_load()` — [`L48`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/pathways_dummy_loader.py#L48)
- `load_dummy_weights_jax(model, mesh: Mesh)` — [`L80`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/pathways_dummy_loader.py#L80) — Fill every nnx.Param in `model` with random TPU-resident data.

## Module values
- `_HIGH` — [`L44`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/pathways_dummy_loader.py#L44)
- `_LOW` — [`L43`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/pathways_dummy_loader.py#L43)
- `_SEED` — [`L45`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/pathways_dummy_loader.py#L45)
- `logger` — [`L41`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/pathways_dummy_loader.py#L41)

