---
title: 'Module: tpu_inference/models/vllm/vllm_model_loader.py'
type: catalog
provenance: extracted
module: tpu_inference/models/vllm/vllm_model_loader.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.vllm.vllm_model_loader`/
symbols:
  attach_incremental_weight_loader: attach_incremental_weight_loader().
  IncrementalModelLoader: IncrementalModelLoader#
  IncrementalModelLoader.__init__: IncrementalModelLoader#__init__().
  IncrementalModelLoader.load_model: IncrementalModelLoader#load_model().
  RunaiIncrementalModelLoader: RunaiIncrementalModelLoader#
  RunaiIncrementalModelLoader.__init__: RunaiIncrementalModelLoader#__init__().
  RunaiIncrementalModelLoader._prepare_weights: RunaiIncrementalModelLoader#_prepare_weights().
  RunaiIncrementalModelLoader.load_model: RunaiIncrementalModelLoader#load_model().
---
# Module: [`tpu_inference/models/vllm/vllm_model_loader.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_loader.py)

## Classes
### `IncrementalModelLoader`
- def: [`tpu_inference/models/vllm/vllm_model_loader.py:73`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_loader.py#L73)
- doc: Model loader that supports incremental weight loading and sharding.
- signature: `class IncrementalModelLoader(DefaultModelLoader):`
- members:
  - `load_model(self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str='')` — [`L88`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_loader.py#L88) — Load a model with the given configurations.
- protocol/private: `__init__`[`L84`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_loader.py#L84)

### `RunaiIncrementalModelLoader`
- def: [`tpu_inference/models/vllm/vllm_model_loader.py:112`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_loader.py#L112)
- doc: Model loader that supports both RunAI streaming and incremental weight sharding.
- signature: `class RunaiIncrementalModelLoader(RunaiModelStreamerLoader):`
- members:
  - `load_model(self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str='')` — [`L128`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_loader.py#L128) — Load a model with the given configurations.
- protocol/private: `__init__`[`L115`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_loader.py#L115), `_prepare_weights`[`L118`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_loader.py#L118)

## Functions
- `attach_incremental_weight_loader(model: torch.nn.Module)` — [`L32`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/vllm/vllm_model_loader.py#L32) — Traverses the model and overrides the weight_loader of each parameter to support incremental loading.

