---
title: 'Module: tpu_inference/models/common/interface.py'
type: catalog
provenance: extracted
module: tpu_inference/models/common/interface.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.common.interface`/
symbols:
  ModelInterface.multimodal_fns: ModelInterface#multimodal_fns.
  ModelInterface: ModelInterface#
  ModelInterface.model_fn: ModelInterface#model_fn.
  ModelInterface.compute_logits_fn: ModelInterface#compute_logits_fn.
  ModelInterface.pooler_fn: ModelInterface#pooler_fn.
  ModelInterface.combine_hidden_states_fn: ModelInterface#combine_hidden_states_fn.
  ModelInterface.state: ModelInterface#state.
  ModelInterface.state_leaves: ModelInterface#state_leaves.
  ModelInterface.model: ModelInterface#model.
  MultiModalInterface: MultiModalInterface#
  MultiModalInterface.precompile_vision_encoder_fn: MultiModalInterface#precompile_vision_encoder_fn.
  MultiModalInterface.embed_multimodal_fn: MultiModalInterface#embed_multimodal_fn.
  MultiModalInterface.embed_input_ids_fn: MultiModalInterface#embed_input_ids_fn.
  MultiModalInterface.get_mrope_input_positions_fn: MultiModalInterface#get_mrope_input_positions_fn.
  ModelInterface.lora_manager: ModelInterface#lora_manager.
  PoolerFunc: PoolerFunc#
  PoolerFunc.__call__: PoolerFunc#__call__().
---
# Module: [`tpu_inference/models/common/interface.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py)

## Classes
### `ModelInterface`
- def: [`tpu_inference/models/common/interface.py:41`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L41)
- signature: `class ModelInterface:`
- members:
  - `combine_hidden_states_fn` — [`L45`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L45)
  - `compute_logits_fn` — [`L43`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L43)
  - `lora_manager` — [`L56`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L56)
  - `model` — [`L57`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L57)
  - `model_fn` — [`L42`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L42)
  - `multimodal_fns` — [`L46`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L46)
  - `pooler_fn` — [`L44`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L44)
  - `state` — [`L49`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L49)
  - `state_leaves` — [`L55`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L55)
- uses (calls/refs, reference-scoped): [`MultiModalInterface`](interface.md#MultiModalInterface)
- used by: [`get_flax_model`](model_loader.md#get_flax_model), [`get_vllm_model`](model_loader.md#get_vllm_model), `state`, [`get_model`](model_loader.md#get_model), `state_leaves`, [`model`](../../spec_decode/jax/eagle3.md#Eagle3Proposer.model), `embed_input_ids_fn`, `compute_logits_fn`, `get_mrope_input_positions_fn`, `model_fn`, `embed_multimodal_fn`, `model`, `precompile_vision_encoder_fn`, [`state`](../../spec_decode/jax/eagle3.md#Eagle3Proposer.state), [`state_leaves`](../../spec_decode/jax/eagle3.md#Eagle3Proposer.state_leaves), [`model_fn`](../../spec_decode/jax/eagle3.md#Eagle3Proposer.model_fn), `pooler_fn`, [`combine_hidden_states_fn`](../../spec_decode/jax/eagle3.md#Eagle3Proposer.combine_hidden_states_fn), [`compute_logits_fn`](../../spec_decode/jax/eagle3.md#Eagle3Proposer.compute_logits_fn), `combine_hidden_states_fn`, `lora_manager`, [`pooler_fn`](../../spec_decode/jax/eagle3.md#Eagle3Proposer.pooler_fn)

### `MultiModalInterface`
- def: [`tpu_inference/models/common/interface.py:33`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L33)
- signature: `class MultiModalInterface:`
- members:
  - `embed_input_ids_fn` — [`L36`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L36)
  - `embed_multimodal_fn` — [`L35`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L35)
  - `get_mrope_input_positions_fn` — [`L37`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L37)
  - `precompile_vision_encoder_fn` — [`L34`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L34)
- used by: [`get_flax_model`](model_loader.md#get_flax_model), [`get_vllm_model`](model_loader.md#get_vllm_model), [`multimodal_fns`](interface.md#ModelInterface.multimodal_fns), `embed_input_ids_fn`, `get_mrope_input_positions_fn`, `embed_multimodal_fn`, `precompile_vision_encoder_fn`

### `PoolerFunc`  ·  implements/extends Protocol
- def: [`tpu_inference/models/common/interface.py:13`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L13)
- doc: The wrapped pooler interface.
- signature: `class PoolerFunc(Protocol):`
- protocol/private: `__call__`[`L22`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/common/interface.py#L22)
- used by: [`build_pooler_func`](../vllm/vllm_model_wrapper.md#VllmModelWrapper.build_pooler_func)

