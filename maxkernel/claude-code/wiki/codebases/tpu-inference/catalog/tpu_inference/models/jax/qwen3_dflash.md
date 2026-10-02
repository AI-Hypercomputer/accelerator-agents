---
title: 'Module: tpu_inference/models/jax/qwen3_dflash.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/qwen3_dflash.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.qwen3_dflash`/
symbols:
  logger: logger.
  init_fn: init_fn.
  _build_target_layer_ids: _build_target_layer_ids().
  _get_dflash_target_layer_ids: _get_dflash_target_layer_ids().
  Qwen3DFlashAttention: Qwen3DFlashAttention#
  Qwen3DFlashAttention.__init__: Qwen3DFlashAttention#__init__().
  Qwen3DFlashAttention.__call__: Qwen3DFlashAttention#__call__().
  Qwen3DFlashDecoderLayer: Qwen3DFlashDecoderLayer#
  Qwen3DFlashDecoderLayer.__init__: Qwen3DFlashDecoderLayer#__init__().
  Qwen3DFlashDecoderLayer.__call__: Qwen3DFlashDecoderLayer#__call__().
  Qwen3DFlashModel: Qwen3DFlashModel#
  Qwen3DFlashModel.__init__: Qwen3DFlashModel#__init__().
  Qwen3DFlashModel.__call__: Qwen3DFlashModel#__call__().
  Qwen3DFlashModel.combine_hidden_states: Qwen3DFlashModel#combine_hidden_states().
  Qwen3DFlashWeightLoader: Qwen3DFlashWeightLoader#
  Qwen3DFlashWeightLoader.__init__: Qwen3DFlashWeightLoader#__init__().
  Qwen3DFlashWeightLoader.load_weights: Qwen3DFlashWeightLoader#load_weights().
  Qwen3DFlashForCausalLM: Qwen3DFlashForCausalLM#
  Qwen3DFlashForCausalLM.WeightLoader: Qwen3DFlashForCausalLM#WeightLoader.
  Qwen3DFlashForCausalLM.__init__: Qwen3DFlashForCausalLM#__init__().
  Qwen3DFlashForCausalLM.__call__: Qwen3DFlashForCausalLM#__call__().
  Qwen3DFlashForCausalLM.compute_logits: Qwen3DFlashForCausalLM#compute_logits().
  Qwen3DFlashForCausalLM.combine_hidden_states: Qwen3DFlashForCausalLM#combine_hidden_states().
  Qwen3DFlashForCausalLM.load_weights: Qwen3DFlashForCausalLM#load_weights().
---
# Module: [`tpu_inference/models/jax/qwen3_dflash.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py)

## Classes
### `Qwen3DFlashAttention`
- def: [`tpu_inference/models/jax/qwen3_dflash.py:69`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L69)
- signature: `class Qwen3DFlashAttention(nnx.Module):`
- protocol/private: `__call__`[`L155`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L155), `__init__`[`L71`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L71)

### `Qwen3DFlashDecoderLayer`
- def: [`tpu_inference/models/jax/qwen3_dflash.py:255`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L255)
- signature: `class Qwen3DFlashDecoderLayer(nnx.Module):`
- protocol/private: `__call__`[`L293`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L293), `__init__`[`L257`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L257)

### `Qwen3DFlashForCausalLM`
- def: [`tpu_inference/models/jax/qwen3_dflash.py:448`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L448)
- signature: `class Qwen3DFlashForCausalLM(nnx.Module):`
- members:
  - `combine_hidden_states(self, hidden_states: jax.Array)` — [`L479`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L479)
  - `compute_logits(self, hidden_states: jax.Array)` — [`L476`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L476)
  - `load_weights(self, _rng_key: jax.Array)` — [`L482`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L482)
  - `WeightLoader` — [`L449`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L449)
- protocol/private: `__call__`[`L462`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L462), `__init__`[`L451`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L451)

### `Qwen3DFlashModel`
- def: [`tpu_inference/models/jax/qwen3_dflash.py:316`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L316)
- signature: `class Qwen3DFlashModel(nnx.Module):`
- members:
  - `combine_hidden_states(self, hidden_states: jax.Array)` — [`L410`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L410)
- protocol/private: `__call__`[`L385`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L385), `__init__`[`L318`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L318)

### `Qwen3DFlashWeightLoader`
- def: [`tpu_inference/models/jax/qwen3_dflash.py:416`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L416)
- signature: `class Qwen3DFlashWeightLoader(BaseWeightLoader):`
- members:
  - `load_weights(self, model: 'Qwen3DFlashForCausalLM', mappings: dict)` — [`L423`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L423)
- protocol/private: `__init__`[`L418`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L418)

## Functions
- `_build_target_layer_ids(num_target_layers: int, num_draft_layers: int)` — [`L46`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L46)
- `_get_dflash_target_layer_ids(draft_hf_config: Qwen3Config, target_num_layers: int)` — [`L56`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L56)

## Module values
- `init_fn` — [`L43`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L43)
- `logger` — [`L41`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_dflash.py#L41)

