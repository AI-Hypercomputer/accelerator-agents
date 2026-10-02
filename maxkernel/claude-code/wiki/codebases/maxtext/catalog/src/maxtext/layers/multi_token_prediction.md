---
title: 'Module: src/maxtext/layers/multi_token_prediction.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/multi_token_prediction.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.multi_token_prediction`/
symbols:
  MultiTokenPredictionLayer.__call__: MultiTokenPredictionLayer#__call__().
  MultiTokenPredictionLayer: MultiTokenPredictionLayer#
  MultiTokenPredictionLayer.rngs: MultiTokenPredictionLayer#rngs.
  MultiTokenPredictionLayer.config: MultiTokenPredictionLayer#config.
  MultiTokenPredictionLayer.layer_number: MultiTokenPredictionLayer#layer_number.
  MultiTokenPredictionBlock.__call__: MultiTokenPredictionBlock#__call__().
  MultiTokenPredictionLayer.embedding_norm: MultiTokenPredictionLayer#embedding_norm().
  MultiTokenPredictionLayer.hidden_state_norm: MultiTokenPredictionLayer#hidden_state_norm().
  MultiTokenPredictionLayer.projection_layer: MultiTokenPredictionLayer#projection_layer().
  MultiTokenPredictionLayer.transformer_layer: MultiTokenPredictionLayer#transformer_layer().
  MultiTokenPredictionLayer.extract_fn: MultiTokenPredictionLayer#extract_fn().
  roll_and_mask: roll_and_mask().
  MultiTokenPredictionLayer.mesh: MultiTokenPredictionLayer#mesh.
  MultiTokenPredictionBlock.decoder: MultiTokenPredictionBlock#decoder.
  MultiTokenPredictionBlock.rngs: MultiTokenPredictionBlock#rngs.
  MultiTokenPredictionBlock.losses: MultiTokenPredictionBlock#losses.
  MultiTokenPredictionBlock.weights: MultiTokenPredictionBlock#weights.
  MultiTokenPredictionBlock.mtp_preds: MultiTokenPredictionBlock#mtp_preds.
  MultiTokenPredictionBlock.mtp_mask: MultiTokenPredictionBlock#mtp_mask.
  calculate_mtp_acceptance_rate: calculate_mtp_acceptance_rate().
  multi_token_prediction_block_as_linen: multi_token_prediction_block_as_linen().
  mtp_losses: mtp_losses#
  mtp_acceptance: mtp_acceptance#
  MultiTokenPredictionBlock: MultiTokenPredictionBlock#
  MultiTokenPredictionBlock.config: MultiTokenPredictionBlock#config.
  MultiTokenPredictionLayer.__init__: MultiTokenPredictionLayer#__init__().
  MultiTokenPredictionLayer.transformer_layer_module: MultiTokenPredictionLayer#transformer_layer_module.
  MultiTokenPredictionBlock.__init__: MultiTokenPredictionBlock#__init__().
  MultiTokenPredictionBlock.mesh: MultiTokenPredictionBlock#mesh.
  MultiTokenPredictionBlock.transformer_layer_module: MultiTokenPredictionBlock#transformer_layer_module.
  calculate_mtp_loss: calculate_mtp_loss().
---
# Module: [`src/maxtext/layers/multi_token_prediction.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py)

## Classes
### `MultiTokenPredictionBlock`  ·  implements/extends Module
- def: [`src/maxtext/layers/multi_token_prediction.py:226`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L226)
- doc: Orchestrates the MTP process by running a sequence of MTP layers.
- signature: `class MultiTokenPredictionBlock(nnx.Module):`
- members:
  - `config` — [`L237`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L237)
  - `decoder` — [`L240`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L240)
  - `losses` — [`L323`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L323)
  - `mesh` — [`L238`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L238)
  - `mtp_mask` — [`L328`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L328)
  - `mtp_preds` — [`L327`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L327)
  - `rngs` — [`L241`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L241)
  - `transformer_layer_module` — [`L239`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L239) — ---
  - `weights` — [`L324`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L324)
- protocol/private: `__call__`[`L254`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L254), `__init__`[`L229`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L229)
- uses (calls/refs, reference-scoped): [`MultiTokenPredictionLayer`](multi_token_prediction.md#MultiTokenPredictionLayer), [`roll_and_mask`](multi_token_prediction.md#roll_and_mask), [`mtp_acceptance`](multi_token_prediction.md#mtp_acceptance), [`mtp_losses`](multi_token_prediction.md#mtp_losses)
- used by: [`multi_token_prediction_block_as_linen`](multi_token_prediction.md#multi_token_prediction_block_as_linen)

### `MultiTokenPredictionLayer`  ·  implements/extends Module
- def: [`src/maxtext/layers/multi_token_prediction.py:63`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L63)
- doc: Multi-Token Prediction layer: normalize, concatenate, project, and transform.
- signature: `class MultiTokenPredictionLayer(nnx.Module):`
- members:
  - `__call__(self, prev_hidden_state: jnp.ndarray, target_token_embedding: jnp.ndarray, *, position_ids: jnp.ndarray, decoder_segment_ids: None | jnp.ndarray, deterministic: bool, model_mode: str = MODEL_MODE_TRAIN)` — [`L151`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L151) — Applies MTP combination, projection, and transformer processing.
  - `embedding_norm(self)` — [`L120`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L120)
  - `extract_fn(x)` — [`L196`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L196)
  - `hidden_state_norm(self)` — [`L128`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L128)
  - `projection_layer(self)` — [`L136`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L136)
  - `transformer_layer(self)` — [`L144`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L144)
  - `config` — [`L79`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L79)
  - `layer_number` — [`L81`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L81)
  - `mesh` — [`L80`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L80)
  - `rngs` — [`L83`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L83)
  - `transformer_layer_module` — [`L82`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L82) — ---
- protocol/private: `__init__`[`L70`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L70)
- used by: [`rngs`](multi_token_prediction.md#MultiTokenPredictionBlock.rngs)

### `mtp_acceptance`  ·  implements/extends Intermediate
- def: [`src/maxtext/layers/multi_token_prediction.py:44`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L44)
- doc: Variable type for storing MTP acceptance predictions -&gt; 'mtp_acceptance' collection.
- signature: `class mtp_acceptance(nnx.Intermediate):`
- used by: [`mtp_mask`](multi_token_prediction.md#MultiTokenPredictionBlock.mtp_mask), [`mtp_preds`](multi_token_prediction.md#MultiTokenPredictionBlock.mtp_preds)

### `mtp_losses`  ·  implements/extends Intermediate
- def: [`src/maxtext/layers/multi_token_prediction.py:40`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L40)
- doc: Variable type for storing MTP loss components -&gt; 'mtp_losses' collection.
- signature: `class mtp_losses(nnx.Intermediate):`
- used by: [`losses`](multi_token_prediction.md#MultiTokenPredictionBlock.losses), [`weights`](multi_token_prediction.md#MultiTokenPredictionBlock.weights)

## Functions
- `calculate_mtp_acceptance_rate(intermediate_outputs, config)` — [`L361`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L361) — Calculates MTP acceptance rate from intermediate outputs.
- `calculate_mtp_loss(intermediate_outputs, config)` — [`L333`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L333) — Calculates Multi-Token Prediction loss from intermediate outputs.
- `multi_token_prediction_block_as_linen(*, config: Config, mesh: Mesh, transformer_layer_module: Type[DecoderLayer], decoder: nnx.Module, rngs: nnx.Rngs, name: str | None = None)` — [`L390`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L390) — Initializes MultiTokenPredictionBlock as a Linen module.
- `roll_and_mask(x: jnp.ndarray, shift: int = -1)` — [`L48`](../../../../../../../raw/code/maxtext/src/maxtext/layers/multi_token_prediction.py#L48) — Performs a leftward roll on sequence axis and masks invalid positions.

