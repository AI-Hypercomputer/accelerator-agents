---
title: 'Module: tpu_inference/models/jax/dflash.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/dflash.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.dflash`/
symbols:
  logger: logger.
  init_fn: init_fn.
  _FA_VMEM_LIMIT: _FA_VMEM_LIMIT.
  DFlashAttention: DFlashAttention#
  DFlashAttention.__init__: DFlashAttention#__init__().
  DFlashAttention.__call__: DFlashAttention#__call__().
  DFlashMLP: DFlashMLP#
  DFlashMLP.__init__: DFlashMLP#__init__().
  DFlashMLP.__call__: DFlashMLP#__call__().
  DFlashDecoderLayer: DFlashDecoderLayer#
  DFlashDecoderLayer.__init__: DFlashDecoderLayer#__init__().
  DFlashDecoderLayer.__call__: DFlashDecoderLayer#__call__().
  DFlashModel: DFlashModel#
  DFlashModel.__init__: DFlashModel#__init__().
  DFlashWeightLoader: DFlashWeightLoader#
  DFlashWeightLoader.__init__: DFlashWeightLoader#__init__().
  DFlashWeightLoader.load_weights: DFlashWeightLoader#load_weights().
  DFlashForCausalLM: DFlashForCausalLM#
  DFlashForCausalLM.WeightLoader: DFlashForCausalLM#WeightLoader.
  DFlashForCausalLM.__init__: DFlashForCausalLM#__init__().
  DFlashForCausalLM.__call__: DFlashForCausalLM#__call__().
  DFlashForCausalLM.compute_logits: DFlashForCausalLM#compute_logits().
  DFlashForCausalLM.combine_hidden_states: DFlashForCausalLM#combine_hidden_states().
  DFlashForCausalLM.load_weights: DFlashForCausalLM#load_weights().
---
# Module: [`tpu_inference/models/jax/dflash.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py)

## Classes
### `DFlashAttention`
- def: [`tpu_inference/models/jax/dflash.py:46`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L46)
- doc: DFlash cross+self attention with on-device KV cache.
- signature: `class DFlashAttention(nnx.Module):`
- members:
  - `__call__(self, x_noise: jax.Array, target_hidden: jax.Array, noise_positions: jax.Array, ctx_positions: jax.Array, kv_cache_k: jax.Array, kv_cache_v: jax.Array, cache_len: jax.Array, actual_ctx_count: jax.Array)` — [`L134`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L134) — Non-causal attention with on-device KV cache.
- protocol/private: `__init__`[`L58`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L58)

### `DFlashDecoderLayer`
- def: [`tpu_inference/models/jax/dflash.py:294`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L294)
- signature: `class DFlashDecoderLayer(nnx.Module):`
- members:
  - `__call__(self, x: jax.Array, target_hidden: jax.Array, noise_positions: jax.Array, ctx_positions: jax.Array, kv_cache_k: jax.Array, kv_cache_v: jax.Array, cache_len: jax.Array, actual_ctx_count: jax.Array)` — [`L328`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L328) — Returns (hidden_states, new_kv_cache_k, new_kv_cache_v).
- protocol/private: `__init__`[`L296`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L296)

### `DFlashForCausalLM`
- def: [`tpu_inference/models/jax/dflash.py:465`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L465)
- doc: DFlash draft model for speculative decoding on TPU.
- signature: `class DFlashForCausalLM(nnx.Module):`
- members:
  - `__call__(self, kv_caches: List[jax.Array], input_ids: jax.Array, target_hidden_states: jax.Array, attention_metadata)` — [`L498`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L498) — Forward pass for the DFlash draft model.
  - `combine_hidden_states(self, hidden_states: jax.Array)` — [`L559`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L559) — Project concatenated target auxiliary hidden states.
  - `compute_logits(self, hidden_states: jax.Array)` — [`L554`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L554) — Compute logits using tied embedding weights.
  - `load_weights(self, rng_key: jax.Array)` — [`L570`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L570)
  - `WeightLoader` — [`L468`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L468)
- protocol/private: `__init__`[`L470`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L470)

### `DFlashMLP`
- def: [`tpu_inference/models/jax/dflash.py:257`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L257)
- signature: `class DFlashMLP(nnx.Module):`
- protocol/private: `__call__`[`L290`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L290), `__init__`[`L259`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L259)

### `DFlashModel`
- def: [`tpu_inference/models/jax/dflash.py:361`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L361)
- signature: `class DFlashModel(nnx.Module):`
- protocol/private: `__init__`[`L363`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L363)

### `DFlashWeightLoader`
- def: [`tpu_inference/models/jax/dflash.py:433`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L433)
- signature: `class DFlashWeightLoader(BaseWeightLoader):`
- members:
  - `load_weights(self, model: 'DFlashForCausalLM', mappings: dict)` — [`L440`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L440)
- protocol/private: `__init__`[`L435`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L435)

## Module values
- `_FA_VMEM_LIMIT` — [`L43`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L43)
- `init_fn` — [`L40`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L40)
- `logger` — [`L38`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/dflash.py#L38)

