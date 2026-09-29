---
title: 'Module: llama_ref/run.py'
type: catalog
provenance: extracted
module: llama_ref/run.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 `llama_ref.run`/
symbols:
  main: main().
  main2: main2().
  _fwd_cost_estimate: _fwd_cost_estimate().
  _process_sharding_name: _process_sharding_name().
  create_sharded_weights: create_sharded_weights().
  sharded_device_put: sharded_device_put().
  main.custom_attention: main().custom_attention().
  main.custom_attention.wrap_flash_attention: main().custom_attention().wrap_flash_attention().
  sharding_map_scan: sharding_map_scan.
  _bytes: _bytes().
  sharding_map_original: sharding_map_original.
  _process_sharding_name.is_integer: _process_sharding_name().is_integer().
  register_attention: register_attention().
---
# Module: [`llama_ref/run.py`](../../../../../raw/code/learning-machine/llama_ref/run.py)

## Functions
- `_bytes(x: jax.Array | jax.ShapeDtypeStruct)` — [`L57`](../../../../../raw/code/learning-machine/llama_ref/run.py#L57)
- `_fwd_cost_estimate(q: jax.Array, k: jax.Array, v: jax.Array, *args, kernel_inputs_specs, kernel_outputs_specs, **kwargs)` — [`L60`](../../../../../raw/code/learning-machine/llama_ref/run.py#L60)
- `_process_sharding_name(name)` — [`L91`](../../../../../raw/code/learning-machine/llama_ref/run.py#L91) — Replace integers in param name with *.
- `create_sharded_weights(model, mesh, sharding_map)` — [`L126`](../../../../../raw/code/learning-machine/llama_ref/run.py#L126)
- `custom_attention(query, key, value, attn_mask=None, dropout_p=0, is_causal=False, scale=None, enable_gqa=False)` — [`L251`](../../../../../raw/code/learning-machine/llama_ref/run.py#L251)
- `is_integer(t)` — [`L97`](../../../../../raw/code/learning-machine/llama_ref/run.py#L97)
- `main(batch_size: int = 64, model_type: str = '8B', lr: float = 0.001, tp: int = 4, seqlen: int = 2048, model_impl: str = 'scan', use_custom_mesh: bool = False, use_custom_offload: bool = True, internal_override_layers: int = -1, profile_dir: str = 'profile/', unroll_layers: int = 1)` — [`L160`](../../../../../raw/code/learning-machine/llama_ref/run.py#L160) — documented in [llama_ref-model](../../concepts/llama_ref-model.md)
- `main2(batch_size: int = 64, model_type: str = '8B', lr: float = 0.001, tp: int = 4, seqlen: int = 2048, model_impl: str = 'scan', use_custom_mesh: bool = False, use_custom_offload: bool = True, internal_override_layers: int = -1, profile_dir: str = 'profile/')` — [`L295`](../../../../../raw/code/learning-machine/llama_ref/run.py#L295)
- `register_attention(fn)` — [`L112`](../../../../../raw/code/learning-machine/llama_ref/run.py#L112)
- `sharded_device_put(tensor, sharding)` — [`L145`](../../../../../raw/code/learning-machine/llama_ref/run.py#L145)
- `wrap_flash_attention(query, key, value)` — [`L258`](../../../../../raw/code/learning-machine/llama_ref/run.py#L258)

## Module values
- `sharding_map_original` — [`L21`](../../../../../raw/code/learning-machine/llama_ref/run.py#L21)
- `sharding_map_scan` — [`L37`](../../../../../raw/code/learning-machine/llama_ref/run.py#L37)

