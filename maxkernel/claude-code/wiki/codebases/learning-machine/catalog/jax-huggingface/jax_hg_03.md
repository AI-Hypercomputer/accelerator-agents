---
title: 'Module: jax-huggingface/jax_hg_03.py'
type: catalog
provenance: extracted
module: jax-huggingface/jax_hg_03.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 `jax-huggingface.jax_hg_03`/
symbols:
  attention_mask: attention_mask.
  weights: weights.
  _unflatten_static_cache: _unflatten_static_cache().
  autoregressive_decode_static: autoregressive_decode_static().
  model_inputs: model_inputs.
  input_ids: input_ids.
  mesh: mesh.
  model: model.
  tokenizer: tokenizer.
  env: env.
  output_flatten: output_flatten().
  output_unflatten: output_unflatten().
  _flatten_dynamic_cache: _flatten_dynamic_cache().
  _unflatten_dynamic_cache: _unflatten_dynamic_cache().
  _flatten_static_cache: _flatten_static_cache().
  shard_weights_llama: shard_weights_llama().
  autoregressive_decode_static.decode_one_tokens: autoregressive_decode_static().decode_one_tokens().
  autoregressive_decode_static.prefill_fn: autoregressive_decode_static().prefill_fn().
  run_twice_and_print_cache: run_twice_and_print_cache().
  run_twice_and_print_cache_static: run_twice_and_print_cache_static().
  autoregressive_decode: autoregressive_decode().
---
# Module: [`jax-huggingface/jax_hg_03.py`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py)

## Functions
- `_flatten_dynamic_cache(dynamic_cache)` — [`L37`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L37)
- `_flatten_static_cache(cache)` — [`L54`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L54)
- `_unflatten_dynamic_cache(aux, children)` — [`L43`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L43)
- `_unflatten_static_cache(aux, children)` — [`L63`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L63)
- `autoregressive_decode(model, input_ids, tokenizer, max_tokens=50, use_static_cache=False)` — [`L144`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L144)
- `autoregressive_decode_static(model, input_ids, tokenizer, attention_mask, max_tokens=50)` — [`L164`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L164)
- `decode_one_tokens(model_weights, cur_token, input_pos, cache_position, past_key_values)` — [`L166`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L166)
- `output_flatten(v)` — [`L23`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L23)
- `output_unflatten(aux, children)` — [`L26`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L26)
- `prefill_fn(model_weights, input_ids, attention_mask, cache_position, past_key_values)` — [`L181`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L181)
- `run_twice_and_print_cache(model, input_ids)` — [`L106`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L106)
- `run_twice_and_print_cache_static(model, input_ids)` — [`L124`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L124)
- `shard_weights_llama(mesh, weights)` — [`L84`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L84)

## Module values
- `attention_mask` — [`L262`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L262)
- `env` — [`L16`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L16)
- `input_ids` — [`L258`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L258)
- `mesh` — [`L15`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L15)
- `model` — [`L80`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L80)
- `model_inputs` — [`L12`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L12)
- `tokenizer` — [`L11`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L11)
- `weights` — [`L255`](../../../../../raw/code/learning-machine/jax-huggingface/jax_hg_03.py#L255)

