---
title: 'Module: torchax/checkpoint.py'
type: catalog
provenance: extracted
module: torchax/checkpoint.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.checkpoint`/
symbols:
  _to_jax: _to_jax().
  load_checkpoint: load_checkpoint().
  _to_jax.to_jax_array: _to_jax().to_jax_array().
  save_checkpoint: save_checkpoint().
  _to_torch: _to_torch().
  _to_torch.to_torch_tensor: _to_torch().to_torch_tensor().
---
# Module: [`torchax/checkpoint.py`](../../../../../raw/code/torchax/torchax/checkpoint.py)

## Functions
- `_to_jax(pytree)` — [`L26`](../../../../../raw/code/torchax/torchax/checkpoint.py#L26)
- `_to_torch(pytree)` — [`L37`](../../../../../raw/code/torchax/torchax/checkpoint.py#L37)
- `load_checkpoint(path: str)` — [`L62`](../../../../../raw/code/torchax/torchax/checkpoint.py#L62) — Loads a checkpoint and returns it in JAX format.
- `save_checkpoint(state: dict[str, Any], path: str, step: int)` — [`L46`](../../../../../raw/code/torchax/torchax/checkpoint.py#L46) — Saves a checkpoint to a file in JAX style.
- `to_jax_array(x)` — [`L27`](../../../../../raw/code/torchax/torchax/checkpoint.py#L27)
- `to_torch_tensor(x)` — [`L38`](../../../../../raw/code/torchax/torchax/checkpoint.py#L38)

