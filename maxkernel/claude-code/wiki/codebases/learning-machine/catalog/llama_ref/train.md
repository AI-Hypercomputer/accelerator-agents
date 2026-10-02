---
title: 'Module: llama_ref/train.py'
type: catalog
provenance: extracted
module: llama_ref/train.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 `llama_ref.train`/
symbols:
  train_loop: train_loop().
  make_train_step: make_train_step().
  make_train_step.loss: make_train_step().loss().
  sharded_device_put: sharded_device_put().
  train_loop.model_forward_shmap: train_loop().model_forward_shmap().
  TraininableLlama.call: TraininableLlama#call().
  mark_sharding: mark_sharding.
  train_loop._shard_first_dim: train_loop()._shard_first_dim().
  TraininableLlama.orig_model: TraininableLlama#orig_model.
  fake_dataloader: fake_dataloader().
  remat: remat.
  make_train_step.step: make_train_step().step().
  _prelower_step: _prelower_step().
  train_loop.model_forward_shmap.gather_weights: train_loop().model_forward_shmap().gather_weights().
  train_loop._expand_input: train_loop()._expand_input().
  train_loop._replicate: train_loop()._replicate().
  SEQLEN: SEQLEN.
  TraininableLlama: TraininableLlama#
  TraininableLlama.__init__: TraininableLlama#__init__().
  group_data: group_data().
---
# Module: [`llama_ref/train.py`](../../../../../raw/code/learning-machine/llama_ref/train.py)

## Classes
### `TraininableLlama`
- def: [`llama_ref/train.py:21`](../../../../../raw/code/learning-machine/llama_ref/train.py#L21)
- signature: `class TraininableLlama:`
- members:
  - `call(self, weights, buffers, args, kwargs)` — [`L27`](../../../../../raw/code/learning-machine/llama_ref/train.py#L27)
  - `orig_model` — [`L24`](../../../../../raw/code/learning-machine/llama_ref/train.py#L24)
- protocol/private: `__init__`[`L23`](../../../../../raw/code/learning-machine/llama_ref/train.py#L23)

## Functions
- `_expand_input(input_seq)` — [`L208`](../../../../../raw/code/learning-machine/llama_ref/train.py#L208)
- `_prelower_step(step, weights, opt_state, args, label, mesh)` — [`L114`](../../../../../raw/code/learning-machine/llama_ref/train.py#L114)
- `_replicate(x)` — [`L227`](../../../../../raw/code/learning-machine/llama_ref/train.py#L227)
- `_shard_first_dim(x)` — [`L217`](../../../../../raw/code/learning-machine/llama_ref/train.py#L217)
- `fake_dataloader(size, seqlen, batch_size)` — [`L36`](../../../../../raw/code/learning-machine/llama_ref/train.py#L36)
- `gather_weights(w, spec)` — [`L180`](../../../../../raw/code/learning-machine/llama_ref/train.py#L180)
- `group_data(dataloader, block_size)` — [`L41`](../../../../../raw/code/learning-machine/llama_ref/train.py#L41) — yields tuple of inputs, label with seqlen == block_size
- `loss(weights, args, label)` — [`L88`](../../../../../raw/code/learning-machine/llama_ref/train.py#L88)
- `make_train_step(model_forward, loss_fn, optax_optimizer, policy)` — [`L81`](../../../../../raw/code/learning-machine/llama_ref/train.py#L81)
- `model_forward_shmap(weight, args)` — [`L179`](../../../../../raw/code/learning-machine/llama_ref/train.py#L179)
- `sharded_device_put(tensor, sharding)` — [`L63`](../../../../../raw/code/learning-machine/llama_ref/train.py#L63)
- `step(weights, opt_state, args, label)` — [`L102`](../../../../../raw/code/learning-machine/llama_ref/train.py#L102)
- `train_loop(mesh, model, weights, data_loader, input_freqs_cis, lr, seqlen, policy, batch_size, use_shmap, profile_dir: str)` — [`L145`](../../../../../raw/code/learning-machine/llama_ref/train.py#L145)

## Module values
- `SEQLEN` — [`L13`](../../../../../raw/code/learning-machine/llama_ref/train.py#L13)
- `mark_sharding` — [`L79`](../../../../../raw/code/learning-machine/llama_ref/train.py#L79)
- `remat` — [`L78`](../../../../../raw/code/learning-machine/llama_ref/train.py#L78)

