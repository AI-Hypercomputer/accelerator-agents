---
title: 'Module: examples/train_llama_torchtitan/train_llama.py'
type: catalog
provenance: extracted
module: examples/train_llama_torchtitan/train_llama.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `examples.train_llama_torchtitan.train_llama`/
symbols:
  main: main().
  Trainer.fit: Trainer#fit().
  create_sharded_weights: create_sharded_weights().
  sharded_device_put: sharded_device_put().
  TransfomerWithScan.forward: TransfomerWithScan#forward().
  Trainer.x_sharding: Trainer#x_sharding.
  main.custom_attention: main().custom_attention().
  P: P.
  Trainer.replicated: Trainer#replicated.
  Trainer.model_fn: Trainer#model_fn().
  _process_sharding_name: _process_sharding_name().
  TransfomerWithScan.layers: TransfomerWithScan#layers.
  num_global_devices: num_global_devices.
  Trainer.mesh: Trainer#mesh.
  num_local_devices: num_local_devices.
  TransfomerWithScan.tok_embeddings: TransfomerWithScan#tok_embeddings.
  TransfomerWithScan.norm: TransfomerWithScan#norm.
  TransfomerWithScan.output: TransfomerWithScan#output.
  sharding_map_original: sharding_map_original.
  sharding_map_scan: sharding_map_scan.
  Trainer: Trainer#
  _process_sharding_name.is_integer: _process_sharding_name().is_integer().
  _make_weight_shard: _make_weight_shard().
  fake_dataloader: fake_dataloader().
  main.loss_fn: main().loss_fn().
  TransfomerWithScan: TransfomerWithScan#
  sharding_map_scan_fsdp: sharding_map_scan_fsdp.
  Trainer.__init__: Trainer#__init__().
  TransfomerWithScan.__init__: TransfomerWithScan#__init__().
---
# Module: [`examples/train_llama_torchtitan/train_llama.py`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py)

## Classes
### `Trainer`
- def: [`examples/train_llama_torchtitan/train_llama.py:177`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L177)
- signature: `class Trainer:`
- members:
  - `fit(self, model, loss_fn, data_loader, train_steps=25)` — [`L183`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L183)
  - `model_fn(weights, buffers, args)` — [`L194`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L194)
  - `mesh` — [`L179`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L179)
  - `replicated` — [`L181`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L181)
  - `x_sharding` — [`L180`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L180)
- protocol/private: `__init__`[`L178`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L178)
- uses (calls/refs, reference-scoped): [`default_env`](../../torchax/__init__.md#default_env), [`jax_view`](../../torchax/interop.md#jax_view), [`call_jax`](../../torchax/interop.md#call_jax), [`torch_view`](../../torchax/interop.md#torch_view), [`params`](../../torchax/interop.md#JittableModule.params), [`make_train_step`](../../torchax/train.md#make_train_step), [`JittableModule`](../../torchax/interop.md#JittableModule), [`functional_call`](../../torchax/interop.md#JittableModule.functional_call)  (3 test-only)
- used by: (1 test-only callers)

### `TransfomerWithScan`
- def: [`examples/train_llama_torchtitan/train_llama.py:472`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L472)
- members:
  - `forward(self, tokens: torch.Tensor)` — [`L484`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L484) — Perform a forward pass through the Transformer model.
  - `layers` — [`L478`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L478)
  - `norm` — [`L476`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L476)
  - `output` — [`L477`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L477)
  - `tok_embeddings` — [`L475`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L475)
- protocol/private: `__init__`[`L473`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L473)
- uses (calls/refs, reference-scoped): [`ScannedModule`](../../torchax/train.md#ScannedModule)
- used by: (1 test-only callers)

## Functions
- `_make_weight_shard(weight_meta, slice_index)` — [`L297`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L297)
- `_process_sharding_name(name)` — [`L276`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L276) — Replace integers in param name with *.
- `create_sharded_weights(model, mesh, sharding_map)` — [`L325`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L325)
- `custom_attention(query, key, value, attn_mask=None, dropout_p=0, is_causal=False, scale=None, enable_gqa=False)` — [`L442`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L442)
- `fake_dataloader(size, seqlen, batch_size)` — [`L347`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L347)
- `is_integer(t)` — [`L282`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L282)
- `loss_fn(logits, y)` — [`L461`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L461)
- `main(model_type="8B", batch_size=8, seqlen=2048, override_num_layers=-1, use_scan=True, tp_parallelism=1, train_steps=25, tpu_num_slices=1)` — [`L353`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L353)
- `sharded_device_put(tensor: jax.Array, sharding)` — [`L47`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L47)

## Module values
- `P` — [`L41`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L41)
- `num_global_devices` — [`L43`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L43)
- `num_local_devices` — [`L44`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L44)
- `sharding_map_original` — [`L65`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L65)
- `sharding_map_scan` — [`L81`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L81)
- `sharding_map_scan_fsdp` — [`L132`](../../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py#L132)

