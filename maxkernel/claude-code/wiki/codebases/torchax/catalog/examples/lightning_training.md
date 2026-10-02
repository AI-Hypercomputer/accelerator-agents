---
title: 'Module: examples/lightning_training.py'
type: catalog
provenance: extracted
module: examples/lightning_training.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `examples.lightning_training`/
symbols:
  trainer_jax: trainer_jax.
  JaxTrainer.fit: JaxTrainer#fit().
  JaxTrainer.lightning_mod_loss: JaxTrainer#lightning_mod_loss().
  LitAutoEncoder.training_step: LitAutoEncoder#training_step().
  trainer: trainer.
  encoder: encoder.
  decoder: decoder.
  LitAutoEncoder: LitAutoEncoder#
  LitAutoEncoder.encoder: LitAutoEncoder#encoder.
  LitAutoEncoder.decoder: LitAutoEncoder#decoder.
  dataset: dataset.
  JaxTrainer: JaxTrainer#
  JaxTrainer.torch_opt_to_jax_opt: JaxTrainer#torch_opt_to_jax_opt().
  LitAutoEncoder.__init__: LitAutoEncoder#__init__().
  LitAutoEncoder.configure_optimizers: LitAutoEncoder#configure_optimizers().
  JaxTrainer.__init__: JaxTrainer#__init__().
---
# Module: [`examples/lightning_training.py`](../../../../../raw/code/torchax/examples/lightning_training.py)

## Classes
### `JaxTrainer`
- def: [`examples/lightning_training.py:55`](../../../../../raw/code/torchax/examples/lightning_training.py#L55)
- signature: `class JaxTrainer:`
- members:
  - `fit(self, lightning_mod, data_loader)` — [`L63`](../../../../../raw/code/torchax/examples/lightning_training.py#L63)
  - `lightning_mod_loss(weights: jax.Array, data: jax.Array, batch_id)` — [`L66`](../../../../../raw/code/torchax/examples/lightning_training.py#L66) — returns loss
  - `torch_opt_to_jax_opt(self, torch_opt)` — [`L59`](../../../../../raw/code/torchax/examples/lightning_training.py#L59)
- protocol/private: `__init__`[`L56`](../../../../../raw/code/torchax/examples/lightning_training.py#L56)
- uses (calls/refs, reference-scoped): [`default_env`](../torchax/__init__.md#default_env), [`jax_view`](../torchax/interop.md#jax_view), [`torch_view`](../torchax/interop.md#torch_view), [`to_xla`](../torchax/tensor.md#Environment.to_xla)
- used by: (1 test-only callers)

### `LitAutoEncoder`
- def: [`examples/lightning_training.py:22`](../../../../../raw/code/torchax/examples/lightning_training.py#L22)
- signature: `class LitAutoEncoder(L.LightningModule):`
- members:
  - `configure_optimizers(self)` — [`L36`](../../../../../raw/code/torchax/examples/lightning_training.py#L36)
  - `training_step(self, batch, batch_idx)` — [`L27`](../../../../../raw/code/torchax/examples/lightning_training.py#L27)
  - `decoder` — [`L25`](../../../../../raw/code/torchax/examples/lightning_training.py#L25)
  - `encoder` — [`L25`](../../../../../raw/code/torchax/examples/lightning_training.py#L25)
- protocol/private: `__init__`[`L23`](../../../../../raw/code/torchax/examples/lightning_training.py#L23)
- used by: (1 test-only callers)

## Module values
- `dataset` — [`L40`](../../../../../raw/code/torchax/examples/lightning_training.py#L40)
- `decoder` — [`L19`](../../../../../raw/code/torchax/examples/lightning_training.py#L19)
- `encoder` — [`L18`](../../../../../raw/code/torchax/examples/lightning_training.py#L18)
- `trainer` — [`L43`](../../../../../raw/code/torchax/examples/lightning_training.py#L43)
- `trainer_jax` — [`L89`](../../../../../raw/code/torchax/examples/lightning_training.py#L89)

