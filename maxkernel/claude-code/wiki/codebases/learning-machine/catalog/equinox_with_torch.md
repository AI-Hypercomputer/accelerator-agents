---
title: 'Module: equinox_with_torch.py'
type: catalog
provenance: extracted
module: equinox_with_torch.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 equinox_with_torch/
symbols:
  loss: loss.
  HybridModel.__init__: HybridModel#__init__().
  HybridModel.__call__: HybridModel#__call__().
  model: model.
  opt_state: opt_state.
  linear_before_training: linear_before_training.
  y_jax: y_jax.
  linear_after_training: linear_after_training.
  train_step: train_step().
  HybridModel: HybridModel#
  HybridModel.torch_weights: HybridModel#torch_weights.
  train_dataset: train_dataset.
  train_loader: train_loader.
  loss_fn: loss_fn().
  x_jax: x_jax.
  y_torch: y_torch.
  batch_idx: batch_idx.
  TorchPart.forward: TorchPart#forward().
  HybridModel.jax_linear1: HybridModel#jax_linear1.
  HybridModel.jax_linear2: HybridModel#jax_linear2.
  HybridModel._func_call_torch: HybridModel#_func_call_torch.
  optimizer: optimizer.
  transform: transform.
  TorchPart: TorchPart#
  TorchPart.linear: TorchPart#linear.
  HybridModel._call_torch_functional: HybridModel#_call_torch_functional().
  torchax_filter: torchax_filter().
  main_key: main_key.
  x_torch: x_torch.
  TorchPart.__init__: TorchPart#__init__().
  epoch: epoch.
---
# Module: [`equinox_with_torch.py`](../../../../raw/code/learning-machine/equinox_with_torch.py)

## Classes
### `HybridModel`
- def: [`equinox_with_torch.py:36`](../../../../raw/code/learning-machine/equinox_with_torch.py#L36) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)
- signature: `class HybridModel(eqx.Module):`
- members:
  - `jax_linear1` — [`L37`](../../../../raw/code/learning-machine/equinox_with_torch.py#L37) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)
  - `jax_linear2` — [`L39`](../../../../raw/code/learning-machine/equinox_with_torch.py#L39) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)
  - `torch_weights` — [`L40`](../../../../raw/code/learning-machine/equinox_with_torch.py#L40) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)
- protocol/private: `__call__`[`L60`](../../../../raw/code/learning-machine/equinox_with_torch.py#L60), `__init__`[`L44`](../../../../raw/code/learning-machine/equinox_with_torch.py#L44), `_call_torch_functional`[`L52`](../../../../raw/code/learning-machine/equinox_with_torch.py#L52), `_func_call_torch`[`L41`](../../../../raw/code/learning-machine/equinox_with_torch.py#L41)
- uses (calls/refs, reference-scoped): [`TorchPart`](equinox_with_torch.md#TorchPart)
- used by: [`model`](equinox_with_torch.md#model), [`linear_after_training`](equinox_with_torch.md#linear_after_training), [`linear_before_training`](equinox_with_torch.md#linear_before_training), [`train_step`](equinox_with_torch.md#train_step), [`loss_fn`](equinox_with_torch.md#loss_fn)

### `TorchPart`  ·  implements/extends Module
- def: [`equinox_with_torch.py:27`](../../../../raw/code/learning-machine/equinox_with_torch.py#L27) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)
- signature: `class TorchPart(torch.nn.Module):`
- members:
  - `forward(self, x)` — [`L32`](../../../../raw/code/learning-machine/equinox_with_torch.py#L32)
  - `linear` — [`L30`](../../../../raw/code/learning-machine/equinox_with_torch.py#L30)
- protocol/private: `__init__`[`L28`](../../../../raw/code/learning-machine/equinox_with_torch.py#L28)
- used by: [`__init__`](equinox_with_torch.md#HybridModel.__init__)

## Functions
- `loss_fn(model: HybridModel, x: jnp.ndarray, y: jnp.ndarray)` — [`L70`](../../../../raw/code/learning-machine/equinox_with_torch.py#L70) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)
- `torchax_filter(element: Any)` — [`L77`](../../../../raw/code/learning-machine/equinox_with_torch.py#L77) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)
- `train_step(model: HybridModel, opt_state: Any, optimizer: optax.GradientTransformation, x: jnp.ndarray, y: jnp.ndarray)` — [`L86`](../../../../raw/code/learning-machine/equinox_with_torch.py#L86) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)

## Module values
- `batch_idx` — [`L111`](../../../../raw/code/learning-machine/equinox_with_torch.py#L111)
- `epoch` — [`L110`](../../../../raw/code/learning-machine/equinox_with_torch.py#L110)
- `linear_after_training` — [`L124`](../../../../raw/code/learning-machine/equinox_with_torch.py#L124)
- `linear_before_training` — [`L106`](../../../../raw/code/learning-machine/equinox_with_torch.py#L106)
- `loss` — [`L115`](../../../../raw/code/learning-machine/equinox_with_torch.py#L115) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)
- `main_key` — [`L100`](../../../../raw/code/learning-machine/equinox_with_torch.py#L100)
- `model` — [`L101`](../../../../raw/code/learning-machine/equinox_with_torch.py#L101)
- `opt_state` — [`L104`](../../../../raw/code/learning-machine/equinox_with_torch.py#L104)
- `optimizer` — [`L103`](../../../../raw/code/learning-machine/equinox_with_torch.py#L103) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)
- `train_dataset` — [`L19`](../../../../raw/code/learning-machine/equinox_with_torch.py#L19) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)
- `train_loader` — [`L22`](../../../../raw/code/learning-machine/equinox_with_torch.py#L22) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)
- `transform` — [`L16`](../../../../raw/code/learning-machine/equinox_with_torch.py#L16)
- `x_jax` — [`L112`](../../../../raw/code/learning-machine/equinox_with_torch.py#L112) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)
- `x_torch` — [`L111`](../../../../raw/code/learning-machine/equinox_with_torch.py#L111) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)
- `y_jax` — [`L113`](../../../../raw/code/learning-machine/equinox_with_torch.py#L113) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)
- `y_torch` — [`L111`](../../../../raw/code/learning-machine/equinox_with_torch.py#L111) — documented in [equinox_with_torch](../concepts/equinox_with_torch.md)

