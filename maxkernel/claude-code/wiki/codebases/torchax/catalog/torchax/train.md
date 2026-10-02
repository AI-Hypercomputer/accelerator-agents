---
title: 'Module: torchax/train.py'
type: catalog
provenance: extracted
module: torchax/train.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.train`/
symbols:
  ScannedModule.forward: ScannedModule#forward().
  make_train_step: make_train_step().
  ScannedModule.c: ScannedModule#c.
  make_train_step.step: make_train_step().step().
  ScannedModule.checkpoint_policy: ScannedModule#checkpoint_policy.
  ScannedModule.layer_weights_keys: ScannedModule#layer_weights_keys.
  ScannedModule.params: ScannedModule#params.
  ScannedModule.eval_one_layer: ScannedModule#eval_one_layer().
  remat: remat.
  mark_sharding: mark_sharding.
  ScannedModule._param_name_new: ScannedModule#_param_name_new().
  make_train_step.loss: make_train_step().loss().
  Container: Container#
  ScannedModule: ScannedModule#
  ScannedModule._stack_layer_weights: ScannedModule#_stack_layer_weights().
  ScannedModule.__init__: ScannedModule#__init__().
  ScannedModule._param_name_old: ScannedModule#_param_name_old().
---
# Module: [`torchax/train.py`](../../../../../raw/code/torchax/torchax/train.py)

## Classes
### `Container`
- def: [`torchax/train.py:72`](../../../../../raw/code/torchax/torchax/train.py#L72)
- signature: `class Container:`
- used by: [`c`](train.md#ScannedModule.c)

### `ScannedModule`  ·  implements/extends Module
- def: [`torchax/train.py:76`](../../../../../raw/code/torchax/torchax/train.py#L76)
- signature: `class ScannedModule(torch.nn.Module):`
- members:
  - `eval_one_layer(args, weight)` — [`L114`](../../../../../raw/code/torchax/torchax/train.py#L114)
  - `forward(self, *args, **kwargs)` — [`L109`](../../../../../raw/code/torchax/torchax/train.py#L109)
  - `c` — [`L80`](../../../../../raw/code/torchax/torchax/train.py#L80)
  - `checkpoint_policy` — [`L84`](../../../../../raw/code/torchax/torchax/train.py#L84)
  - `layer_weights_keys` — [`L87`](../../../../../raw/code/torchax/torchax/train.py#L87)
  - `params` — [`L88`](../../../../../raw/code/torchax/torchax/train.py#L88)
- protocol/private: `__init__`[`L77`](../../../../../raw/code/torchax/torchax/train.py#L77), `_param_name_new`[`L103`](../../../../../raw/code/torchax/torchax/train.py#L103), `_param_name_old`[`L106`](../../../../../raw/code/torchax/torchax/train.py#L106), `_stack_layer_weights`[`L92`](../../../../../raw/code/torchax/torchax/train.py#L92)
- uses (calls/refs, reference-scoped): [`torch_view`](interop.md#torch_view), [`gradient_checkpoint`](interop.md#gradient_checkpoint), [`Container`](train.md#Container)
- used by: (1 test-only callers)

## Functions
- `loss(weights, buffers, args, label)` — [`L48`](../../../../../raw/code/torchax/torchax/train.py#L48)
- `make_train_step(model_fn, loss_fn, optax_optimizer, remat_policy=None)` — [`L29`](../../../../../raw/code/torchax/torchax/train.py#L29) — Make a function that do one train step given model and loss. — documented in [docs-docs-tutorials-trainingyt](../../concepts/docs-docs-tutorials-trainingyt.md)
- `step(weights, buffers, opt_state, args, label)` — [`L57`](../../../../../raw/code/torchax/torchax/train.py#L57)

## Module values
- `mark_sharding` — [`L26`](../../../../../raw/code/torchax/torchax/train.py#L26) — documented in [torchax-interop](../../concepts/torchax-interop.md)
- `remat` — [`L25`](../../../../../raw/code/torchax/torchax/train.py#L25) — documented in [torchax-interop](../../concepts/torchax-interop.md)

