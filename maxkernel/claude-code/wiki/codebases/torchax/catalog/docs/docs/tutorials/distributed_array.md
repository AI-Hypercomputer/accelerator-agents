---
title: 'Module: docs/docs/tutorials/distributed_array.py'
type: catalog
provenance: extracted
module: docs/docs/tutorials/distributed_array.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `docs.docs.tutorials.distributed_array`/
symbols:
  y: y.
  tensor: tensor.
  visualize_array_sharding: visualize_array_sharding.
  x: x.
  permuted_devices: permuted_devices.
  updates: updates.
  params: params.
  z: z.
  jax_device_put: jax_device_put.
  mesh: mesh.
  batch: batch.
  x_single: x_single.
  sharding2: sharding2.
  params_single: params_single.
  w: w.
  model: model.
  inputs: inputs.
  grads: grads.
  loss_jit: loss_jit.
  opt_state: opt_state.
  replicated_sharding: replicated_sharding.
  mesh_sharding: mesh_sharding().
  w_copy: w_copy.
  e: e.
  optimizer: optimizer.
  loss_fun: loss_fun().
  name_to_sharding: name_to_sharding.
  f: f().
  batch_single: batch_single.
  grad_fn_jit: grad_fn_jit.
  sharding1: sharding1.
  layer_sizes: layer_sizes.
  pure_model_fun: pure_model_fun().
  grad_fn: grad_fn.
  sharding: sharding.
  batch_size: batch_size.
  step_size: step_size.
  i: i.
  loss: loss.
  Model.forward: Model#forward().
  print_exception: print_exception().
  devices: devices.
  default_mesh: default_mesh.
  Model: Model#
  Model.layers: Model#layers.
  torch_loss: torch_loss.
  name: name.
  Model.__init__: Model#__init__().
---
# Module: [`docs/docs/tutorials/distributed_array.py`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py)

## Classes
### `Model`  ·  implements/extends Module
- def: [`docs/docs/tutorials/distributed_array.py:404`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L404)
- signature: `class Model(torch.nn.Module):`
- members:
  - `forward(self, x)` — [`L413`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L413)
  - `layers` — [`L409`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L409)
- protocol/private: `__init__`[`L406`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L406)
- used by: [`model`](distributed_array.md#model)

## Functions
- `f(x)` — [`L346`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L346)
- `loss_fun(weight, batch)` — [`L464`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L464) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `mesh_sharding(pspec: PartitionSpec, mesh: Optional[Mesh] = None)` — [`L162`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L162)
- `print_exception(e)` — [`L290`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L290)
- `pure_model_fun(weights, inputs)` — [`L459`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L459)

## Module values
- `batch` — [`L485`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L485)
- `batch_single` — [`L518`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L518)
- `batch_size` — [`L443`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L443)
- `default_mesh` — [`L160`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L160)
- `devices` — [`L305`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L305)
- `e` — [`L302`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L302)
- `grad_fn` — [`L470`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L470)
- `grad_fn_jit` — [`L471`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L471) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `grads` — [`L507`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L507) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `i` — [`L506`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L506) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `inputs` — [`L450`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L450)
- `jax_device_put` — [`L488`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L488)
- `layer_sizes` — [`L442`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L442)
- `loss` — [`L507`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L507) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `loss_jit` — [`L472`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L472)
- `mesh` — [`L79`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L79) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `model` — [`L448`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L448) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `name` — [`L569`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L569)
- `name_to_sharding` — [`L558`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L558) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `opt_state` — [`L503`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L503) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `optimizer` — [`L501`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L501)
- `params` — [`L456`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L456) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `params_single` — [`L519`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L519) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `permuted_devices` — [`L306`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L306)
- `replicated_sharding` — [`L482`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L482) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `sharding` — [`L481`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L481)
- `sharding1` — [`L296`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L296)
- `sharding2` — [`L297`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L297) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `step_size` — [`L497`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L497)
- `tensor` — [`L569`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L569)
- `torch_loss` — [`L462`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L462)
- `updates` — [`L508`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L508) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `visualize_array_sharding` — [`L93`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L93) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `w` — [`L247`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L247) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `w_copy` — [`L274`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L274)
- `x` — [`L84`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L84) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `x_single` — [`L257`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L257)
- `y` — [`L89`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L89) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)
- `z` — [`L102`](../../../../../../../raw/code/torchax/docs/docs/tutorials/distributed_array.py#L102) — documented in [docs-docs-tutorials-distributed_array](../../../../concepts/docs-docs-tutorials-distributed_array.md)

