---
title: 'Module: torchax/ops/ops_registry.py'
type: catalog
provenance: extracted
module: torchax/ops/ops_registry.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.ops.ops_registry`/
symbols:
  register_torch_dispatch_op: register_torch_dispatch_op().
  register_torch_function_op: register_torch_function_op().
  all_aten_ops.all_aten_ops: all_aten_ops.all_aten_ops.
  Operator.func: Operator#func.
  all_torch_functions.all_torch_functions: all_torch_functions.all_torch_functions.
  Operator: Operator#
  Operator.is_jax_function: Operator#is_jax_function.
  Operator.is_user_defined: Operator#is_user_defined.
  Operator.needs_env: Operator#needs_env.
  Operator.is_view_op: Operator#is_view_op.
  Operator.torch_op: Operator#torch_op.
---
# Module: [`torchax/ops/ops_registry.py`](../../../../../../raw/code/torchax/torchax/ops/ops_registry.py)

## Classes
### `Operator`
- def: [`torchax/ops/ops_registry.py:22`](../../../../../../raw/code/torchax/torchax/ops/ops_registry.py#L22) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
- signature: `class Operator:`
- members:
  - `func` — [`L24`](../../../../../../raw/code/torchax/torchax/ops/ops_registry.py#L24)
  - `is_jax_function` — [`L25`](../../../../../../raw/code/torchax/torchax/ops/ops_registry.py#L25) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
  - `is_user_defined` — [`L26`](../../../../../../raw/code/torchax/torchax/ops/ops_registry.py#L26)
  - `is_view_op` — [`L28`](../../../../../../raw/code/torchax/torchax/ops/ops_registry.py#L28) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
  - `needs_env` — [`L27`](../../../../../../raw/code/torchax/torchax/ops/ops_registry.py#L27) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)
  - `torch_op` — [`L23`](../../../../../../raw/code/torchax/torchax/ops/ops_registry.py#L23)
- uses (calls/refs, reference-scoped): [`TorchCallable`](../types.md#TorchCallable.TorchCallable), [`JaxCallable`](../types.md#JaxCallable.JaxCallable)
- used by: [`load_ops`](../tensor.md#Environment.load_ops), [`register_torch_dispatch_op`](ops_registry.md#register_torch_dispatch_op), [`register_torch_function_op`](ops_registry.md#register_torch_function_op), [`override_op_definition`](../tensor.md#Environment.override_op_definition), [`all_aten_ops`](ops_registry.md#all_aten_ops.all_aten_ops), [`call_function`](../export.md#JaxInterpreter.call_function), [`all_aten_jax_ops`](__init__.md#all_aten_jax_ops), [`all_torch_functions`](ops_registry.md#all_torch_functions.all_torch_functions)

## Functions
- `register_torch_dispatch_op(aten_op, impl_callable, is_jax_function=True, is_user_defined=False, needs_env=False, is_view_op=False)` — [`L35`](../../../../../../raw/code/torchax/torchax/ops/ops_registry.py#L35) — documented in [torchax-ops-jaten](../../../concepts/torchax-ops-jaten.md)
- `register_torch_function_op(torch_func, impl_callable, is_jax_function=True, is_user_defined=False, needs_env=False, is_view_op=False)` — [`L57`](../../../../../../raw/code/torchax/torchax/ops/ops_registry.py#L57) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)

## Module values
- `all_aten_ops` — [`L31`](../../../../../../raw/code/torchax/torchax/ops/ops_registry.py#L31) — documented in [torchax-ops-jaten](../../../concepts/torchax-ops-jaten.md)
- `all_torch_functions` — [`L32`](../../../../../../raw/code/torchax/torchax/ops/ops_registry.py#L32) — documented in [torchax-ops-jtorch](../../../concepts/torchax-ops-jtorch.md)

