---
title: 'Module: torchax/config.py'
type: catalog
provenance: extracted
module: torchax/config.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.config`/Configuration#
symbols:
  Configuration.debug_print_each_op: debug_print_each_op.
  Configuration.use_dlpack_for_data_conversion: use_dlpack_for_data_conversion.
  Configuration.treat_cuda_as_jax_device: treat_cuda_as_jax_device.
  Configuration.internal_respect_torch_return_dtypes: internal_respect_torch_return_dtypes.
  Configuration: ''
  Configuration.debug_accuracy_for_each_op: debug_accuracy_for_each_op.
  Configuration.debug_mixed_tensor: debug_mixed_tensor.
  Configuration.allow_mixed_math_with_scalar_tensor: allow_mixed_math_with_scalar_tensor.
  Configuration.force_materialize_views: force_materialize_views.
  Configuration.debug_print_each_op_operands: debug_print_each_op_operands.
  Configuration.use_int32_for_index: use_int32_for_index.
---
# Module: [`torchax/config.py`](../../../../../raw/code/torchax/torchax/config.py)

## Classes
### `Configuration`
- def: [`torchax/config.py:19`](../../../../../raw/code/torchax/torchax/config.py#L19)
- signature: `class Configuration:`
- members:
  - `allow_mixed_math_with_scalar_tensor` — [`L30`](../../../../../raw/code/torchax/torchax/config.py#L30)
  - `debug_accuracy_for_each_op` — [`L21`](../../../../../raw/code/torchax/torchax/config.py#L21)
  - `debug_mixed_tensor` — [`L22`](../../../../../raw/code/torchax/torchax/config.py#L22)
  - `debug_print_each_op` — [`L20`](../../../../../raw/code/torchax/torchax/config.py#L20)
  - `debug_print_each_op_operands` — [`L23`](../../../../../raw/code/torchax/torchax/config.py#L23)
  - `force_materialize_views` — [`L33`](../../../../../raw/code/torchax/torchax/config.py#L33)
  - `internal_respect_torch_return_dtypes` — [`L40`](../../../../../raw/code/torchax/torchax/config.py#L40)
  - `treat_cuda_as_jax_device` — [`L39`](../../../../../raw/code/torchax/torchax/config.py#L39)
  - `use_dlpack_for_data_conversion` — [`L36`](../../../../../raw/code/torchax/torchax/config.py#L36)
  - `use_int32_for_index` — [`L25`](../../../../../raw/code/torchax/torchax/config.py#L25)
- used by: [`dispatch`](tensor.md#Environment.dispatch), [`t2j_copy`](tensor.md#Environment.t2j_copy), [`config`](tensor.md#Environment.config), [`j2t_copy`](tensor.md#Environment.j2t_copy), [`enable_accuracy_mode`](__init__.md#enable_accuracy_mode), [`to_jax`](tensor.md#Environment.to_jax), [`_should_use_torchax_tensor`](tensor.md#Environment._should_use_torchax_tensor), [`enable_performance_mode`](__init__.md#enable_performance_mode), [`_is_same_device`](tensor.md#Environment._is_same_device)  (3 test-only)

