---
title: 'Module: torchax/ops/jlibrary.py'
type: catalog
provenance: extracted
module: torchax/ops/jlibrary.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.ops.jlibrary`/
symbols:
  register_torch_composite._composite_impl: register_torch_composite()._composite_impl().
  register_jax_composite._composite_impl: register_jax_composite()._composite_impl().
  _jit_composite_impl: _jit_composite_impl().
  _jit_composite_impl.composite_impl: _jit_composite_impl().composite_impl().
  register_jax_composite: register_jax_composite().
  register_torch_composite: register_torch_composite().
  register_torch_composite._composite_impl.ImplWrapper: register_torch_composite()._composite_impl().ImplWrapper#
  register_torch_composite._composite_impl.jaxpr_impl: register_torch_composite()._composite_impl().jaxpr_impl().
  register_torch_composite._composite_impl.ImplWrapper.__init__: register_torch_composite()._composite_impl().ImplWrapper#__init__().
  register_torch_composite._composite_impl.ImplWrapper.forward: register_torch_composite()._composite_impl().ImplWrapper#forward().
---
# Module: [`torchax/ops/jlibrary.py`](../../../../../../raw/code/torchax/torchax/ops/jlibrary.py)

## Classes
### `ImplWrapper`  ·  implements/extends Module
- def: [`torchax/ops/jlibrary.py:77`](../../../../../../raw/code/torchax/torchax/ops/jlibrary.py#L77)
- signature: `class ImplWrapper(torch.nn.Module):`
- members:
  - `forward(self, *args)` — [`L81`](../../../../../../raw/code/torchax/torchax/ops/jlibrary.py#L81)
- protocol/private: `__init__`[`L78`](../../../../../../raw/code/torchax/torchax/ops/jlibrary.py#L78)
- used by: [`_composite_impl`](jlibrary.md#register_torch_composite._composite_impl)

## Functions
- `_composite_impl(*args)` — [`L57`](../../../../../../raw/code/torchax/torchax/ops/jlibrary.py#L57)
- `_composite_impl(*args)` — [`L76`](../../../../../../raw/code/torchax/torchax/ops/jlibrary.py#L76)
- `_jit_composite_impl(composite_name, jaxpr_impl, **jit_args)` — [`L26`](../../../../../../raw/code/torchax/torchax/ops/jlibrary.py#L26) — Wrap a jaxpr in a jitted function with the proper composite name
- `composite_impl(*args)` — [`L31`](../../../../../../raw/code/torchax/torchax/ops/jlibrary.py#L31)
- `jaxpr_impl(*args)` — [`L91`](../../../../../../raw/code/torchax/torchax/ops/jlibrary.py#L91)
- `register_jax_composite(composite_name, impl, *ops, **jit_args)` — [`L39`](../../../../../../raw/code/torchax/torchax/ops/jlibrary.py#L39) — Register a composite using a JAX implementation.
- `register_torch_composite(composite_name, impl, *ops, **jit_args)` — [`L61`](../../../../../../raw/code/torchax/torchax/ops/jlibrary.py#L61) — Register a torch decomposition as a composite.

