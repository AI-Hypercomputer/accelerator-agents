---
title: 'Module: torchax/interop.py'
type: catalog
provenance: extracted
module: torchax/interop.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.interop`/
symbols:
  jax_view: jax_view.
  call_jax: call_jax().
  torch_view: torch_view.
  _jax_view: _jax_view().
  JittableModule.params: JittableModule#params.
  _torch_view: _torch_view().
  call_torch: call_torch().
  JittableModule.jittable_call: JittableModule#jittable_call().
  JittableModule.make_jitted: JittableModule#make_jitted().
  jax_jit: jax_jit().
  JittableModule: JittableModule#
  JittableModule.functional_call: JittableModule#functional_call().
  wrap_jax_jit: wrap_jax_jit().
  JittableModule._extra_dumped_weights: JittableModule#_extra_dumped_weights.
  j2t_autograd: j2t_autograd().
  JittableModule._jitted: JittableModule#_jitted.
  JittableModule._model: JittableModule#_model.
  JittableModule.__getattr__: JittableModule#__getattr__().
  j2t_autograd.inner.JaxFun.forward: j2t_autograd().inner().JaxFun#forward().
  extract_all_buffers: extract_all_buffers().
  extract_all_buffers.extract_one: extract_all_buffers().extract_one().
  set_all_buffers.set_one: set_all_buffers().set_one().
  jax_value_and_grad: jax_value_and_grad().
  _T: _T.
  torch_view_elem: torch_view_elem().
  jax_view_elem: jax_view_elem().
  set_all_buffers: set_all_buffers().
  JittableModule.jitted_forward: JittableModule#jitted_forward().
  JittableModule.forward: JittableModule#forward().
  JittableModule.call: JittableModule#call().
  j2t_autograd._jax_forward: j2t_autograd()._jax_forward().
  j2t_autograd._jax_forward.fn_wrapper: j2t_autograd()._jax_forward().fn_wrapper().
  j2t_autograd.inner: j2t_autograd().inner().
  gradient_checkpoint: gradient_checkpoint().
  JittableModule.__class__: JittableModule#__class__().
  JittableModule.__call__: JittableModule#__call__().
  j2t_autograd.inner.JaxFun.backward: j2t_autograd().inner().JaxFun#backward().
  fori_loop: fori_loop.
  jax_shard_map: jax_shard_map().
  JittableModule._extra_jit_args: JittableModule#_extra_jit_args.
  j2t_autograd._jax_backward: j2t_autograd()._jax_backward().
  j2t_autograd.inner.JaxFun: j2t_autograd().inner().JaxFun#
  JittableModule.__init__: JittableModule#__init__().
---
# Module: [`torchax/interop.py`](../../../../../raw/code/torchax/torchax/interop.py)

## Classes
### `JaxFun`  ·  implements/extends Function
- def: [`torchax/interop.py:333`](../../../../../raw/code/torchax/torchax/interop.py#L333)
- signature: `class JaxFun(torch.autograd.Function):`
- members:
  - `backward(ctx, *grad_out)` — [`L352`](../../../../../raw/code/torchax/torchax/interop.py#L352)
  - `forward(ctx, tree_def, *flat_args_kwargs)` — [`L335`](../../../../../raw/code/torchax/torchax/interop.py#L335)
- uses (calls/refs, reference-scoped): [`partition`](util.md#partition), [`_jax_forward`](interop.md#j2t_autograd._jax_forward), [`_jax_backward`](interop.md#j2t_autograd._jax_backward)
- used by: [`inner`](interop.md#j2t_autograd.inner)

### `JittableModule`  ·  implements/extends Module
- def: [`torchax/interop.py:79`](../../../../../raw/code/torchax/torchax/interop.py#L79) — documented in [torchax](../../concepts/torchax.md)
- signature: `class JittableModule(torch.nn.Module):`
- members:
  - `call(*args, **kwargs)` — [`L164`](../../../../../raw/code/torchax/torchax/interop.py#L164)
  - `forward(self, *args, **kwargs)` — [`L148`](../../../../../raw/code/torchax/torchax/interop.py#L148)
  - `functional_call(self, method_or_name, params, buffers, *args, **kwargs)` — [`L113`](../../../../../raw/code/torchax/torchax/interop.py#L113)
  - `jittable_call(self, method_name: str, *args, **kwargs)` — [`L135`](../../../../../raw/code/torchax/torchax/interop.py#L135)
  - `jitted_forward(*args, **kwargs)` — [`L142`](../../../../../raw/code/torchax/torchax/interop.py#L142)
  - `make_jitted(self, key)` — [`L158`](../../../../../raw/code/torchax/torchax/interop.py#L158)
  - `params` — [`L84`](../../../../../raw/code/torchax/torchax/interop.py#L84)
- protocol/private: `__call__`[`L110`](../../../../../raw/code/torchax/torchax/interop.py#L110), `__class__`[`L105`](../../../../../raw/code/torchax/torchax/interop.py#L105), `__getattr__`[`L151`](../../../../../raw/code/torchax/torchax/interop.py#L151), `__init__`[`L80`](../../../../../raw/code/torchax/torchax/interop.py#L80), `_extra_dumped_weights`[`L90`](../../../../../raw/code/torchax/torchax/interop.py#L90), `_extra_jit_args`[`L88`](../../../../../raw/code/torchax/torchax/interop.py#L88), `_jitted`[`L86`](../../../../../raw/code/torchax/torchax/interop.py#L86), `_model`[`L85`](../../../../../raw/code/torchax/torchax/interop.py#L85)
- uses (calls/refs, reference-scoped): [`jax_jit`](interop.md#jax_jit), [`extract_all_buffers`](interop.md#extract_all_buffers)
- used by: [`extract_jax`](__init__.md#extract_jax), [`compile`](__init__.md#compile), [`jax_func`](__init__.md#extract_jax.jax_func)  (13 test-only)

## Functions
- `_jax_backward(vjp_spec, saved_tensors, grad_out)` — [`L319`](../../../../../raw/code/torchax/torchax/interop.py#L319) — JAX function to compute input gradients.
- `_jax_forward(fn, other, tree_def, tensors)` — [`L303`](../../../../../raw/code/torchax/torchax/interop.py#L303) — JAX function to compute output and vjp function.
- `_jax_view(t: TorchValue)` — [`L220`](../../../../../raw/code/torchax/torchax/interop.py#L220) — documented in [docs-docs-tutorials-trainingyt](../../concepts/docs-docs-tutorials-trainingyt.md)
- `_torch_view(t: JaxValue)` — [`L170`](../../../../../raw/code/torchax/torchax/interop.py#L170) — documented in [torchax-interop](../../concepts/torchax-interop.md)
- `call_jax(jax_func: JaxCallable, *args: TorchValue, **kwargs: TorchValue)` — [`L273`](../../../../../raw/code/torchax/torchax/interop.py#L273) — documented in [docs-docs-tutorials-trainingyt](../../concepts/docs-docs-tutorials-trainingyt.md)
- `call_torch(torch_func: TorchCallable, *args: JaxValue, **kwargs: JaxValue)` — [`L281`](../../../../../raw/code/torchax/torchax/interop.py#L281) — documented in [torchax-interop](../../concepts/torchax-interop.md)
- `extract_all_buffers(m: torch.nn.Module)` — [`L42`](../../../../../raw/code/torchax/torchax/interop.py#L42)
- `extract_one(module, prefix)` — [`L46`](../../../../../raw/code/torchax/torchax/interop.py#L46)
- `fn_wrapper(*tensors)` — [`L311`](../../../../../raw/code/torchax/torchax/interop.py#L311)
- `gradient_checkpoint(torch_function, kwargs=None)` — [`L413`](../../../../../raw/code/torchax/torchax/interop.py#L413)
- `inner(*args, **kwargs)` — [`L330`](../../../../../raw/code/torchax/torchax/interop.py#L330)
- `j2t_autograd(fn, call_jax=call_jax)` — [`L290`](../../../../../raw/code/torchax/torchax/interop.py#L290) — Given a JAX function, returns a PyTorch autograd function implemented with `jax.vjp(fn)`. — documented in [torchax-interop](../../concepts/torchax-interop.md)
- `jax_jit(torch_function, kwargs_for_jax_jit=None, fix_for_buffer_donation=False)` — [`L393`](../../../../../raw/code/torchax/torchax/interop.py#L393) — documented in [torchax](../../concepts/torchax.md)
- `jax_shard_map(torch_function, kwargs_for_jax_shard_map=None)` — [`L399`](../../../../../raw/code/torchax/torchax/interop.py#L399)
- `jax_value_and_grad(torch_function, kwargs_for_value_and_grad=None)` — [`L405`](../../../../../raw/code/torchax/torchax/interop.py#L405) — documented in [docs-docs-tutorials-trainingyt](../../concepts/docs-docs-tutorials-trainingyt.md)
- `jax_view_elem(v: torch.Tensor)` — [`L244`](../../../../../raw/code/torchax/torchax/interop.py#L244)
- `set_all_buffers(m, params, buffers)` — [`L64`](../../../../../raw/code/torchax/torchax/interop.py#L64)
- `set_one(module, prefix)` — [`L65`](../../../../../raw/code/torchax/torchax/interop.py#L65)
- `torch_view_elem(v: jax.Array)` — [`L191`](../../../../../raw/code/torchax/torchax/interop.py#L191)
- `wrap_jax_jit(torch_function, jax_jit_func=jax.jit, kwargs_for_jax=None)` — [`L386`](../../../../../raw/code/torchax/torchax/interop.py#L386) — documented in [docs-docs-tutorials-trainingyt](../../concepts/docs-docs-tutorials-trainingyt.md)

## Module values
- `_T` — [`L33`](../../../../../raw/code/torchax/torchax/interop.py#L33)
- `fori_loop` — [`L383`](../../../../../raw/code/torchax/torchax/interop.py#L383) — documented in [torchax-interop](../../concepts/torchax-interop.md)
- `jax_view` — [`L236`](../../../../../raw/code/torchax/torchax/interop.py#L236) — documented in [docs-docs-tutorials-trainingyt](../../concepts/docs-docs-tutorials-trainingyt.md)
- `torch_view` — [`L183`](../../../../../raw/code/torchax/torchax/interop.py#L183) — documented in [docs-docs-tutorials-trainingyt](../../concepts/docs-docs-tutorials-trainingyt.md)

