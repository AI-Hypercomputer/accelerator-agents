---
title: 'Module: examples/_diffusion.py'
type: catalog
provenance: extracted
module: examples/_diffusion.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `examples._diffusion`/
symbols:
  CompiledModule.forward: CompiledModule#forward().
  CompiledModule.f: CompiledModule#f().
  CompiledModule: CompiledModule#
  CompiledModule._maybe_move_tensor: CompiledModule#_maybe_move_tensor().
  CompiledModule._make_jitted: CompiledModule#_make_jitted().
  compile_pipe: compile_pipe().
  main: main().
  CompiledModule._func_jitted_torch: CompiledModule#_func_jitted_torch.
  CompiledModule.__call__: CompiledModule#__call__().
  CompiledModule.__getattr__: CompiledModule#__getattr__().
  CompiledModule._model: CompiledModule#_model.
  CompiledModule._weights: CompiledModule#_weights.
  CompiledModule.__init__: CompiledModule#__init__().
---
# Module: [`examples/_diffusion.py`](../../../../../raw/code/torchax/examples/_diffusion.py)

## Classes
### `CompiledModule`
- def: [`examples/_diffusion.py:30`](../../../../../raw/code/torchax/examples/_diffusion.py#L30)
- signature: `class CompiledModule:`
- members:
  - `f(weights, *args, **kwargs)` — [`L58`](../../../../../raw/code/torchax/examples/_diffusion.py#L58)
  - `forward(self, *args, **kwargs)` — [`L69`](../../../../../raw/code/torchax/examples/_diffusion.py#L69)
- protocol/private: `__call__`[`L75`](../../../../../raw/code/torchax/examples/_diffusion.py#L75), `__getattr__`[`L78`](../../../../../raw/code/torchax/examples/_diffusion.py#L78), `__init__`[`L31`](../../../../../raw/code/torchax/examples/_diffusion.py#L31), `_func_jitted_torch`[`L39`](../../../../../raw/code/torchax/examples/_diffusion.py#L39), `_make_jitted`[`L48`](../../../../../raw/code/torchax/examples/_diffusion.py#L48), `_maybe_move_tensor`[`L41`](../../../../../raw/code/torchax/examples/_diffusion.py#L41), `_model`[`L37`](../../../../../raw/code/torchax/examples/_diffusion.py#L37), `_weights`[`L34`](../../../../../raw/code/torchax/examples/_diffusion.py#L34)
- uses (calls/refs, reference-scoped): [`Tensor`](../torchax/tensor.md#Tensor), [`XLADispatchMode`](../torchax/tensor.md#XLADispatchMode)
- used by: (1 test-only callers)

## Functions
- `compile_pipe(pipe)` — [`L82`](../../../../../raw/code/torchax/examples/_diffusion.py#L82)
- `main()` — [`L89`](../../../../../raw/code/torchax/examples/_diffusion.py#L89)

