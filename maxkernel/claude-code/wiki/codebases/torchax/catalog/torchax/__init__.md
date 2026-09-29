---
title: 'Module: torchax/__init__.py'
type: catalog
provenance: extracted
module: torchax/__init__.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 torchax/
symbols:
  default_env: default_env().
  extract_jax: extract_jax().
  compile: compile().
  enable_globally: enable_globally().
  enable_accuracy_mode: enable_accuracy_mode().
  disable_temporarily: disable_temporarily().
  extract_jax.jax_func: extract_jax().jax_func().
  _env._env: _env._env.
  enable_performance_mode: enable_performance_mode().
  disable_globally: disable_globally().
  CompileOptions.mode: CompileOptions#mode.
  VERSION: VERSION.
  CompileOptions: CompileOptions#
  __version__: __version__.
  _env_lock: _env_lock.
  CompileOptions.methods_to_compile: CompileOptions#methods_to_compile.
  CompileOptions.jax_jit_kwargs: CompileOptions#jax_jit_kwargs.
  __all__: __all__.
  unsupported_dtype: unsupported_dtype.
---
# Module: [`torchax/__init__.py`](../../../../../raw/code/torchax/torchax/__init__.py)

## Classes
### `CompileOptions`
- def: [`torchax/__init__.py:157`](../../../../../raw/code/torchax/torchax/__init__.py#L157)
- signature: `class CompileOptions:`
- members:
  - `jax_jit_kwargs` — [`L160`](../../../../../raw/code/torchax/torchax/__init__.py#L160)
  - `methods_to_compile` — [`L159`](../../../../../raw/code/torchax/torchax/__init__.py#L159)
  - `mode` — [`L161`](../../../../../raw/code/torchax/torchax/__init__.py#L161)
- used by: [`compile`](__init__.md#compile)

## Functions
- `compile(fn, options: CompileOptions | None = None)` — [`L164`](../../../../../raw/code/torchax/torchax/__init__.py#L164) — documented in [torchax](../../concepts/torchax.md)
- `default_env()` — [`L64`](../../../../../raw/code/torchax/torchax/__init__.py#L64) — Returns the default environment. — documented in [torchax](../../concepts/torchax.md)
- `disable_globally()` — [`L121`](../../../../../raw/code/torchax/torchax/__init__.py#L121) — Disables torchax globally. — documented in [torchax](../../concepts/torchax.md)
- `disable_temporarily()` — [`L128`](../../../../../raw/code/torchax/torchax/__init__.py#L128) — documented in [torchax](../../concepts/torchax.md)
- `enable_accuracy_mode()` — [`L144`](../../../../../raw/code/torchax/torchax/__init__.py#L144)
- `enable_globally()` — [`L115`](../../../../../raw/code/torchax/torchax/__init__.py#L115) — Enables torchax globally. — documented in [docs-docs-tutorials-trainingyt](../../concepts/docs-docs-tutorials-trainingyt.md)
- `enable_performance_mode()` — [`L150`](../../../../../raw/code/torchax/torchax/__init__.py#L150)
- `extract_jax(mod: torch.nn.Module, env=None, *, dedup_parameters=True)` — [`L87`](../../../../../raw/code/torchax/torchax/__init__.py#L87) — Returns a pytree of jax.ndarray and a jax callable. — documented in [torchax](../../concepts/torchax.md)
- `jax_func(states, args, kwargs=None)` — [`L98`](../../../../../raw/code/torchax/torchax/__init__.py#L98)

## Module values
- `VERSION` — [`L33`](../../../../../raw/code/torchax/torchax/__init__.py#L33)
- `__all__` — [`L40`](../../../../../raw/code/torchax/torchax/__init__.py#L40)
- `__version__` — [`L32`](../../../../../raw/code/torchax/torchax/__init__.py#L32)
- `_env` — [`L60`](../../../../../raw/code/torchax/torchax/__init__.py#L60)
- `_env_lock` — [`L61`](../../../../../raw/code/torchax/torchax/__init__.py#L61)
- `unsupported_dtype` — [`L138`](../../../../../raw/code/torchax/torchax/__init__.py#L138)

