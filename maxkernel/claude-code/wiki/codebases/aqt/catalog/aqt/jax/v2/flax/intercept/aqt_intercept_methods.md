---
title: 'Module: aqt/jax/v2/flax/intercept/aqt_intercept_methods.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/flax/intercept/aqt_intercept_methods.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.flax.intercept.aqt_intercept_methods`/
symbols:
  _Args: _Args.
  _Kwargs: _Kwargs.
  _NextGetter: _NextGetter.
  _Interceptor: _Interceptor.
  _DotGeneralScope: _DotGeneralScope#
  _DotGeneralScope.__init__: _DotGeneralScope#__init__().
  _DotGeneralScope.__enter__: _DotGeneralScope#__enter__().
  _DotGeneralScope.__exit__: _DotGeneralScope#__exit__().
  DotGeneralGeneratorBase: DotGeneralGeneratorBase#
  DotGeneralGeneratorBase.__call__: DotGeneralGeneratorBase#__call__().
  DotGeneralGenerator: DotGeneralGenerator#
  DotGeneralGenerator.__init__: DotGeneralGenerator#__init__().
  DotGeneralGenerator.__call__: DotGeneralGenerator#__call__().
  DotGeneralGeneratorByModule: DotGeneralGeneratorByModule#
  DotGeneralGeneratorByModule.__call__: DotGeneralGeneratorByModule#__call__().
  DotGeneralGeneratorByModule.generate_by_module: DotGeneralGeneratorByModule#generate_by_module().
  intercept_methods_replace_dot_general: intercept_methods_replace_dot_general().
  AqtDotGeneralGenerator: AqtDotGeneralGenerator#
  AqtDotGeneralGenerator.__init__: AqtDotGeneralGenerator#__init__().
  AqtDotGeneralGenerator.generate_by_module: AqtDotGeneralGenerator#generate_by_module().
  intercept_methods: intercept_methods().
  intercept_wrapper: intercept_wrapper().
---
# Module: [`aqt/jax/v2/flax/intercept/aqt_intercept_methods.py`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py)

## Classes
### `AqtDotGeneralGenerator`
- def: [`aqt/jax/v2/flax/intercept/aqt_intercept_methods.py:195`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L195)
- doc: Generate AQT dot general by module.
- signature: `class AqtDotGeneralGenerator(DotGeneralGeneratorByModule):`
- members:
  - `generate_by_module(self, module: nn.Module)` — [`L203`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L203) — Generate AQT dot general method by the intercepted module object.
- protocol/private: `__init__`[`L198`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L198)

### `DotGeneralGenerator`
- def: [`aqt/jax/v2/flax/intercept/aqt_intercept_methods.py:75`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L75)
- doc: Generate saved dot general.
- signature: `class DotGeneralGenerator(DotGeneralGeneratorBase):`
- members:
  - `__call__(self, next_f: _NextGetter, args: _Args, kwargs: _Kwargs, context: flax.linen.module.InterceptorContext)` — [`L86`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L86) — Return the dot general method passed on initializes.
  - `__init__(self, dot_general: Any)` — [`L78`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L78) — Initializes dot general generator with given dot general method.

### `DotGeneralGeneratorBase`
- def: [`aqt/jax/v2/flax/intercept/aqt_intercept_methods.py:47`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L47)
- doc: Abstract class to generate dot general.
- signature: `class DotGeneralGeneratorBase(metaclass=abc.ABCMeta):`
- members:
  - `__call__(self, next_f: _NextGetter, args: _Args, kwargs: _Kwargs, context: flax.linen.module.InterceptorContext)` — [`L55`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L55) — Generate dot general method by intercepted arguments.

### `DotGeneralGeneratorByModule`
- def: [`aqt/jax/v2/flax/intercept/aqt_intercept_methods.py:110`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L110)
- doc: Abstract class to generate dot general by module.
- signature: `class DotGeneralGeneratorByModule(DotGeneralGenerator):`
- members:
  - `__call__(self, next_f: _NextGetter, args: _Args, kwargs: _Kwargs, context: flax.linen.module.InterceptorContext)` — [`L113`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L113) — Return the dot general method by module on the context.
  - `generate_by_module(self, module: nn.Module)` — [`L136`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L136) — Generate dot general method by the intercepted module object.

### `_DotGeneralScope`
- def: [`aqt/jax/v2/flax/intercept/aqt_intercept_methods.py:34`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L34)
- signature: `class _DotGeneralScope:`
- protocol/private: `__enter__`[`L38`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L38), `__exit__`[`L43`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L43), `__init__`[`L35`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L35)

## Functions
- `intercept_methods(*args, **kwargs)` — [`L218`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L218)
- `intercept_methods_replace_dot_general(dot_general_generator: DotGeneralGeneratorBase=DotGeneralGenerator(lax.dot_general))` — [`L148`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L148) — Flax intercept method wrapper to replace dot general.
- `intercept_wrapper(func, *aqt_args, **aqt_kwargs)` — [`L223`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L223)

## Module values
- `_Args` — [`L26`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L26)
- `_Interceptor` — [`L29`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L29)
- `_Kwargs` — [`L27`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L27)
- `_NextGetter` — [`L28`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/intercept/aqt_intercept_methods.py#L28)

