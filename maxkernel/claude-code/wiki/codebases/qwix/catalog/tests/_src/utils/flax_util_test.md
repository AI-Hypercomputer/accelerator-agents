---
title: 'Module: tests/_src/utils/flax_util_test.py'
type: catalog
provenance: extracted
module: tests/_src/utils/flax_util_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.utils.flax_util_test`/
symbols:
  FlaxUtilTest.test_get_current_module.Bar.__call__: FlaxUtilTest#test_get_current_module().Bar#__call__().
  FlaxUtilTest.test_get_current_module.Foo.__call__: FlaxUtilTest#test_get_current_module().Foo#__call__().
  FlaxUtilTest.test_find_param_nnx.MyModule.__call__: FlaxUtilTest#test_find_param_nnx().MyModule#__call__().
  MyModule.w: MyModule#w.
  FlaxUtilTest.test_get_or_create_variable_linen.Foo.__call__: FlaxUtilTest#test_get_or_create_variable_linen().Foo#__call__().
  FlaxUtilTest.test_get_or_create_variable_nnx: FlaxUtilTest#test_get_or_create_variable_nnx().
  FlaxUtilTest.test_get_or_create_variable_nnx.Foo.__call__: FlaxUtilTest#test_get_or_create_variable_nnx().Foo#__call__().
  FlaxUtilTest.test_create_param_nnx.Foo.__call__: FlaxUtilTest#test_create_param_nnx().Foo#__call__().
  FlaxUtilTest.test_find_param_linen.MyModule.__call__: FlaxUtilTest#test_find_param_linen().MyModule#__call__().
  Foo.bar: Foo#bar.
  Foo.weight: Foo#weight.
  FlaxUtilTest.test_get_current_module: FlaxUtilTest#test_get_current_module().
  FlaxUtilTest.test_get_or_create_variable_linen: FlaxUtilTest#test_get_or_create_variable_linen().
  FlaxUtilTest.test_create_param_nnx: FlaxUtilTest#test_create_param_nnx().
  FlaxUtilTest.test_update_sharding: FlaxUtilTest#test_update_sharding().
  FlaxUtilTest.test_unbox: FlaxUtilTest#test_unbox().
  FlaxUtilTest.test_update_boxed: FlaxUtilTest#test_update_boxed().
  FlaxUtilTest.test_make_rng_linen: FlaxUtilTest#test_make_rng_linen().
  FlaxUtilTest.test_make_rng_linen.MyModule.__call__: FlaxUtilTest#test_make_rng_linen().MyModule#__call__().
  FlaxUtilTest.test_make_rng_nnx: FlaxUtilTest#test_make_rng_nnx().
  FlaxUtilTest.test_make_rng_nnx.MyModule.__call__: FlaxUtilTest#test_make_rng_nnx().MyModule#__call__().
  FlaxUtilTest.test_find_param_linen: FlaxUtilTest#test_find_param_linen().
  FlaxUtilTest.test_find_param_nnx: FlaxUtilTest#test_find_param_nnx().
  FlaxUtilTest.test_get_current_module.Bar: FlaxUtilTest#test_get_current_module().Bar#
  Bar.linear1: Bar#linear1.
  Bar.linear2: Bar#linear2.
  FlaxUtilTest.test_get_current_module.Foo: FlaxUtilTest#test_get_current_module().Foo#
  Foo.linear1: Foo#linear1.
  FlaxUtilTest.test_get_or_create_variable_linen.Foo: FlaxUtilTest#test_get_or_create_variable_linen().Foo#
  FlaxUtilTest.test_get_or_create_variable_nnx.Foo: FlaxUtilTest#test_get_or_create_variable_nnx().Foo#
  FlaxUtilTest.test_create_param_nnx.Foo: FlaxUtilTest#test_create_param_nnx().Foo#
  FlaxUtilTest.test_make_rng_linen.MyModule: FlaxUtilTest#test_make_rng_linen().MyModule#
  FlaxUtilTest.test_make_rng_nnx.MyModule: FlaxUtilTest#test_make_rng_nnx().MyModule#
  FlaxUtilTest.test_find_param_linen.MyModule: FlaxUtilTest#test_find_param_linen().MyModule#
  FlaxUtilTest.test_find_param_nnx.MyModule: FlaxUtilTest#test_find_param_nnx().MyModule#
  FlaxUtilTest: FlaxUtilTest#
  FlaxUtilTest.test_get_current_module.Bar.__init__: FlaxUtilTest#test_get_current_module().Bar#__init__().
  FlaxUtilTest.test_get_current_module.Foo.__init__: FlaxUtilTest#test_get_current_module().Foo#__init__().
  FlaxUtilTest.test_get_or_create_variable_linen.Foo.setup: FlaxUtilTest#test_get_or_create_variable_linen().Foo#setup().
  FlaxUtilTest.test_get_or_create_variable_nnx.Foo.__init__: FlaxUtilTest#test_get_or_create_variable_nnx().Foo#__init__().
  FlaxUtilTest.test_create_param_nnx.Foo.__init__: FlaxUtilTest#test_create_param_nnx().Foo#__init__().
  Foo.qwix_rngs: Foo#qwix_rngs.
  FlaxUtilTest.test_make_rng_nnx.MyModule.__init__: FlaxUtilTest#test_make_rng_nnx().MyModule#__init__().
  MyModule.qwix_rngs: MyModule#qwix_rngs.
  FlaxUtilTest.test_find_param_nnx.MyModule.__init__: FlaxUtilTest#test_find_param_nnx().MyModule#__init__().
  MyModule.w2: MyModule#w2.
---
# Module: [`tests/_src/utils/flax_util_test.py`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py)

## Classes
### `Bar`  ·  implements/extends Module
- def: [`tests/_src/utils/flax_util_test.py:32`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L32)
- signature: `class Bar(nnx.Module):`
- members:
  - `linear1` — [`L35`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L35)
  - `linear2` — [`L36`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L36)
- protocol/private: `__call__`[`L38`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L38), `__init__`[`L34`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L34)
- uses (calls/refs, reference-scoped): [`get_current_module`](../../../qwix/_src/utils/flax_util.md#get_current_module)
- used by: (1 test-only callers)

### `FlaxUtilTest`  ·  implements/extends TestCase
- def: [`tests/_src/utils/flax_util_test.py:27`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L27)
- signature: `class FlaxUtilTest(parameterized.TestCase):`
- members:
  - `test_create_param_nnx(self)` — [`L111`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L111)
  - `test_find_param_linen(self)` — [`L289`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L289)
  - `test_find_param_nnx(self)` — [`L307`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L307)
  - `test_get_current_module(self)` — [`L29`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L29)
  - `test_get_or_create_variable_linen(self)` — [`L59`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L59)
  - `test_get_or_create_variable_nnx(self)` — [`L85`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L85)
  - `test_make_rng_linen(self)` — [`L257`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L257)
  - `test_make_rng_nnx(self)` — [`L275`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L275)
  - `test_unbox(self)` — [`L177`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L177)
  - `test_update_boxed(self)` — [`L195`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L195)
  - `test_update_sharding(self)` — [`L149`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L149)
- uses (calls/refs, reference-scoped): [`unbox`](../../../qwix/_src/utils/flax_util.md#unbox), [`QuantStat`](../../../qwix/_src/utils/flax_util.md#QuantStat), [`update_boxed`](../../../qwix/_src/utils/flax_util.md#update_boxed), [`update_sharding`](../../../qwix/_src/utils/flax_util.md#update_sharding)  (8 test-only)

### `Foo`  ·  implements/extends Module
- def: [`tests/_src/utils/flax_util_test.py:112`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L112)
- signature: `class Foo(nnx.Module):`
- members:
  - `setup(self)` — [`L62`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L62)
  - `bar` — [`L47`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L47)
  - `linear1` — [`L48`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L48)
  - `qwix_rngs` — [`L116`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L116)
  - `weight` — [`L63`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L63)
- protocol/private: `__call__`[`L50`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L50), `__call__`[`L68`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L68), `__call__`[`L91`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L91), `__call__`[`L119`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L119), `__init__`[`L46`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L46), `__init__`[`L88`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L88), `__init__`[`L114`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L114)
- uses (calls/refs, reference-scoped): [`get_or_create_param`](../../../qwix/_src/utils/flax_util.md#get_or_create_param), [`get_current_module`](../../../qwix/_src/utils/flax_util.md#get_current_module), [`get_or_create_variable`](../../../qwix/_src/utils/flax_util.md#get_or_create_variable)  (1 test-only)
- used by: (1 test-only callers)

### `MyModule`  ·  implements/extends Module
- def: [`tests/_src/utils/flax_util_test.py:310`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L310)
- signature: `class MyModule(nnx.Module):`
- members:
  - `qwix_rngs` — [`L280`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L280)
  - `w` — [`L313`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L313)
  - `w2` — [`L314`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L314)
- protocol/private: `__call__`[`L261`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L261), `__call__`[`L282`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L282), `__call__`[`L295`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L295), `__call__`[`L316`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L316), `__init__`[`L278`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L278), `__init__`[`L312`](../../../../../../../raw/code/qwix/tests/_src/utils/flax_util_test.py#L312)
- uses (calls/refs, reference-scoped): [`WithAux`](../../../qwix/_src/providers/ptq.md#WithAux), [`find_param`](../../../qwix/_src/utils/flax_util.md#find_param), [`make_rng`](../../../qwix/_src/utils/flax_util.md#make_rng)
- used by: (1 test-only callers)

