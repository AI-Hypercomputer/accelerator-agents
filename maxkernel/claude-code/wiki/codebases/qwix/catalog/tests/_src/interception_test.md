---
title: 'Module: tests/_src/interception_test.py'
type: catalog
provenance: extracted
module: tests/_src/interception_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.interception_test`/InterceptionTest#
symbols:
  InterceptionTest.func: func().
  InterceptionTest.test_scan_custom_vjp: test_scan_custom_vjp().
  InterceptionTest.test_interception_recursion: test_interception_recursion().
  InterceptionTest.test_interception_recursion_disable_jit_false: test_interception_recursion_disable_jit_false().
  InterceptionTest.test_multiple_interceptions: test_multiple_interceptions().
  InterceptionTest.func2: func2().
  InterceptionTest.test_interception_of_code_object: test_interception_of_code_object().
  InterceptionTest.test_interception_of_pjit_function_disable_jit_false: test_interception_of_pjit_function_disable_jit_false().
  InterceptionTest.test_intercept_class_method: test_intercept_class_method().
  InterceptionTest.test_interception_manager_multiple_interceptions: test_interception_manager_multiple_interceptions().
  InterceptionTest.test_interception_manager_code_object: test_interception_manager_code_object().
  InterceptionTest.replaced_sin: replaced_sin().
  InterceptionTest.test_interception_thread_local: test_interception_thread_local().
  InterceptionTest.test_interception_of_pjit_function: test_interception_of_pjit_function().
  InterceptionTest.test_double_interception: test_double_interception().
  InterceptionTest.func1: func1().
  InterceptionTest.replaced_sin_fwd: replaced_sin_fwd().
  InterceptionTest.func_standard: func_standard().
  InterceptionTest.test_has_attribute: test_has_attribute().
  InterceptionTest.replaced_cos: replaced_cos().
  InterceptionTest.custom_jvp_call: custom_jvp_call().
  InterceptionTest.replaced_sin_bwd: replaced_sin_bwd().
  InterceptionTest: ''
---
# Module: [`tests/_src/interception_test.py`](../../../../../../raw/code/qwix/tests/_src/interception_test.py)

## Classes
### `InterceptionTest`  ·  implements/extends TestCase
- def: [`tests/_src/interception_test.py:22`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L22)
- signature: `class InterceptionTest(absltest.TestCase):`
- members:
  - `custom_jvp_call(self, *args, **kwargs)` — [`L263`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L263)
  - `func(x, in_thread2=False)` — [`L134`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L134)
  - `func1(x)` — [`L25`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L25)
  - `func2(x)` — [`L28`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L28)
  - `func_standard(x, y)` — [`L236`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L236)
  - `replaced_cos(x)` — [`L330`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L330)
  - `replaced_sin(x)` — [`L33`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L33)
  - `replaced_sin_bwd(res, g)` — [`L297`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L297)
  - `replaced_sin_fwd(x)` — [`L294`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L294)
  - `test_double_interception(self)` — [`L248`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L248)
  - `test_has_attribute(self)` — [`L321`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L321)
  - `test_intercept_class_method(self)` — [`L259`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L259)
  - `test_interception_manager_code_object(self)` — [`L394`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L394)
  - `test_interception_manager_multiple_interceptions(self)` — [`L361`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L361)
  - `test_interception_of_code_object(self)` — [`L172`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L172)
  - `test_interception_of_pjit_function(self)` — [`L200`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L200)
  - `test_interception_of_pjit_function_disable_jit_false(self)` — [`L215`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L215)
  - `test_interception_recursion(self)` — [`L24`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L24)
  - `test_interception_recursion_disable_jit_false(self)` — [`L86`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L86)
  - `test_interception_thread_local(self)` — [`L127`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L127)
  - `test_multiple_interceptions(self)` — [`L326`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L326)
  - `test_scan_custom_vjp(self)` — [`L275`](../../../../../../raw/code/qwix/tests/_src/interception_test.py#L275)
- uses (calls/refs, reference-scoped): [`Interceptor`](../../qwix/_src/interception.md#Interceptor), [`mapping`](../../qwix/_src/interception.md#Interceptor.mapping), [`id`](../../qwix/_src/interception.md#Interceptor.id), [`activate_interceptor`](../../qwix/_src/interception.md#_InterceptionManager.activate_interceptor), [`wrap_func_intercepted`](../../qwix/_src/interception.md#wrap_func_intercepted), [`deactivate_interceptor`](../../qwix/_src/interception.md#_InterceptionManager.deactivate_interceptor), [`interception_manager`](../../qwix/_src/interception.md#interception_manager), [`disable_interceptions`](../../qwix/_src/interception.md#disable_interceptions), [`has_attribute`](../../qwix/_src/interception.md#has_attribute)

