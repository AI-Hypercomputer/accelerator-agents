---
title: 'Module: test/test_amp.py'
type: catalog
provenance: extracted
module: test/test_amp.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_amp`/AutocastTest#
symbols:
  AutocastTest.test_auto_cast_ir: test_auto_cast_ir().
  AutocastTest.env: env.
  AutocastTest.setUp: setUp().
  AutocastTest.test_auto_cast_matmul: test_auto_cast_matmul().
  AutocastTest: ''
---
# Module: [`test/test_amp.py`](../../../../../raw/code/torchax/test/test_amp.py)

## Classes
### `AutocastTest`  ·  implements/extends TestCase
- def: [`test/test_amp.py:25`](../../../../../raw/code/torchax/test/test_amp.py#L25)
- signature: `class AutocastTest(unittest.TestCase):`
- members:
  - `setUp(self)` — [`L26`](../../../../../raw/code/torchax/test/test_amp.py#L26)
  - `test_auto_cast_ir(self)` — [`L29`](../../../../../raw/code/torchax/test/test_amp.py#L29)
  - `test_auto_cast_matmul(self)` — [`L37`](../../../../../raw/code/torchax/test/test_amp.py#L37)
  - `env` — [`L27`](../../../../../raw/code/torchax/test/test_amp.py#L27)
- uses (calls/refs, reference-scoped): [`default_env`](../torchax/__init__.md#default_env), [`jax_view`](../torchax/interop.md#jax_view)

