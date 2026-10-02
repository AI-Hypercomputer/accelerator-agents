---
title: 'Module: jax/_src/pallas/pallas_test_util.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/pallas_test_util.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.pallas_test_util`/PallasT
symbols:
  PallasTest.setUp: est#setUp().
  PallasTest.pallas_call: est#pallas_call().
  PallasTest.INTERPRET: est#INTERPRET.
  PallasTPUTest: PUTest#
  PallasTest: est#
  PallasTPUTest.setUp: PUTest#setUp().
---
# Module: [`jax/_src/pallas/pallas_test_util.py`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_test_util.py)

## Classes
### `PallasTPUTest`  ·  implements/extends JaxTestCase
- def: [`jax/_src/pallas/pallas_test_util.py:48`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_test_util.py#L48)
- doc: A test case that only runs on TPUs or in interpret mode on CPU.
- signature: `class PallasTPUTest(PallasTest):`
- members:
  - `setUp(self)` — [`L51`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_test_util.py#L51)
- uses (calls/refs, reference-scoped): [`PallasTest`](pallas_test_util.md#PallasTest)

### `PallasTest`  ·  implements/extends TestCase
- def: [`jax/_src/pallas/pallas_test_util.py:23`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_test_util.py#L23)
- signature: `class PallasTest(jtu.JaxTestCase):`
- members:
  - `pallas_call(self, *args, **kwargs)` — [`L44`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_test_util.py#L44)
  - `setUp(self)` — [`L26`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_test_util.py#L26)
  - `INTERPRET` — [`L24`](../../../../../../../raw/code/jax/jax/_src/pallas/pallas_test_util.py#L24)
- uses (calls/refs, reference-scoped): [`pallas_call`](pallas_call.md#pallas_call), [`jax_pallas_use_mosaic_gpu`](../config.md#jax_pallas_use_mosaic_gpu)
- used by: [`PallasTPUTest`](pallas_test_util.md#PallasTPUTest)

