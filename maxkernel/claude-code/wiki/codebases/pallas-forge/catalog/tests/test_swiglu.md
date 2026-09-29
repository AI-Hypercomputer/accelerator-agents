---
title: 'Module: tests/test_swiglu.py'
type: catalog
provenance: extracted
module: tests/test_swiglu.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `tests.test_swiglu`/
symbols:
  TestFusedSwiGLU.test_basic_correctness: TestFusedSwiGLU#test_basic_correctness().
  TestFusedSwiGLU.test_batched_input: TestFusedSwiGLU#test_batched_input().
  TestFusedSwiGLU.test_unaligned_dimensions: TestFusedSwiGLU#test_unaligned_dimensions().
  TestFusedGeGLU.test_basic_correctness: TestFusedGeGLU#test_basic_correctness().
  TestFusedGeGLU.test_geglu_vs_swiglu_different: TestFusedGeGLU#test_geglu_vs_swiglu_different().
  swiglu_reference: swiglu_reference().
  TestFusedSwiGLU.test_bfloat16: TestFusedSwiGLU#test_bfloat16().
  geglu_reference: geglu_reference().
  TestFusedSwiGLU: TestFusedSwiGLU#
  TestFusedGeGLU: TestFusedGeGLU#
---
# Module: [`tests/test_swiglu.py`](../../../../../raw/code/pallas-forge/tests/test_swiglu.py)

## Classes
### `TestFusedGeGLU`
- def: [`tests/test_swiglu.py:86`](../../../../../raw/code/pallas-forge/tests/test_swiglu.py#L86)
- doc: Correctness tests for fused_geglu.
- signature: `class TestFusedGeGLU:`
- members:
  - `test_basic_correctness(self, rng_key)` — [`L89`](../../../../../raw/code/pallas-forge/tests/test_swiglu.py#L89) — GeGLU matches reference implementation.
  - `test_geglu_vs_swiglu_different(self, rng_key)` — [`L103`](../../../../../raw/code/pallas-forge/tests/test_swiglu.py#L103) — GeGLU and SwiGLU should produce different results.
- uses (calls/refs, reference-scoped): [`fused_swiglu`](../pallas_forge/kernels/swiglu.md#fused_swiglu), [`fused_geglu`](../pallas_forge/kernels/swiglu.md#fused_geglu)  (1 test-only)

### `TestFusedSwiGLU`
- def: [`tests/test_swiglu.py:25`](../../../../../raw/code/pallas-forge/tests/test_swiglu.py#L25)
- doc: Correctness tests for fused_swiglu.
- signature: `class TestFusedSwiGLU:`
- members:
  - `test_basic_correctness(self, rng_key)` — [`L28`](../../../../../raw/code/pallas-forge/tests/test_swiglu.py#L28) — SwiGLU matches reference implementation.
  - `test_batched_input(self, rng_key)` — [`L42`](../../../../../raw/code/pallas-forge/tests/test_swiglu.py#L42) — Should handle 3D (batch, seq, dim) inputs.
  - `test_bfloat16(self, rng_key)` — [`L59`](../../../../../raw/code/pallas-forge/tests/test_swiglu.py#L59) — Should work with bfloat16 and maintain output dtype.
  - `test_unaligned_dimensions(self, rng_key)` — [`L71`](../../../../../raw/code/pallas-forge/tests/test_swiglu.py#L71) — Non-aligned dimensions should still work via padding.
- uses (calls/refs, reference-scoped): [`fused_swiglu`](../pallas_forge/kernels/swiglu.md#fused_swiglu)  (1 test-only)

## Functions
- `geglu_reference(x, w_gate, w_up)` — [`L18`](../../../../../raw/code/pallas-forge/tests/test_swiglu.py#L18) — Reference JAX implementation of GeGLU.
- `swiglu_reference(x, w_gate, w_up)` — [`L11`](../../../../../raw/code/pallas-forge/tests/test_swiglu.py#L11) — Reference JAX implementation of SwiGLU.

