---
title: 'Module: test/test_libraries.py'
type: catalog
provenance: extracted
module: test/test_libraries.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_libraries`/
symbols:
  LibraryTest.test_basic_sdpa_library: LibraryTest#test_basic_sdpa_library().
  m: m.
  _mylib_scaled_dot_product_attention: _mylib_scaled_dot_product_attention().
  LibraryTest.test_basic_sdpa_library.CustomOpExample: LibraryTest#test_basic_sdpa_library().CustomOpExample#
  _mylib_scaled_dot_product_attention_meta: _mylib_scaled_dot_product_attention_meta().
  LibraryTest: LibraryTest#
  LibraryTest.setUp: LibraryTest#setUp().
  LibraryTest.test_basic_sdpa_library.CustomOpExample.forward: LibraryTest#test_basic_sdpa_library().CustomOpExample#forward().
---
# Module: [`test/test_libraries.py`](../../../../../raw/code/torchax/test/test_libraries.py)

## Classes
### `CustomOpExample`  ·  implements/extends Module
- def: [`test/test_libraries.py:76`](../../../../../raw/code/torchax/test/test_libraries.py#L76)
- signature: `class CustomOpExample(torch.nn.Module):`
- members:
  - `forward(self, q, k, v)` — [`L77`](../../../../../raw/code/torchax/test/test_libraries.py#L77)
- used by: (1 test-only callers)

### `LibraryTest`  ·  implements/extends TestCase
- def: [`test/test_libraries.py:71`](../../../../../raw/code/torchax/test/test_libraries.py#L71)
- signature: `class LibraryTest(unittest.TestCase):`
- members:
  - `setUp(self)` — [`L72`](../../../../../raw/code/torchax/test/test_libraries.py#L72)
  - `test_basic_sdpa_library(self)` — [`L75`](../../../../../raw/code/torchax/test/test_libraries.py#L75)
- uses (calls/refs, reference-scoped): [`exported_program_to_stablehlo`](../torchax/export.md#exported_program_to_stablehlo)  (1 test-only)

## Functions
- `_mylib_scaled_dot_product_attention(q, k, v)` — [`L31`](../../../../../raw/code/torchax/test/test_libraries.py#L31) — Basic scaled dot product attention without all the flags/features.
- `_mylib_scaled_dot_product_attention_meta(q, k, v)` — [`L48`](../../../../../raw/code/torchax/test/test_libraries.py#L48)

## Module values
- `m` — [`L26`](../../../../../raw/code/torchax/test/test_libraries.py#L26)

