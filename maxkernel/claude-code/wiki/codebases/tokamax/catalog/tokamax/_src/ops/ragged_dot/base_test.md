---
title: 'Module: tokamax/_src/ops/ragged_dot/base_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/base_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.base_test`/
symbols:
  RaggedDotWithExplicitVjpTest.__init__: RaggedDotWithExplicitVjpTest#__init__().
  RaggedDotTest.__init__: RaggedDotTest#__init__().
  RaggedDotTest: RaggedDotTest#
  RaggedDotWithExplicitVjpTest: RaggedDotWithExplicitVjpTest#
  GenerateGroupSizesTest.test_generate_group_sizes: GenerateGroupSizesTest#test_generate_group_sizes().
  RaggedDotTest.setUp: RaggedDotTest#setUp().
  RaggedDotWithExplicitVjpTest.setUp: RaggedDotWithExplicitVjpTest#setUp().
  GenerateGroupSizesTest: GenerateGroupSizesTest#
---
# Module: [`tokamax/_src/ops/ragged_dot/base_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base_test.py)

## Classes
### `GenerateGroupSizesTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/ragged_dot/base_test.py:45`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base_test.py#L45)
- signature: `class GenerateGroupSizesTest(parameterized.TestCase):`
- members:
  - `test_generate_group_sizes(self, m, num_groups, p=None)` — [`L48`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base_test.py#L48)
- uses (calls/refs, reference-scoped): [`generate_group_sizes`](base.md#generate_group_sizes)

### `RaggedDotTest`  ·  implements/extends RaggedDotTestBase
- def: [`tokamax/_src/ops/ragged_dot/base_test.py:23`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base_test.py#L23)
- signature: `class RaggedDotTest(test_base.RaggedDotTestBase):`
- members:
  - `setUp(self)` — [`L28`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base_test.py#L28)
- protocol/private: `__init__`[`L25`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base_test.py#L25)
- uses (calls/refs, reference-scoped): [`RaggedDot`](base.md#RaggedDot)  (2 test-only)
- used by: (1 test-only callers)

### `RaggedDotWithExplicitVjpTest`  ·  implements/extends RaggedDotTestBase
- def: [`tokamax/_src/ops/ragged_dot/base_test.py:34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base_test.py#L34)
- signature: `class RaggedDotWithExplicitVjpTest(test_base.RaggedDotTestBase):`
- members:
  - `setUp(self)` — [`L39`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base_test.py#L39)
- protocol/private: `__init__`[`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/base_test.py#L36)
- uses (calls/refs, reference-scoped): [`RaggedDot`](base.md#RaggedDot), [`vjp`](../op.md#Op.vjp), [`vjp`](base.md#vjp)  (2 test-only)
- used by: (1 test-only callers)

