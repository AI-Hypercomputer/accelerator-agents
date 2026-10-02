---
title: 'Module: simply/tool_lib_test.py'
type: catalog
provenance: extracted
module: simply/tool_lib_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.tool_lib_test`/ToolLibTest#
symbols:
  ToolLibTest.test_calculator_tool_executor: test_calculator_tool_executor().
  ToolLibTest.test_calculator_tool_executor_error: test_calculator_tool_executor_error().
  ToolLibTest.test_calculator_tool_executor_batch: test_calculator_tool_executor_batch().
  ToolLibTest.test_calculator_tool_executor_safe_eval: test_calculator_tool_executor_safe_eval().
  ToolLibTest: ''
---
# Module: [`simply/tool_lib_test.py`](../../../../../raw/code/simply/simply/tool_lib_test.py)

## Classes
### `ToolLibTest`  ·  implements/extends TestCase
- def: [`simply/tool_lib_test.py:22`](../../../../../raw/code/simply/simply/tool_lib_test.py#L22)
- signature: `class ToolLibTest(absltest.TestCase):`
- members:
  - `test_calculator_tool_executor(self)` — [`L24`](../../../../../raw/code/simply/simply/tool_lib_test.py#L24)
  - `test_calculator_tool_executor_batch(self)` — [`L51`](../../../../../raw/code/simply/simply/tool_lib_test.py#L51)
  - `test_calculator_tool_executor_error(self)` — [`L45`](../../../../../raw/code/simply/simply/tool_lib_test.py#L45)
  - `test_calculator_tool_executor_safe_eval(self)` — [`L64`](../../../../../raw/code/simply/simply/tool_lib_test.py#L64)
- uses (calls/refs, reference-scoped): [`CalculatorToolExecutor`](tool_lib.md#CalculatorToolExecutor)

