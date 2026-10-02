---
title: 'Module: simply/agent/tools_test.py'
type: catalog
provenance: extracted
module: simply/agent/tools_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.agent.tools_test`/BashToolTest#
symbols:
  BashToolTest.test_execute_success: test_execute_success().
  BashToolTest.test_execute_timeout: test_execute_timeout().
  BashToolTest.test_input_validation_error: test_input_validation_error().
  BashToolTest.test_output_truncation: test_output_truncation().
  BashToolTest: ''
---

# Module: [`simply/agent/tools_test.py`](../../../../../../raw/code/simply/simply/agent/tools_test.py)

## Classes

### `BashToolTest` · implements/extends TestCase

-   def:
    [`simply/agent/tools_test.py:23`](../../../../../../raw/code/simply/simply/agent/tools_test.py#L23)
-   signature: `class BashToolTest(absltest.TestCase):`
-   members:
    -   `test_execute_success(self)` —
        [`L25`](../../../../../../raw/code/simply/simply/agent/tools_test.py#L25)
    -   `test_execute_timeout(self)` —
        [`L39`](../../../../../../raw/code/simply/simply/agent/tools_test.py#L39)
    -   `test_input_validation_error(self)` —
        [`L53`](../../../../../../raw/code/simply/simply/agent/tools_test.py#L53)
    -   `test_output_truncation(self)` —
        [`L60`](../../../../../../raw/code/simply/simply/agent/tools_test.py#L60)
-   uses (calls/refs, reference-scoped): [`execute`](tools.md#Tool.execute),
    [`execute_bash_locally`](env.md#execute_bash_locally),
    [`BashTool`](tools.md#BashTool), [`executor`](tools.md#BashTool.executor)
