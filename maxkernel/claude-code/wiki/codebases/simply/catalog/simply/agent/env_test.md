---
title: 'Module: simply/agent/env_test.py'
type: catalog
provenance: extracted
module: simply/agent/env_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.agent.env_test`/
symbols:
  EnvRegistryTest.test_get_local_env: EnvRegistryTest#test_get_local_env().
  LocalEnvTest.test_get_tools_returns_bash_tool: LocalEnvTest#test_get_tools_returns_bash_tool().
  LocalEnvTest.test_working_dir_is_absolute: LocalEnvTest#test_working_dir_is_absolute().
  LocalEnvTest.test_absolute_working_dir_unchanged: LocalEnvTest#test_absolute_working_dir_unchanged().
  LocalEnvTest.test_bash_tool_description_includes_cwd: LocalEnvTest#test_bash_tool_description_includes_cwd().
  LocalEnvTest: LocalEnvTest#
  EnvRegistryTest: EnvRegistryTest#
---

# Module: [`simply/agent/env_test.py`](../../../../../../raw/code/simply/simply/agent/env_test.py)

## Classes

### `EnvRegistryTest` · implements/extends TestCase

-   def:
    [`simply/agent/env_test.py:55`](../../../../../../raw/code/simply/simply/agent/env_test.py#L55)
-   signature: `class EnvRegistryTest(absltest.TestCase):`
-   members:
    -   `test_get_local_env(self)` —
        [`L57`](../../../../../../raw/code/simply/simply/agent/env_test.py#L57)
        — EnvRegistry can create a Local env from a spec.
-   uses (calls/refs, reference-scoped): [`Local`](env.md#Local),
    [`get_env`](env.md#EnvRegistry.get_env), [`EnvRegistry`](env.md#EnvRegistry)

### `LocalEnvTest` · implements/extends TestCase

-   def:
    [`simply/agent/env_test.py:23`](../../../../../../raw/code/simply/simply/agent/env_test.py#L23)
-   signature: `class LocalEnvTest(absltest.TestCase):`
-   members:
    -   `test_absolute_working_dir_unchanged(self)` —
        [`L34`](../../../../../../raw/code/simply/simply/agent/env_test.py#L34)
        — Absolute paths are preserved as-is.
    -   `test_bash_tool_description_includes_cwd(self)` —
        [`L47`](../../../../../../raw/code/simply/simply/agent/env_test.py#L47)
        — BashTool description includes the working directory.
    -   `test_get_tools_returns_bash_tool(self)` —
        [`L39`](../../../../../../raw/code/simply/simply/agent/env_test.py#L39)
        — get_tools returns a list containing a BashTool.
    -   `test_working_dir_is_absolute(self)` —
        [`L25`](../../../../../../raw/code/simply/simply/agent/env_test.py#L25)
        — Relative paths are converted to absolute paths.
-   uses (calls/refs, reference-scoped): [`Local`](env.md#Local),
    [`BashTool`](tools.md#BashTool)
