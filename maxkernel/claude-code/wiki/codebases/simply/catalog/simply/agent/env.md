---
title: 'Module: simply/agent/env.py'
type: catalog
provenance: extracted
module: simply/agent/env.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.agent.env`/
symbols:
  execute_bash_locally: execute_bash_locally().
  Local.get_tools: Local#get_tools().
  Local: Local#
  EnvRegistry.get_env: EnvRegistry#get_env().
  EnvRegistry: EnvRegistry#
  Environment.get_tools: Environment#get_tools().
  Local.working_dir: Local#working_dir.
  Environment: Environment#
  Local.__str__: Local#__str__().
  EnvRegistry.namespace: EnvRegistry#namespace.
  Local.__init__: Local#__init__().
---

# Module: [`simply/agent/env.py`](../../../../../../raw/code/simply/simply/agent/env.py)

## Classes

### `EnvRegistry` · implements/extends RootRegistry

-   def:
    [`simply/agent/env.py:30`](../../../../../../raw/code/simply/simply/agent/env.py#L30)
-   doc: Registry for execution environments.
-   signature: `class EnvRegistry(registry.RootRegistry):`
-   members:
    -   `get_env(cls, env_spec: str)` —
        [`L36`](../../../../../../raw/code/simply/simply/agent/env.py#L36) —
        Factory method to create an environment from a spec.
    -   `namespace` —
        [`L33`](../../../../../../raw/code/simply/simply/agent/env.py#L33)
-   uses (calls/refs, reference-scoped):
    [`RootRegistry`](../utils/registry.md#RootRegistry),
    [`get`](../utils/registry.md#RootRegistry.get),
    [`Environment`](env.md#Environment)
-   used by: [`main`](main.md#main),
    [`RootRegistry`](../utils/registry.md#RootRegistry),
    [`Local`](env.md#Local),
    [`test_get_local_env`](env_test.md#EnvRegistryTest.test_get_local_env)

### `Environment`

-   def:
    [`simply/agent/env.py:49`](../../../../../../raw/code/simply/simply/agent/env.py#L49)
-   members:
    -   `get_tools(self)` —
        [`L57`](../../../../../../raw/code/simply/simply/agent/env.py#L57) —
        Returns tools for interacting with this environment.
-   uses (calls/refs, reference-scoped): [`Tool`](tools.md#Tool),
    [`get_tools`](env.md#Local.get_tools)
-   used by: [`__post_init__`](agent.md#Agent.__post_init__),
    [`Local`](env.md#Local), [`get_env`](env.md#EnvRegistry.get_env),
    [`env`](agent.md#Agent.env)

### `Local`

-   def:
    [`simply/agent/env.py:114`](../../../../../../raw/code/simply/simply/agent/env.py#L114)
-   doc: An environment that executes commands locally via subprocess.
-   signature: `class Local(Environment):`
-   members:
    -   `get_tools(self)` —
        [`L123`](../../../../../../raw/code/simply/simply/agent/env.py#L123) —
        Returns a BashTool configured for local execution.
    -   `working_dir` —
        [`L118`](../../../../../../raw/code/simply/simply/agent/env.py#L118)
-   protocol/private:
    `__init__`[`L117`](../../../../../../raw/code/simply/simply/agent/env.py#L117),
    `__str__`[`L120`](../../../../../../raw/code/simply/simply/agent/env.py#L120)
-   uses (calls/refs, reference-scoped):
    [`register`](../utils/registry.md#RootRegistry.register),
    [`Tool`](tools.md#Tool),
    [`execute_bash_locally`](env.md#execute_bash_locally),
    [`BashTool`](tools.md#BashTool), [`executor`](tools.md#BashTool.executor),
    [`EnvRegistry`](env.md#EnvRegistry), [`Environment`](env.md#Environment),
    [`description`](tools.md#BashTool.description)
-   used by:
    [`test_get_local_env`](env_test.md#EnvRegistryTest.test_get_local_env),
    [`get_tools`](env.md#Environment.get_tools),
    [`test_get_tools_returns_bash_tool`](env_test.md#LocalEnvTest.test_get_tools_returns_bash_tool),
    [`test_absolute_working_dir_unchanged`](env_test.md#LocalEnvTest.test_absolute_working_dir_unchanged),
    [`test_bash_tool_description_includes_cwd`](env_test.md#LocalEnvTest.test_bash_tool_description_includes_cwd),
    [`test_working_dir_is_absolute`](env_test.md#LocalEnvTest.test_working_dir_is_absolute)

## Functions

-   `execute_bash_locally(action: tools_lib.BashAction, cwd: str | None = None)`
    — [`L64`](../../../../../../raw/code/simply/simply/agent/env.py#L64) —
    Executes a bash command locally via subprocess.
