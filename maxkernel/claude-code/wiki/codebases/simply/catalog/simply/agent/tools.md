---
title: 'Module: simply/agent/tools.py'
type: catalog
provenance: extracted
module: simply/agent/tools.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.agent.tools`/
symbols:
  Action: Action#
  Tool.execute: Tool#execute().
  Tool: Tool#
  BashTool: BashTool#
  Tool.name: Tool#name.
  Tool.action_type: Tool#action_type.
  Tool.executor: Tool#executor.
  BashAction: BashAction#
  Tool.schema: Tool#schema().
  BashTool.executor: BashTool#executor.
  Tool.description: Tool#description.
  BashAction.timeout: BashAction#timeout.
  BashAction.to_llm: BashAction#to_llm().
  BashAction.max_output_length: BashAction#max_output_length.
  BashAction.format_output: BashAction#format_output().
  Action.to_llm: Action#to_llm().
  DEFAULT_TIMEOUT: DEFAULT_TIMEOUT.
  DEFAULT_MAX_OUTPUT_LENGTH: DEFAULT_MAX_OUTPUT_LENGTH.
  truncate_text: truncate_text().
  BashAction.command: BashAction#command.
  BashTool.description: BashTool#description.
  ToolCallError: ToolCallError#
  BashTool.name: BashTool#name.
  BashTool.action_type: BashTool#action_type.
---

# Module: [`simply/agent/tools.py`](../../../../../../raw/code/simply/simply/agent/tools.py)

## Classes

### `Action` · implements/extends BaseModel

-   def:
    [`simply/agent/tools.py:23`](../../../../../../raw/code/simply/simply/agent/tools.py#L23)
    — documented in
    [simply-agent-tools](../../../concepts/simply-agent-tools.md)
-   doc: An action describe a tool use call.
-   signature: `class Action(pydantic.BaseModel):`
-   members:
    -   `to_llm(self)` —
        [`L30`](../../../../../../raw/code/simply/simply/agent/tools.py#L30) —
        Returns the string representation of the action for LLM consumption.
-   uses (calls/refs, reference-scoped): [`BashAction`](tools.md#BashAction),
    [`MemoryDisplayAction`](memory.md#MemoryDisplayAction),
    [`to_llm`](tools.md#BashAction.to_llm),
    [`MemoryCompressHistoryAction`](memory.md#MemoryCompressHistoryAction),
    [`MemoryDeleteAction`](memory.md#MemoryDeleteAction),
    [`MemoryWriteAction`](memory.md#MemoryWriteAction),
    [`RecordProgressAction`](memory.md#RecordProgressAction)
-   used by: [`execute`](tools.md#Tool.execute), [`Tool`](tools.md#Tool),
    [`make_tool_call`](agent.md#Agent.make_tool_call),
    [`BashAction`](tools.md#BashAction),
    [`MemoryDisplayAction`](memory.md#MemoryDisplayAction),
    [`MemoryCompressHistoryAction`](memory.md#MemoryCompressHistoryAction),
    [`MemoryDeleteAction`](memory.md#MemoryDeleteAction),
    [`MemoryWriteAction`](memory.md#MemoryWriteAction),
    [`RecordProgressAction`](memory.md#RecordProgressAction)

### `BashAction` · implements/extends Action

-   def:
    [`simply/agent/tools.py:113`](../../../../../../raw/code/simply/simply/agent/tools.py#L113)
    — documented in
    [simply-agent-tools](../../../concepts/simply-agent-tools.md)
-   doc: An action to execute a bash command.
-   signature: `class BashAction(Action):`
-   members:
    -   `format_output(cls, stdout: str, stderr: str, exit_code: int, elapsed:
        float, max_output_length: int | None)` —
        [`L139`](../../../../../../raw/code/simply/simply/agent/tools.py#L139) —
        Formats the output of the bash command.
    -   `to_llm(self)` —
        [`L134`](../../../../../../raw/code/simply/simply/agent/tools.py#L134) —
        Returns the string representation of the action for LLM consumption.
    -   `command` —
        [`L115`](../../../../../../raw/code/simply/simply/agent/tools.py#L115)
    -   `max_output_length` —
        [`L124`](../../../../../../raw/code/simply/simply/agent/tools.py#L124)
    -   `timeout` —
        [`L118`](../../../../../../raw/code/simply/simply/agent/tools.py#L118)
-   uses (calls/refs, reference-scoped): [`Action`](tools.md#Action),
    [`DEFAULT_MAX_OUTPUT_LENGTH`](tools.md#DEFAULT_MAX_OUTPUT_LENGTH),
    [`DEFAULT_TIMEOUT`](tools.md#DEFAULT_TIMEOUT),
    [`truncate_text`](tools.md#truncate_text)
-   used by: [`Action`](tools.md#Action),
    [`execute_bash_locally`](env.md#execute_bash_locally),
    [`BashTool`](tools.md#BashTool), [`executor`](tools.md#BashTool.executor),
    [`to_llm`](tools.md#Action.to_llm)

### `BashTool` · implements/extends Tool

-   def:
    [`simply/agent/tools.py:173`](../../../../../../raw/code/simply/simply/agent/tools.py#L173)
    — documented in
    [simply-agent-tools](../../../concepts/simply-agent-tools.md)
-   doc: Bash tool.
-   signature: `class BashTool(Tool[BashAction]):`
-   members:
    -   `action_type` —
        [`L183`](../../../../../../raw/code/simply/simply/agent/tools.py#L183)
        — ---
    -   `description` —
        [`L179`](../../../../../../raw/code/simply/simply/agent/tools.py#L179)
    -   `executor` —
        [`L176`](../../../../../../raw/code/simply/simply/agent/tools.py#L176)
    -   `name` —
        [`L178`](../../../../../../raw/code/simply/simply/agent/tools.py#L178)
-   uses (calls/refs, reference-scoped): [`Tool`](tools.md#Tool),
    [`BashAction`](tools.md#BashAction)
-   used by: [`Tool`](tools.md#Tool), [`get_tools`](env.md#Local.get_tools),
    [`test_execute_success`](tools_test.md#BashToolTest.test_execute_success),
    [`test_execute_timeout`](tools_test.md#BashToolTest.test_execute_timeout),
    [`test_input_validation_error`](tools_test.md#BashToolTest.test_input_validation_error),
    [`test_output_truncation`](tools_test.md#BashToolTest.test_output_truncation),
    [`test_get_tools_returns_bash_tool`](env_test.md#LocalEnvTest.test_get_tools_returns_bash_tool)

### `Tool`

-   def:
    [`simply/agent/tools.py:44`](../../../../../../raw/code/simply/simply/agent/tools.py#L44)
    — documented in
    [simply-agent-memory](../../../concepts/simply-agent-memory.md)
-   doc: A tool for the agent.
-   signature: `class Tool:`
-   members:
    -   `execute(self, args_json: str)` —
        [`L63`](../../../../../../raw/code/simply/simply/agent/tools.py#L63) —
        Executes the action. — documented in
        [simply-agent-tools](../../../concepts/simply-agent-tools.md)
    -   `schema(self)` —
        [`L52`](../../../../../../raw/code/simply/simply/agent/tools.py#L52) —
        Returns the JSON schema of the action type.
    -   `action_type` —
        [`L48`](../../../../../../raw/code/simply/simply/agent/tools.py#L48) —
        documented in
        [simply-agent-tools](../../../concepts/simply-agent-tools.md)
    -   `description` —
        [`L47`](../../../../../../raw/code/simply/simply/agent/tools.py#L47)
    -   `executor` —
        [`L49`](../../../../../../raw/code/simply/simply/agent/tools.py#L49) —
        documented in
        [simply-agent-memory](../../../concepts/simply-agent-memory.md)
    -   `name` —
        [`L46`](../../../../../../raw/code/simply/simply/agent/tools.py#L46) —
        documented in
        [simply-agent-tools](../../../concepts/simply-agent-tools.md)
-   uses (calls/refs, reference-scoped): [`Action`](tools.md#Action),
    [`BashTool`](tools.md#BashTool), [`ToolCallError`](tools.md#ToolCallError)
-   used by: [`get_memory_tools`](memory.md#get_memory_tools),
    [`__post_init__`](agent.md#Agent.__post_init__),
    [`test_tool_call_error`](memory_test.md#MemoryTest.test_tool_call_error),
    [`completion`](llm.md#LiteLLM.completion),
    [`make_tool_call`](agent.md#Agent.make_tool_call),
    [`get_tools`](env.md#Local.get_tools),
    [`restore_memory_system`](agent.md#Agent.restore_memory_system),
    [`BashTool`](tools.md#BashTool), [`completion`](llm.md#LLMBase.completion),
    [`test_execute_success`](tools_test.md#BashToolTest.test_execute_success),
    [`test_execute_timeout`](tools_test.md#BashToolTest.test_execute_timeout),
    [`test_input_validation_error`](tools_test.md#BashToolTest.test_input_validation_error),
    [`test_output_truncation`](tools_test.md#BashToolTest.test_output_truncation),
    [`tools`](agent.md#Agent.tools), [`get_tools`](env.md#Environment.get_tools)

### `ToolCallError` · implements/extends Exception

-   def:
    [`simply/agent/tools.py:35`](../../../../../../raw/code/simply/simply/agent/tools.py#L35)
    — documented in
    [simply-agent-tools](../../../concepts/simply-agent-tools.md)
-   doc: An exception raised by a tool.
-   signature: `class ToolCallError(Exception):`
-   used by: [`execute`](tools.md#Tool.execute)

## Functions

-   `truncate_text(text: str, max_length: int)` —
    [`L100`](../../../../../../raw/code/simply/simply/agent/tools.py#L100) —
    Truncates the text to the maximum number of characters.

## Module values

-   `DEFAULT_MAX_OUTPUT_LENGTH` —
    [`L97`](../../../../../../raw/code/simply/simply/agent/tools.py#L97)
-   `DEFAULT_TIMEOUT` —
    [`L94`](../../../../../../raw/code/simply/simply/agent/tools.py#L94)
