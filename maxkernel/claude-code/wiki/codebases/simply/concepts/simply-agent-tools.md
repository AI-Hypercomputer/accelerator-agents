---
title: simply.agent.tools — pydantic-validated tool-call actions for the agent
type: concept
provenance: mixed
concept: simply-agent-tools
updated: 2026-07-03
status: fresh
---
# simply.agent.tools — pydantic-validated tool-call actions for the agent

## Overview

This module defines the agent's tool-call contract: an [`Action`](../catalog/simply/agent/tools.md#Action)
is a `pydantic.BaseModel` (giving free JSON-schema generation and input validation), and a
[`Tool`](../catalog/simply/agent/tools.md#Tool) pairs an action type with an executor function and
knows how to turn a raw LLM-emitted JSON string into a validated action plus an execution result via
[`Tool.execute`](../catalog/simply/agent/tools.md#Tool.execute). The one concrete tool shipped here,
[`BashTool`](../catalog/simply/agent/tools.md#BashTool), demonstrates the pattern end-to-end:
[`BashAction`](../catalog/simply/agent/tools.md#BashAction) validates `command`/`timeout`/
`max_output_length` via pydantic field annotations, and
[`format_output`](../catalog/simply/agent/tools.md#BashAction)/[`truncate_text`](../catalog/simply/agent/tools.md#Tool.execute)
implement a stderr-prioritized truncation policy for long command output.

## Diagram

```mermaid
flowchart TD
  LLM["LLM emits args_json"] --> EXEC["Tool.execute(args_json)"]
  EXEC --> PARSE["json.loads + action_type(**kwargs)"]
  PARSE -->|JSONDecodeError| ERR1["(None, 'Error: Invalid JSON...')"]
  PARSE -->|pydantic.ValidationError| ERR2["(None, 'Error: Invalid arguments...')"]
  PARSE -->|ok| ACTION["validated Action instance"]
  ACTION --> RUN["self.executor(action)"]
  RUN -->|ToolCallError| ERR3["(action, 'Error: Caught ...')"]
  RUN -->|ok| RESULT["(action, result: str)"]
  ACTION -.->|to_llm()| SCHEMA["Tool.schema\n(JSON schema for LLM function-calling)"]
```

## Design rationale (why it's built this way)

**Pydantic is chosen specifically for its two side effects: JSON schema generation and input
validation, not for general data modeling.** [`Action`](../catalog/simply/agent/tools.md#Action)'s
docstring states this directly ("We use pydantic for automatic JSON schema generation and input
validation") — [`Tool.schema`](../catalog/simply/agent/tools.md#Tool) calls
`self.action_type.model_json_schema()` to build the `{'type': 'function', 'function': {...}}`
payload LLM function-calling APIs expect, and [`Tool.execute`](../catalog/simply/agent/tools.md#Tool.execute)
relies on `pydantic.ValidationError` being raised (and caught) when the LLM emits malformed
arguments — both come for free from inheriting `pydantic.BaseModel`.

**`Tool.execute` distinguishes three failure modes with three different return shapes, all
returning `(Action | None, str)` rather than raising.** Invalid JSON and invalid-per-schema
arguments both return `(None, error_string)` (no valid action was ever constructed);
a [`ToolCallError`](../catalog/simply/agent/tools.md#ToolCallError) raised by the executor returns
`(action, error_string)` (the action *was* valid, only its execution failed) — the doc comment on
`ToolCallError` explains why: it's "automatically caught and reported by the agent loop", whereas "a
general Python exception indicates a system bug" and is deliberately allowed to propagate and crash
the program rather than being silently absorbed into a tool-error string.

**Output truncation prioritizes stderr over stdout by processing parts in reverse order.**
[`BashAction.format_output`](../catalog/simply/agent/tools.md#BashAction) builds `parts = [stdout,
stderr, return_code, elapsed]` in that display order, but when truncating to fit
`max_output_length`, it iterates `reversed(parts)` — truncating stdout first (since it's usually much
larger and less diagnostically important) while preserving stderr and the trailing metadata intact
as long as budget allows.

**`Tool` is generic over its action type (`Tool[T: Action]`), so `executor: Callable[[T], str]` is
statically tied to `action_type: Type[T]`.** [`Tool`](../catalog/simply/agent/tools.md#Tool) is a
frozen, `kw_only` dataclass parameterized by a Python 3.12-style type parameter `T: Action` — this
is why [`BashTool`](../catalog/simply/agent/tools.md#BashTool) can override just `executor`'s runtime
default while `Tool[BashAction]`'s static shape guarantees the executor receives a `BashAction`, not
a bare `Action`.

## Entry points

- [`Tool.execute`](../catalog/simply/agent/tools.md#Tool.execute) — the single call every agent-loop
  iteration makes per tool invocation; owns the parse/validate/run/catch pipeline end to end.
- [`Action.to_llm`](../catalog/simply/agent/tools.md#Action) — called to render an action back into
  the conversation for the LLM to see what it invoked (`model_dump_json(indent=2)`, overridden by
  [`BashAction.to_llm`](../catalog/simply/agent/tools.md#BashAction) for a more compact
  command/timeout summary).
- [`BashAction.format_output`](../catalog/simply/agent/tools.md#BashAction) — the truncation-aware
  formatter a bash executor implementation is expected to call after actually running the command.

## Mechanism (step-by-step)

1. **The LLM emits a JSON string of tool arguments.** [`Tool.execute`](../catalog/simply/agent/tools.md#Tool.execute)
   receives this as `args_json`.
2. **Parse and validate.** `json.loads(args_json)` then `self.`[`action_type`](../catalog/simply/agent/tools.md#Tool.action_type)`(**kwargs)`
   — pydantic validates every field against its `Annotated[..., pydantic.Field(...)]` constraints
   (e.g. [`BashAction.command`](../catalog/simply/agent/tools.md#BashAction)'s required `str`,
   `timeout`'s optional float with a documented default).
3. **On success, the executor runs.** `self.`[`executor`](../catalog/simply/agent/tools.md#Tool.executor)`(action)`
   — for [`BashTool`](../catalog/simply/agent/tools.md#BashTool), this is whatever callable was
   injected at construction (not itself part of this packet's own subgraph — the tool definition and
   its execution backend are decoupled).
4. **Output is capped via a head/tail truncation, not a hard cutoff.**
   [`truncate_text`](../catalog/simply/agent/tools.md#Tool.execute) (used inside
   [`format_output`](../catalog/simply/agent/tools.md#BashAction)) keeps the first and last halves of
   the text and inserts a `[...N chars omitted...]` marker in the middle — preserving both the
   command's initial output and its final state/error, the two most commonly useful parts of a long
   log.
5. **[`Tool.execute`](../catalog/simply/agent/tools.md#Tool.execute)'s `format_output` helper applies
   the truncation budget across parts, stderr-first.** It computes
   `char_budget` (accounting for the joining `\n\n` separators), then greedily truncates
   `reversed(parts)` until the budget is exhausted, re-reversing before joining — so under a tight
   budget, `RETURN CODE`/`TIME ELAPSED` and `STDERR` survive intact longest.

## Key data structures

- **[`Action`](../catalog/simply/agent/tools.md#Action)** (`pydantic.BaseModel`) — the base for every
  tool's argument schema; [`BashAction`](../catalog/simply/agent/tools.md#BashAction) is the one
  concrete subclass in this packet.
- **[`Tool[T: Action]`](../catalog/simply/agent/tools.md#Tool)** (frozen, `kw_only` dataclass) —
  [`name`](../catalog/simply/agent/tools.md#Tool.name),
  [`action_type`](../catalog/simply/agent/tools.md#Tool.action_type),
  [`executor`](../catalog/simply/agent/tools.md#Tool.executor), and a derived
  [`schema`](../catalog/simply/agent/tools.md#Tool) property.
- **[`ToolCallError`](../catalog/simply/agent/tools.md#ToolCallError)** — the one exception type an
  executor is expected to raise for user-facing (LLM-visible) failures; anything else is a bug.

## Dynamics (design intent)

Because [`Tool`](../catalog/simply/agent/tools.md#Tool) and
[`BashAction`](../catalog/simply/agent/tools.md#BashAction) are frozen dataclasses/pydantic models,
a `Tool` instance is immutable after construction — the agent loop can share one `Tool` instance
across many calls to [`execute`](../catalog/simply/agent/tools.md#Tool.execute) without any risk of
cross-call state leakage.

## Edge cases

- [`truncate_text`](../catalog/simply/agent/tools.md#Tool.execute) computes `max_digits =
  str(len(text))` as an *overestimate* of the omitted-character-count string length "to simplify
  implementation" (per its own comment) — the truncation marker's reserved space is conservative,
  not exact.
- [`BashAction.max_output_length`](../catalog/simply/agent/tools.md#BashAction) being `None` disables
  truncation entirely in [`format_output`](../catalog/simply/agent/tools.md#BashAction) (`if
  max_output_length is not None and len(text) > max_output_length`) — a caller can opt out of the
  truncation policy per-call, not just via the class default.

## Open questions

- The actual bash-execution backend behind [`BashTool.executor`](../catalog/simply/agent/tools.md#Tool.executor)
  (subprocess invocation, sandboxing, timeout enforcement) lives outside this packet's subgraph.

## See also
- [simply-agent-memory](simply-agent-memory.md) — the agent's memory-management tools, built on
  the same `Action`/`Tool` pattern.
