---
title: 'Module: simply/agent/agent.py'
type: catalog
provenance: extracted
module: simply/agent/agent.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.agent.agent`/
symbols:
  Agent.step: Agent#step().
  Agent.__post_init__: Agent#__post_init__().
  Agent.memory_system: Agent#memory_system.
  Agent.save_memory_snapshot: Agent#save_memory_snapshot().
  Agent.make_tool_call: Agent#make_tool_call().
  Agent.restore_memory_system: Agent#restore_memory_system().
  Agent.tui: Agent#tui.
  Agent.tools: Agent#tools.
  Agent.llm: Agent#llm.
  Agent.session_dir: Agent#session_dir.
  Agent._token_counter: Agent#_token_counter().
  Agent.env: Agent#env.
  Agent.predefined_knowledge: Agent#predefined_knowledge.
  Agent.task: Agent#task.
  Agent: Agent#
  Agent.llm_scheme: Agent#llm_scheme.
  save_pickle_atomic: save_pickle_atomic().
---

# Module: [`simply/agent/agent.py`](../../../../../../raw/code/simply/simply/agent/agent.py)

## Classes

### `Agent`

-   def:
    [`simply/agent/agent.py:47`](../../../../../../raw/code/simply/simply/agent/agent.py#L47)
-   doc: Agent with a memory system that drives the LLM / tool call loop.
-   signature: `class Agent:`
-   members:
    -   `_token_counter(self, text: str)` —
        [`L88`](../../../../../../raw/code/simply/simply/agent/agent.py#L88) —
        Token counter using the agent's LLM.
    -   `make_tool_call(self, name: str, args: str, tool_call_idx: int)` —
        [`L153`](../../../../../../raw/code/simply/simply/agent/agent.py#L153) —
        Makes a tool call and records the result in the memory.
    -   `restore_memory_system(self, memory_system: mem_lib.MemorySystem)` —
        [`L92`](../../../../../../raw/code/simply/simply/agent/agent.py#L92) —
        Restores a previously saved MemorySystem (e.g. for resuming).
    -   `save_memory_snapshot(self)` —
        [`L167`](../../../../../../raw/code/simply/simply/agent/agent.py#L167) —
        Saves the memory snapshot to the session directory.
    -   `step(self)` —
        [`L105`](../../../../../../raw/code/simply/simply/agent/agent.py#L105) —
        Run a single step of the agent.
    -   `env` —
        [`L55`](../../../../../../raw/code/simply/simply/agent/agent.py#L55)
    -   `llm` —
        [`L66`](../../../../../../raw/code/simply/simply/agent/agent.py#L66)
    -   `llm_scheme` —
        [`L60`](../../../../../../raw/code/simply/simply/agent/agent.py#L60)
    -   `memory_system` —
        [`L69`](../../../../../../raw/code/simply/simply/agent/agent.py#L69)
    -   `predefined_knowledge` —
        [`L56`](../../../../../../raw/code/simply/simply/agent/agent.py#L56)
    -   `session_dir` —
        [`L61`](../../../../../../raw/code/simply/simply/agent/agent.py#L61)
    -   `task` —
        [`L54`](../../../../../../raw/code/simply/simply/agent/agent.py#L54)
    -   `tools` —
        [`L72`](../../../../../../raw/code/simply/simply/agent/agent.py#L72)
    -   `tui` —
        [`L62`](../../../../../../raw/code/simply/simply/agent/agent.py#L62)
-   protocol/private:
    `__post_init__`[`L76`](../../../../../../raw/code/simply/simply/agent/agent.py#L76)
-   uses (calls/refs, reference-scoped):
    [`get_memory_tools`](memory.md#get_memory_tools),
    [`record_llm_output`](memory.md#MemorySystem.record_llm_output),
    [`record_tool_call`](memory.md#MemorySystem.record_tool_call),
    [`capture_snapshot`](memory.md#MemorySystem.capture_snapshot),
    [`MemoryFile`](memory.md#MemoryFile),
    [`llm_view`](memory.md#MemorySystem.llm_view),
    [`execute`](tools.md#Tool.execute), [`Tool`](tools.md#Tool),
    [`system_status`](memory.md#MemorySnapshot.system_status),
    [`last_snapshot`](memory.md#MemorySystem.last_snapshot),
    [`MemorySystem`](memory.md#MemorySystem),
    [`status_step`](memory.md#SystemStatus.status_step),
    [`DisplayBase`](tui.md#DisplayBase),
    [`completion`](llm.md#LLMBase.completion), [`name`](tools.md#Tool.name),
    [`update_system_status`](tui.md#DisplayBase.update_system_status),
    [`update_status`](tui.md#DisplayBase.update_status),
    [`LLMBase`](llm.md#LLMBase),
    [`display_llm_output`](tui.md#DisplayBase.display_llm_output),
    [`display_tool_call`](tui.md#DisplayBase.display_tool_call),
    [`task`](memory.md#MemorySystem.task), [`LLMRegistry`](llm.md#LLMRegistry),
    [`get_llm`](llm.md#LLMRegistry.get_llm),
    [`max_token_budget`](memory.md#MemorySystem.max_token_budget),
    [`text`](llm.md#LLMOutput.text),
    [`tool_calls`](llm.md#LLMOutput.tool_calls),
    [`get_tools`](env.md#Environment.get_tools),
    [`count_tokens`](llm.md#LLMBase.count_tokens),
    [`max_tokens`](llm.md#LLMBase.max_tokens),
    [`predefined_knowledge`](memory.md#MemorySystem.predefined_knowledge),
    [`PrintDisplay`](tui.md#PrintDisplay), [`to_llm`](tools.md#Action.to_llm),
    [`Environment`](env.md#Environment), [`name`](llm.md#ToolCall.name),
    [`arguments`](llm.md#ToolCall.arguments),
    [`save_pickle_atomic`](agent.md#save_pickle_atomic)
-   used by: [`main`](main.md#main), [`run_agent_loop`](main.md#run_agent_loop)

## Functions

-   `save_pickle_atomic(obj: Any, path: epath.Path)` —
    [`L30`](../../../../../../raw/code/simply/simply/agent/agent.py#L30) — Saves
    an object to a pickle file with atomic write and backup.
