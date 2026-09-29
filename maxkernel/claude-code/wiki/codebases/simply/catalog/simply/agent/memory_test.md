---
title: 'Module: simply/agent/memory_test.py'
type: catalog
provenance: extracted
module: simply/agent/memory_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.agent.memory_test`/MemoryTest#
symbols:
  MemoryTest.test_write_knowledge: test_write_knowledge().
  MemoryTest.test_tool_call_error: test_tool_call_error().
  MemoryTest.test_memory_file_to_llm: test_memory_file_to_llm().
  MemoryTest.test_update_step: test_update_step().
  MemoryTest.test_compress_history: test_compress_history().
  MemoryTest.test_llm_view: test_llm_view().
  MemoryTest.test_memory_uri_names: test_memory_uri_names().
  MemoryTest: ''
---

# Module: [`simply/agent/memory_test.py`](../../../../../../raw/code/simply/simply/agent/memory_test.py)

## Classes

### `MemoryTest` · implements/extends TestCase

-   def:
    [`simply/agent/memory_test.py:22`](../../../../../../raw/code/simply/simply/agent/memory_test.py#L22)
-   signature: `class MemoryTest(absltest.TestCase):`
-   members:
    -   `test_compress_history(self)` —
        [`L166`](../../../../../../raw/code/simply/simply/agent/memory_test.py#L166)
    -   `test_llm_view(self)` —
        [`L24`](../../../../../../raw/code/simply/simply/agent/memory_test.py#L24)
    -   `test_memory_file_to_llm(self)` —
        [`L58`](../../../../../../raw/code/simply/simply/agent/memory_test.py#L58)
    -   `test_memory_uri_names(self)` —
        [`L41`](../../../../../../raw/code/simply/simply/agent/memory_test.py#L41)
    -   `test_tool_call_error(self)` —
        [`L145`](../../../../../../raw/code/simply/simply/agent/memory_test.py#L145)
    -   `test_update_step(self)` —
        [`L135`](../../../../../../raw/code/simply/simply/agent/memory_test.py#L135)
    -   `test_write_knowledge(self)` —
        [`L88`](../../../../../../raw/code/simply/simply/agent/memory_test.py#L88)
-   uses (calls/refs, reference-scoped):
    [`compress_history`](memory.md#MemorySystem.compress_history),
    [`get_memory_tools`](memory.md#get_memory_tools),
    [`record_llm_output`](memory.md#MemorySystem.record_llm_output),
    [`write`](memory.md#MemorySystem.write),
    [`to_llm`](memory.md#MemoryFile.to_llm),
    [`capture_snapshot`](memory.md#MemorySystem.capture_snapshot),
    [`DisplayMode`](memory.md#DisplayMode),
    [`MemoryFile`](memory.md#MemoryFile), [`files`](memory.md#Memory.files),
    [`llm_view`](memory.md#MemorySystem.llm_view),
    [`execute`](tools.md#Tool.execute),
    [`content`](memory.md#MemoryFile.content),
    [`uri`](memory.md#MemoryFile.uri),
    [`memory_snapshots`](memory.md#MemorySystem.memory_snapshots),
    [`MemorySystem`](memory.md#MemorySystem),
    [`summary`](memory.md#MemoryFile.summary),
    [`display`](memory.md#MemoryFile.display),
    [`memory`](memory.md#MemorySnapshot.memory),
    [`update_step`](memory.md#MemoryFile.update_step),
    [`name`](tools.md#Tool.name), [`delete`](memory.md#MemorySystem.delete),
    [`SUMMARY`](memory.md#DisplayMode.SUMMARY),
    [`FULL`](memory.md#DisplayMode.FULL), [`task`](memory.md#MemorySystem.task),
    [`max_token_budget`](memory.md#MemorySystem.max_token_budget),
    [`is_valid_memory_uri`](memory.md#is_valid_memory_uri),
    [`memory_system_description`](memory.md#MemorySystem.memory_system_description),
    [`default_summary_length`](memory.md#MemorySystem.default_summary_length)
