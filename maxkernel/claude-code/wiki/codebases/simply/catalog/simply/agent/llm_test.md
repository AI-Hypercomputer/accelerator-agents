---
title: 'Module: simply/agent/llm_test.py'
type: catalog
provenance: extracted
module: simply/agent/llm_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.agent.llm_test`/LlmTest#
symbols:
  LlmTest.test_llm_registry_has_litellm: test_llm_registry_has_litellm().
  LlmTest.test_llm_scheme_parsing: test_llm_scheme_parsing().
  LlmTest: ''
---

# Module: [`simply/agent/llm_test.py`](../../../../../../raw/code/simply/simply/agent/llm_test.py)

## Classes

### `LlmTest` · implements/extends TestCase

-   def:
    [`simply/agent/llm_test.py:19`](../../../../../../raw/code/simply/simply/agent/llm_test.py#L19)
-   signature: `class LlmTest(absltest.TestCase):`
-   members:
    -   `test_llm_registry_has_litellm(self)` —
        [`L21`](../../../../../../raw/code/simply/simply/agent/llm_test.py#L21)
    -   `test_llm_scheme_parsing(self)` —
        [`L24`](../../../../../../raw/code/simply/simply/agent/llm_test.py#L24)
        — Tests that LLM scheme parsing works.
-   uses (calls/refs, reference-scoped):
    [`get`](../utils/registry.md#RootRegistry.get),
    [`LLMRegistry`](llm.md#LLMRegistry)
