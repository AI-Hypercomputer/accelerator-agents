---
title: 'Module: simply/agent/llm.py'
type: catalog
provenance: extracted
module: simply/agent/llm.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.agent.llm`/
symbols:
  LiteLLM.completion: LiteLLM#completion().
  LLMBase.completion: LLMBase#completion().
  LLMBase: LLMBase#
  LLMRegistry: LLMRegistry#
  LiteLLM: LiteLLM#
  LLMRegistry.get_llm: LLMRegistry#get_llm().
  LLMOutput.text: LLMOutput#text.
  LLMOutput.tool_calls: LLMOutput#tool_calls.
  LLMBase.max_tokens: LLMBase#max_tokens().
  LLMBase.count_tokens: LLMBase#count_tokens().
  LLMBase.llm_spec: LLMBase#llm_spec.
  ToolCall.name: ToolCall#name.
  LLMOutput: LLMOutput#
  LiteLLM.max_tokens: LiteLLM#max_tokens().
  LiteLLM.count_tokens: LiteLLM#count_tokens().
  ToolCall: ToolCall#
  ToolCall.arguments: ToolCall#arguments.
  LLMRegistry.namespace: LLMRegistry#namespace.
---

# Module: [`simply/agent/llm.py`](../../../../../../raw/code/simply/simply/agent/llm.py)

## Classes

### `LLMBase`

-   def:
    [`simply/agent/llm.py:74`](../../../../../../raw/code/simply/simply/agent/llm.py#L74)
-   members:
    -   `completion(self, messages: Sequence[Mapping[str, Any]], tools:
        Sequence[tools_lib.Tool], system_prompt: str | None = None, num_retries:
        int | None = None)` —
        [`L79`](../../../../../../raw/code/simply/simply/agent/llm.py#L79) —
        Query LLM to generate a response.
    -   `count_tokens(self, messages: Sequence[Mapping[str, Any]])` —
        [`L94`](../../../../../../raw/code/simply/simply/agent/llm.py#L94) —
        Returns the approxmate number of tokens in the given messages.
    -   `max_tokens(self)` —
        [`L90`](../../../../../../raw/code/simply/simply/agent/llm.py#L90) —
        Returns the maximum number of tokens supported.
    -   `llm_spec` —
        [`L76`](../../../../../../raw/code/simply/simply/agent/llm.py#L76)
-   uses (calls/refs, reference-scoped):
    [`completion`](llm.md#LiteLLM.completion), [`Tool`](tools.md#Tool),
    [`LLMOutput`](llm.md#LLMOutput),
    [`count_tokens`](llm.md#LiteLLM.count_tokens),
    [`max_tokens`](llm.md#LiteLLM.max_tokens)
-   used by: [`load_skills`](skill_loader.md#load_skills),
    [`step`](agent.md#Agent.step), [`run_agent_loop`](main.md#run_agent_loop),
    [`_maybe_summarize`](critique.md#TrajectoryCritique._maybe_summarize),
    [`__post_init__`](agent.md#Agent.__post_init__),
    [`finalize`](critique.md#TrajectoryCritique.finalize),
    [`load_skills_cached`](skill_loader.md#load_skills_cached),
    [`add_step`](critique.md#TrajectoryCritique.add_step),
    [`completion`](llm.md#LiteLLM.completion), [`LiteLLM`](llm.md#LiteLLM),
    [`get_llm`](llm.md#LLMRegistry.get_llm), [`llm`](agent.md#Agent.llm),
    [`_token_counter`](agent.md#Agent._token_counter),
    [`count_tokens`](llm.md#LiteLLM.count_tokens),
    [`max_tokens`](llm.md#LiteLLM.max_tokens)

### `LLMOutput`

-   def:
    [`simply/agent/llm.py:67`](../../../../../../raw/code/simply/simply/agent/llm.py#L67)
-   doc: LLM output.
-   signature: `class LLMOutput:`
-   members:
    -   `text` —
        [`L69`](../../../../../../raw/code/simply/simply/agent/llm.py#L69)
    -   `tool_calls` —
        [`L70`](../../../../../../raw/code/simply/simply/agent/llm.py#L70)
-   uses (calls/refs, reference-scoped): [`ToolCall`](llm.md#ToolCall)
-   used by: [`load_skills`](skill_loader.md#load_skills),
    [`step`](agent.md#Agent.step),
    [`_maybe_summarize`](critique.md#TrajectoryCritique._maybe_summarize),
    [`finalize`](critique.md#TrajectoryCritique.finalize),
    [`completion`](llm.md#LiteLLM.completion),
    [`completion`](llm.md#LLMBase.completion)

### `LLMRegistry` · implements/extends RootRegistry

-   def:
    [`simply/agent/llm.py:28`](../../../../../../raw/code/simply/simply/agent/llm.py#L28)
-   doc: Registry for LLM providers.
-   signature: `class LLMRegistry(registry.RootRegistry):`
-   members:
    -   `get_llm(cls, llm_scheme: str)` —
        [`L33`](../../../../../../raw/code/simply/simply/agent/llm.py#L33) —
        Gets an LLM instance from a scheme string.
    -   `namespace` —
        [`L30`](../../../../../../raw/code/simply/simply/agent/llm.py#L30)
-   uses (calls/refs, reference-scoped):
    [`RootRegistry`](../utils/registry.md#RootRegistry),
    [`get`](../utils/registry.md#RootRegistry.get), [`LLMBase`](llm.md#LLMBase)
-   used by: [`main`](main.md#main),
    [`RootRegistry`](../utils/registry.md#RootRegistry),
    [`__post_init__`](agent.md#Agent.__post_init__),
    [`LiteLLM`](llm.md#LiteLLM),
    [`test_llm_registry_has_litellm`](llm_test.md#LlmTest.test_llm_registry_has_litellm),
    [`test_llm_scheme_parsing`](llm_test.md#LlmTest.test_llm_scheme_parsing)

### `LiteLLM`

-   def:
    [`simply/agent/llm.py:100`](../../../../../../raw/code/simply/simply/agent/llm.py#L100)
-   doc: LiteLLM interface.
-   signature: `class LiteLLM(LLMBase):`
-   members:
    -   `completion(self, messages: Sequence[Mapping[str, Any]], tools:
        Sequence[tools_lib.Tool], system_prompt: str | None = None, num_retries:
        int | None = None)` —
        [`L103`](../../../../../../raw/code/simply/simply/agent/llm.py#L103) —
        Query LLM to generate a response.
    -   `count_tokens(self, messages: Sequence[Mapping[str, Any]])` —
        [`L159`](../../../../../../raw/code/simply/simply/agent/llm.py#L159) —
        Returns the approxmate number of tokens in the given messages.
    -   `max_tokens(self)` —
        [`L154`](../../../../../../raw/code/simply/simply/agent/llm.py#L154) —
        Returns the maximum number of tokens supported.
-   uses (calls/refs, reference-scoped):
    [`register`](../utils/registry.md#RootRegistry.register),
    [`Tool`](tools.md#Tool), [`LLMBase`](llm.md#LLMBase),
    [`schema`](tools.md#Tool.schema), [`LLMRegistry`](llm.md#LLMRegistry),
    [`text`](llm.md#LLMOutput.text),
    [`tool_calls`](llm.md#LLMOutput.tool_calls),
    [`llm_spec`](llm.md#LLMBase.llm_spec), [`LLMOutput`](llm.md#LLMOutput),
    [`name`](llm.md#ToolCall.name), [`ToolCall`](llm.md#ToolCall),
    [`arguments`](llm.md#ToolCall.arguments)
-   used by: [`completion`](llm.md#LLMBase.completion),
    [`count_tokens`](llm.md#LLMBase.count_tokens),
    [`max_tokens`](llm.md#LLMBase.max_tokens)

### `ToolCall`

-   def:
    [`simply/agent/llm.py:60`](../../../../../../raw/code/simply/simply/agent/llm.py#L60)
-   doc: Tool call.
-   signature: `class ToolCall:`
-   members:
    -   `arguments` —
        [`L63`](../../../../../../raw/code/simply/simply/agent/llm.py#L63)
    -   `name` —
        [`L62`](../../../../../../raw/code/simply/simply/agent/llm.py#L62)
-   used by: [`step`](agent.md#Agent.step),
    [`completion`](llm.md#LiteLLM.completion),
    [`tool_calls`](llm.md#LLMOutput.tool_calls)
