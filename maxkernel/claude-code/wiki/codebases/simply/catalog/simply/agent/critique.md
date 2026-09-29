---
title: 'Module: simply/agent/critique.py'
type: catalog
provenance: extracted
module: simply/agent/critique.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.agent.critique`/
symbols:
  TrajectoryCritique._maybe_summarize: TrajectoryCritique#_maybe_summarize().
  TrajectoryCritique.finalize: TrajectoryCritique#finalize().
  TrajectoryCritique.add_step: TrajectoryCritique#add_step().
  TrajectoryCritique.critique_text: TrajectoryCritique#critique_text().
  _format_step_events: _format_step_events().
  TrajectoryCritique.segment_critiques: TrajectoryCritique#segment_critiques.
  TrajectoryCritique._pending_start_step: TrajectoryCritique#_pending_start_step.
  TrajectoryCritique._pending_end_step: TrajectoryCritique#_pending_end_step.
  SegmentCritique.start_step: SegmentCritique#start_step.
  SegmentCritique.end_step: SegmentCritique#end_step.
  SegmentCritique.summary: SegmentCritique#summary.
  TrajectoryCritique.task: TrajectoryCritique#task.
  TrajectoryCritique.final_summary: TrajectoryCritique#final_summary.
  TrajectoryCritique._pending_parts: TrajectoryCritique#_pending_parts.
  TrajectoryCritique._pending_token_count: TrajectoryCritique#_pending_token_count.
  TrajectoryCritique: TrajectoryCritique#
  SegmentCritique: SegmentCritique#
  _CRITIQUE_SYSTEM_PROMPT: _CRITIQUE_SYSTEM_PROMPT.
  _CRITIQUE_PROMPT: _CRITIQUE_PROMPT.
  _FINAL_SUMMARY_SYSTEM_PROMPT: _FINAL_SUMMARY_SYSTEM_PROMPT.
  _FINAL_SUMMARY_PROMPT: _FINAL_SUMMARY_PROMPT.
  TrajectoryCritique.token_budget_fraction: TrajectoryCritique#token_budget_fraction.
---

# Module: [`simply/agent/critique.py`](../../../../../../raw/code/simply/simply/agent/critique.py)

## Classes

### `SegmentCritique`

-   def:
    [`simply/agent/critique.py:118`](../../../../../../raw/code/simply/simply/agent/critique.py#L118)
-   doc: A critique summary produced by the system LLM for one segment.
-   signature: `class SegmentCritique:`
-   members:
    -   `end_step` —
        [`L121`](../../../../../../raw/code/simply/simply/agent/critique.py#L121)
    -   `start_step` —
        [`L120`](../../../../../../raw/code/simply/simply/agent/critique.py#L120)
    -   `summary` —
        [`L122`](../../../../../../raw/code/simply/simply/agent/critique.py#L122)
-   used by:
    [`_maybe_summarize`](critique.md#TrajectoryCritique._maybe_summarize),
    [`finalize`](critique.md#TrajectoryCritique.finalize),
    [`critique_text`](critique.md#TrajectoryCritique.critique_text),
    [`segment_critiques`](critique.md#TrajectoryCritique.segment_critiques)

### `TrajectoryCritique`

-   def:
    [`simply/agent/critique.py:126`](../../../../../../raw/code/simply/simply/agent/critique.py#L126)
-   doc: Auto-critiques the agent's trajectory in segments.
-   signature: `class TrajectoryCritique:`
-   members:
    -   `_maybe_summarize(self, llm: llm_lib.LLMBase, force: bool = False)` —
        [`L229`](../../../../../../raw/code/simply/simply/agent/critique.py#L229)
        — Triggers a critique summarization if the pending buffer is large
        enough.
    -   `add_step(self, step: int, events: list[memory_lib.MemoryFile], llm:
        llm_lib.LLMBase)` —
        [`L161`](../../../../../../raw/code/simply/simply/agent/critique.py#L161)
        — Adds events from a step to the pending buffer.
    -   `critique_text(self)` —
        [`L303`](../../../../../../raw/code/simply/simply/agent/critique.py#L303)
        — Critique text for all completed segments.
    -   `finalize(self, llm: llm_lib.LLMBase)` —
        [`L186`](../../../../../../raw/code/simply/simply/agent/critique.py#L186)
        — Finalize the critique: flush pending events and generate a final
        summary.
    -   `final_summary` —
        [`L149`](../../../../../../raw/code/simply/simply/agent/critique.py#L149)
    -   `segment_critiques` —
        [`L144`](../../../../../../raw/code/simply/simply/agent/critique.py#L144)
    -   `task` —
        [`L137`](../../../../../../raw/code/simply/simply/agent/critique.py#L137)
    -   `token_budget_fraction` —
        [`L138`](../../../../../../raw/code/simply/simply/agent/critique.py#L138)
-   protocol/private:
    `_pending_end_step`[`L158`](../../../../../../raw/code/simply/simply/agent/critique.py#L158),
    `_pending_parts`[`L154`](../../../../../../raw/code/simply/simply/agent/critique.py#L154),
    `_pending_start_step`[`L157`](../../../../../../raw/code/simply/simply/agent/critique.py#L157),
    `_pending_token_count`[`L159`](../../../../../../raw/code/simply/simply/agent/critique.py#L159)
-   uses (calls/refs, reference-scoped): [`MemoryFile`](memory.md#MemoryFile),
    [`completion`](llm.md#LLMBase.completion),
    [`_format_step_events`](critique.md#_format_step_events),
    [`LLMBase`](llm.md#LLMBase), [`text`](llm.md#LLMOutput.text),
    [`count_tokens`](llm.md#LLMBase.count_tokens),
    [`max_tokens`](llm.md#LLMBase.max_tokens),
    [`end_step`](critique.md#SegmentCritique.end_step),
    [`start_step`](critique.md#SegmentCritique.start_step),
    [`summary`](critique.md#SegmentCritique.summary),
    [`SegmentCritique`](critique.md#SegmentCritique),
    [`_CRITIQUE_PROMPT`](critique.md#_CRITIQUE_PROMPT),
    [`_CRITIQUE_SYSTEM_PROMPT`](critique.md#_CRITIQUE_SYSTEM_PROMPT),
    [`_FINAL_SUMMARY_PROMPT`](critique.md#_FINAL_SUMMARY_PROMPT),
    [`_FINAL_SUMMARY_SYSTEM_PROMPT`](critique.md#_FINAL_SUMMARY_SYSTEM_PROMPT)
-   used by: [`main`](main.md#main), [`run_agent_loop`](main.md#run_agent_loop)

## Functions

-   `_format_step_events(step: int, events: list[memory_lib.MemoryFile])` —
    [`L107`](../../../../../../raw/code/simply/simply/agent/critique.py#L107) —
    Formats a step's events into a concise text block for the critique LLM.

## Module values

-   `_CRITIQUE_PROMPT` —
    [`L61`](../../../../../../raw/code/simply/simply/agent/critique.py#L61)
-   `_CRITIQUE_SYSTEM_PROMPT` —
    [`L25`](../../../../../../raw/code/simply/simply/agent/critique.py#L25)
-   `_FINAL_SUMMARY_PROMPT` —
    [`L82`](../../../../../../raw/code/simply/simply/agent/critique.py#L82)
-   `_FINAL_SUMMARY_SYSTEM_PROMPT` —
    [`L75`](../../../../../../raw/code/simply/simply/agent/critique.py#L75)
