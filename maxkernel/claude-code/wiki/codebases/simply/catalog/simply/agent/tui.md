---
title: 'Module: simply/agent/tui.py'
type: catalog
provenance: extracted
module: simply/agent/tui.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.agent.tui`/
symbols:
  StatusDisplay.update_system_status: StatusDisplay#update_system_status().
  FullDisplay.update_system_status: FullDisplay#update_system_status().
  PrintDisplay.update_system_status: PrintDisplay#update_system_status().
  StatusDisplay._render: StatusDisplay#_render().
  DisplayBase: DisplayBase#
  DisplayBase.update_system_status: DisplayBase#update_system_status().
  DisplayBase.update_status: DisplayBase#update_status().
  StatusDisplay._refresh: StatusDisplay#_refresh().
  DisplayBase.display_llm_output: DisplayBase#display_llm_output().
  DisplayBase.display_tool_call: DisplayBase#display_tool_call().
  DisplayBase.start: DisplayBase#start().
  DisplayBase.stop: DisplayBase#stop().
  DisplayBase.set_header: DisplayBase#set_header().
  DisplayBase.set_task: DisplayBase#set_task().
  FullDisplay.console: FullDisplay#console.
  StatusDisplay._live: StatusDisplay#_live.
  FullDisplay.set_header: FullDisplay#set_header().
  StatusDisplay._truncated_task: StatusDisplay#_truncated_task().
  _format_tokens: _format_tokens().
  DisplayBase.print_header: DisplayBase#print_header().
  StatusDisplay.start: StatusDisplay#start().
  StatusDisplay.set_header: StatusDisplay#set_header().
  StatusDisplay.set_task: StatusDisplay#set_task().
  StatusDisplay.update_status: StatusDisplay#update_status().
  StatusDisplay._header: StatusDisplay#_header.
  StatusDisplay._task: StatusDisplay#_task.
  PrintDisplay: PrintDisplay#
  FullDisplay: FullDisplay#
  FullDisplay.set_task: FullDisplay#set_task().
  StatusDisplay: StatusDisplay#
  StatusDisplay._status_step: StatusDisplay#_status_step.
  StatusDisplay._status_message: StatusDisplay#_status_message.
  _format_elapsed: _format_elapsed().
  PrintDisplay.set_header: PrintDisplay#set_header().
  PrintDisplay.print_header: PrintDisplay#print_header().
  FullDisplay.update_status: FullDisplay#update_status().
  FullDisplay.display_llm_output: FullDisplay#display_llm_output().
  FullDisplay.display_tool_call: FullDisplay#display_tool_call().
  StatusDisplay.stop: StatusDisplay#stop().
  StatusDisplay.print_header: StatusDisplay#print_header().
  PrintDisplay._header: PrintDisplay#_header.
  StatusDisplay._token_usage: StatusDisplay#_token_usage.
  StatusDisplay._elapsed: StatusDisplay#_elapsed.
  _format_tokens._fmt: _format_tokens()._fmt().
  FullDisplay._header: FullDisplay#_header.
  FullDisplay.start: FullDisplay#start().
  FullDisplay.stop: FullDisplay#stop().
  PrintDisplay.__init__: PrintDisplay#__init__().
  PrintDisplay.start: PrintDisplay#start().
  PrintDisplay.stop: PrintDisplay#stop().
  PrintDisplay.set_task: PrintDisplay#set_task().
  PrintDisplay.update_status: PrintDisplay#update_status().
  PrintDisplay.display_llm_output: PrintDisplay#display_llm_output().
  PrintDisplay.display_tool_call: PrintDisplay#display_tool_call().
  FullDisplay.__init__: FullDisplay#__init__().
  StatusDisplay.__init__: StatusDisplay#__init__().
  StatusDisplay.display_llm_output: StatusDisplay#display_llm_output().
  StatusDisplay.display_tool_call: StatusDisplay#display_tool_call().
---

# Module: [`simply/agent/tui.py`](../../../../../../raw/code/simply/simply/agent/tui.py)

## Classes

### `DisplayBase` · implements/extends ABC

-   def:
    [`simply/agent/tui.py:42`](../../../../../../raw/code/simply/simply/agent/tui.py#L42)
-   doc: Abstract base class for agent display.
-   signature: `class DisplayBase(abc.ABC):`
-   members:
    -   `display_llm_output(self, output: str)` —
        [`L70`](../../../../../../raw/code/simply/simply/agent/tui.py#L70) —
        Display the LLM output.
    -   `display_tool_call(self, name: str, tool_inputs: str, result: str)` —
        [`L74`](../../../../../../raw/code/simply/simply/agent/tui.py#L74) —
        Display the tool call inputs and results.
    -   `print_header(self)` —
        [`L77`](../../../../../../raw/code/simply/simply/agent/tui.py#L77) —
        Print the header info to stdout (for use after the display stops).
    -   `set_header(self, info: dict[str, str])` —
        [`L54`](../../../../../../raw/code/simply/simply/agent/tui.py#L54) — Set
        the static header info (LLM, session dir, run ID, workspace).
    -   `set_task(self, task: str)` —
        [`L62`](../../../../../../raw/code/simply/simply/agent/tui.py#L62) — Set
        the task content to display.
    -   `start(self)` —
        [`L46`](../../../../../../raw/code/simply/simply/agent/tui.py#L46) —
        Start the display.
    -   `stop(self)` —
        [`L50`](../../../../../../raw/code/simply/simply/agent/tui.py#L50) —
        Stop the display.
    -   `update_status(self, message: str)` —
        [`L66`](../../../../../../raw/code/simply/simply/agent/tui.py#L66) —
        Update the bottom status bar message.
    -   `update_system_status(self, status: memory_lib.SystemStatus)` —
        [`L58`](../../../../../../raw/code/simply/simply/agent/tui.py#L58) —
        Update the system status row (step, tokens, elapsed). — documented in
        [simply-agent-tui](../../../concepts/simply-agent-tui.md)
-   uses (calls/refs, reference-scoped):
    [`update_system_status`](tui.md#StatusDisplay.update_system_status),
    [`update_system_status`](tui.md#FullDisplay.update_system_status),
    [`update_system_status`](tui.md#PrintDisplay.update_system_status),
    [`SystemStatus`](memory.md#SystemStatus),
    [`set_header`](tui.md#FullDisplay.set_header),
    [`set_header`](tui.md#StatusDisplay.set_header),
    [`set_task`](tui.md#StatusDisplay.set_task),
    [`start`](tui.md#StatusDisplay.start),
    [`update_status`](tui.md#StatusDisplay.update_status),
    [`FullDisplay`](tui.md#FullDisplay), [`PrintDisplay`](tui.md#PrintDisplay),
    [`StatusDisplay`](tui.md#StatusDisplay),
    [`set_task`](tui.md#FullDisplay.set_task),
    [`display_llm_output`](tui.md#FullDisplay.display_llm_output),
    [`display_tool_call`](tui.md#FullDisplay.display_tool_call),
    [`print_header`](tui.md#PrintDisplay.print_header),
    [`print_header`](tui.md#StatusDisplay.print_header),
    [`set_header`](tui.md#PrintDisplay.set_header),
    [`stop`](tui.md#StatusDisplay.stop),
    [`update_status`](tui.md#FullDisplay.update_status),
    [`start`](tui.md#FullDisplay.start), [`stop`](tui.md#FullDisplay.stop),
    [`display_llm_output`](tui.md#PrintDisplay.display_llm_output),
    [`display_llm_output`](tui.md#StatusDisplay.display_llm_output),
    [`display_tool_call`](tui.md#PrintDisplay.display_tool_call),
    [`display_tool_call`](tui.md#StatusDisplay.display_tool_call),
    [`set_task`](tui.md#PrintDisplay.set_task),
    [`start`](tui.md#PrintDisplay.start), [`stop`](tui.md#PrintDisplay.stop),
    [`update_status`](tui.md#PrintDisplay.update_status)
-   used by: [`step`](agent.md#Agent.step),
    [`run_agent_loop`](main.md#run_agent_loop),
    [`make_tool_call`](agent.md#Agent.make_tool_call),
    [`tui`](agent.md#Agent.tui), [`FullDisplay`](tui.md#FullDisplay),
    [`PrintDisplay`](tui.md#PrintDisplay),
    [`StatusDisplay`](tui.md#StatusDisplay)

### `FullDisplay` · implements/extends DisplayBase

-   def:
    [`simply/agent/tui.py:124`](../../../../../../raw/code/simply/simply/agent/tui.py#L124)
-   doc: Display that outputs full agent output including LLM text and tool
    calls.
-   signature: `class FullDisplay(DisplayBase):`
-   members:
    -   `display_llm_output(self, output: str)` —
        [`L161`](../../../../../../raw/code/simply/simply/agent/tui.py#L161)
    -   `display_tool_call(self, name: str, tool_inputs: str, result: str)` —
        [`L168`](../../../../../../raw/code/simply/simply/agent/tui.py#L168)
    -   `set_header(self, info: dict[str, str])` —
        [`L137`](../../../../../../raw/code/simply/simply/agent/tui.py#L137)
    -   `set_task(self, task: str)` —
        [`L153`](../../../../../../raw/code/simply/simply/agent/tui.py#L153)
    -   `start(self)` —
        [`L131`](../../../../../../raw/code/simply/simply/agent/tui.py#L131)
    -   `stop(self)` —
        [`L134`](../../../../../../raw/code/simply/simply/agent/tui.py#L134)
    -   `update_status(self, message: str)` —
        [`L158`](../../../../../../raw/code/simply/simply/agent/tui.py#L158)
    -   `update_system_status(self, status: memory_lib.SystemStatus)` —
        [`L143`](../../../../../../raw/code/simply/simply/agent/tui.py#L143)
    -   `console` —
        [`L129`](../../../../../../raw/code/simply/simply/agent/tui.py#L129)
-   protocol/private:
    `__init__`[`L127`](../../../../../../raw/code/simply/simply/agent/tui.py#L127),
    `_header`[`L128`](../../../../../../raw/code/simply/simply/agent/tui.py#L128)
-   uses (calls/refs, reference-scoped):
    [`status_step`](memory.md#SystemStatus.status_step),
    [`DisplayBase`](tui.md#DisplayBase),
    [`SystemStatus`](memory.md#SystemStatus),
    [`approximate_token_usage`](memory.md#SystemStatus.approximate_token_usage),
    [`max_token_budget`](memory.md#SystemStatus.max_token_budget),
    [`_format_tokens`](tui.md#_format_tokens),
    [`elapsed_seconds`](memory.md#SystemStatus.elapsed_seconds),
    [`_format_elapsed`](tui.md#_format_elapsed)
-   used by: [`main`](main.md#main), [`DisplayBase`](tui.md#DisplayBase),
    [`update_system_status`](tui.md#DisplayBase.update_system_status),
    [`update_status`](tui.md#DisplayBase.update_status),
    [`display_llm_output`](tui.md#DisplayBase.display_llm_output),
    [`display_tool_call`](tui.md#DisplayBase.display_tool_call),
    [`set_header`](tui.md#DisplayBase.set_header),
    [`set_task`](tui.md#DisplayBase.set_task),
    [`start`](tui.md#DisplayBase.start), [`stop`](tui.md#DisplayBase.stop)

### `PrintDisplay` · implements/extends DisplayBase

-   def:
    [`simply/agent/tui.py:81`](../../../../../../raw/code/simply/simply/agent/tui.py#L81)
-   doc: Minimal fallback display that outputs plain one-liners.
-   signature: `class PrintDisplay(DisplayBase):`
-   members:
    -   `display_llm_output(self, output: str)` —
        [`L113`](../../../../../../raw/code/simply/simply/agent/tui.py#L113)
    -   `display_tool_call(self, name: str, tool_inputs: str, result: str)` —
        [`L116`](../../../../../../raw/code/simply/simply/agent/tui.py#L116)
    -   `print_header(self)` —
        [`L119`](../../../../../../raw/code/simply/simply/agent/tui.py#L119)
    -   `set_header(self, info: dict[str, str])` —
        [`L93`](../../../../../../raw/code/simply/simply/agent/tui.py#L93)
    -   `set_task(self, task: str)` —
        [`L107`](../../../../../../raw/code/simply/simply/agent/tui.py#L107)
    -   `start(self)` —
        [`L87`](../../../../../../raw/code/simply/simply/agent/tui.py#L87)
    -   `stop(self)` —
        [`L90`](../../../../../../raw/code/simply/simply/agent/tui.py#L90)
    -   `update_status(self, message: str)` —
        [`L110`](../../../../../../raw/code/simply/simply/agent/tui.py#L110)
    -   `update_system_status(self, status: memory_lib.SystemStatus)` —
        [`L98`](../../../../../../raw/code/simply/simply/agent/tui.py#L98) —
        documented in [simply-agent-tui](../../../concepts/simply-agent-tui.md)
-   protocol/private:
    `__init__`[`L84`](../../../../../../raw/code/simply/simply/agent/tui.py#L84),
    `_header`[`L85`](../../../../../../raw/code/simply/simply/agent/tui.py#L85)
-   uses (calls/refs, reference-scoped):
    [`status_step`](memory.md#SystemStatus.status_step),
    [`DisplayBase`](tui.md#DisplayBase),
    [`SystemStatus`](memory.md#SystemStatus),
    [`approximate_token_usage`](memory.md#SystemStatus.approximate_token_usage),
    [`max_token_budget`](memory.md#SystemStatus.max_token_budget),
    [`_format_tokens`](tui.md#_format_tokens),
    [`elapsed_seconds`](memory.md#SystemStatus.elapsed_seconds),
    [`_format_elapsed`](tui.md#_format_elapsed)
-   used by: [`tui`](agent.md#Agent.tui), [`DisplayBase`](tui.md#DisplayBase),
    [`update_system_status`](tui.md#DisplayBase.update_system_status),
    [`update_status`](tui.md#DisplayBase.update_status),
    [`display_llm_output`](tui.md#DisplayBase.display_llm_output),
    [`display_tool_call`](tui.md#DisplayBase.display_tool_call),
    [`set_header`](tui.md#DisplayBase.set_header),
    [`set_task`](tui.md#DisplayBase.set_task),
    [`start`](tui.md#DisplayBase.start), [`stop`](tui.md#DisplayBase.stop),
    [`print_header`](tui.md#DisplayBase.print_header)

### `StatusDisplay` · implements/extends DisplayBase

-   def:
    [`simply/agent/tui.py:181`](../../../../../../raw/code/simply/simply/agent/tui.py#L181)
-   doc: A compact, live-updating status panel using rich.live.Live.
-   signature: `class StatusDisplay(DisplayBase):`
-   members:
    -   `_render(self)` —
        [`L242`](../../../../../../raw/code/simply/simply/agent/tui.py#L242) —
        Build the full status panel. — documented in
        [simply-agent-tui](../../../concepts/simply-agent-tui.md)
    -   `_truncated_task(self)` —
        [`L287`](../../../../../../raw/code/simply/simply/agent/tui.py#L287) —
        Return the task text, truncated to fit available terminal height. —
        documented in [simply-agent-tui](../../../concepts/simply-agent-tui.md)
    -   `display_llm_output(self, output: str)` —
        [`L226`](../../../../../../raw/code/simply/simply/agent/tui.py#L226)
    -   `display_tool_call(self, name: str, tool_inputs: str, result: str)` —
        [`L229`](../../../../../../raw/code/simply/simply/agent/tui.py#L229)
    -   `print_header(self)` —
        [`L232`](../../../../../../raw/code/simply/simply/agent/tui.py#L232)
    -   `set_header(self, info: dict[str, str])` —
        [`L206`](../../../../../../raw/code/simply/simply/agent/tui.py#L206)
    -   `set_task(self, task: str)` —
        [`L218`](../../../../../../raw/code/simply/simply/agent/tui.py#L218)
    -   `start(self)` —
        [`L193`](../../../../../../raw/code/simply/simply/agent/tui.py#L193) —
        documented in [simply-agent-tui](../../../concepts/simply-agent-tui.md)
    -   `stop(self)` —
        [`L201`](../../../../../../raw/code/simply/simply/agent/tui.py#L201)
    -   `update_status(self, message: str)` —
        [`L222`](../../../../../../raw/code/simply/simply/agent/tui.py#L222)
    -   `update_system_status(self, status: memory_lib.SystemStatus)` —
        [`L210`](../../../../../../raw/code/simply/simply/agent/tui.py#L210) —
        documented in [simply-agent-tui](../../../concepts/simply-agent-tui.md)
-   protocol/private:
    `__init__`[`L184`](../../../../../../raw/code/simply/simply/agent/tui.py#L184),
    `_elapsed`[`L188`](../../../../../../raw/code/simply/simply/agent/tui.py#L188),
    `_header`[`L185`](../../../../../../raw/code/simply/simply/agent/tui.py#L185),
    `_live`[`L191`](../../../../../../raw/code/simply/simply/agent/tui.py#L191),
    `_refresh`[`L238`](../../../../../../raw/code/simply/simply/agent/tui.py#L238),
    `_status_message`[`L190`](../../../../../../raw/code/simply/simply/agent/tui.py#L190),
    `_status_step`[`L186`](../../../../../../raw/code/simply/simply/agent/tui.py#L186),
    `_task`[`L189`](../../../../../../raw/code/simply/simply/agent/tui.py#L189),
    `_token_usage`[`L187`](../../../../../../raw/code/simply/simply/agent/tui.py#L187)
-   uses (calls/refs, reference-scoped):
    [`status_step`](memory.md#SystemStatus.status_step),
    [`DisplayBase`](tui.md#DisplayBase),
    [`SystemStatus`](memory.md#SystemStatus),
    [`approximate_token_usage`](memory.md#SystemStatus.approximate_token_usage),
    [`max_token_budget`](memory.md#SystemStatus.max_token_budget),
    [`_format_tokens`](tui.md#_format_tokens),
    [`elapsed_seconds`](memory.md#SystemStatus.elapsed_seconds),
    [`_format_elapsed`](tui.md#_format_elapsed)
-   used by: [`main`](main.md#main), [`DisplayBase`](tui.md#DisplayBase),
    [`update_system_status`](tui.md#DisplayBase.update_system_status),
    [`update_status`](tui.md#DisplayBase.update_status),
    [`display_llm_output`](tui.md#DisplayBase.display_llm_output),
    [`display_tool_call`](tui.md#DisplayBase.display_tool_call),
    [`set_header`](tui.md#DisplayBase.set_header),
    [`set_task`](tui.md#DisplayBase.set_task),
    [`start`](tui.md#DisplayBase.start), [`stop`](tui.md#DisplayBase.stop),
    [`print_header`](tui.md#DisplayBase.print_header)

## Functions

-   `_fmt(n: int)` —
    [`L330`](../../../../../../raw/code/simply/simply/agent/tui.py#L330)
-   `_format_elapsed(seconds: float)` —
    [`L312`](../../../../../../raw/code/simply/simply/agent/tui.py#L312) —
    Format elapsed seconds as a human-readable string. — documented in
    [simply-agent-tui](../../../concepts/simply-agent-tui.md)
-   `_format_tokens(usage: int, budget: int)` —
    [`L327`](../../../../../../raw/code/simply/simply/agent/tui.py#L327) —
    Format token usage as e.g. '12.3k / 200k'. — documented in
    [simply-agent-tui](../../../concepts/simply-agent-tui.md)
