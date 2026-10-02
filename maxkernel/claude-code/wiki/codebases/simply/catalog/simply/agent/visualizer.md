---
title: 'Module: simply/agent/visualizer.py'
type: catalog
provenance: extracted
module: simply/agent/visualizer.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.agent.visualizer`/
symbols:
  generate_html: generate_html().
  generate_progress_section: generate_progress_section().
  format_memory_file: format_memory_file().
  generate_token_usage_plot: generate_token_usage_plot().
  format_event: format_event().
  _BASH_CMD_REGEX: _BASH_CMD_REGEX.
  _BASH_COMMANDS: _BASH_COMMANDS.
  _CSS: _CSS.
  _JS: _JS.
---

# Module: [`simply/agent/visualizer.py`](../../../../../../raw/code/simply/simply/agent/visualizer.py)

## Functions

-   `format_event(file: memory_lib.MemoryFile)` —
    [`L260`](../../../../../../raw/code/simply/simply/agent/visualizer.py#L260)
    — Formats a log event for the holistic view.
-   `format_memory_file(file: memory_lib.MemoryFile, display_override:
    memory_lib.DisplayMode | None = None)` —
    [`L235`](../../../../../../raw/code/simply/simply/agent/visualizer.py#L235)
    — Formats a MemoryFile as an interactive HTML details block.
-   `generate_html(mem_system: memory_lib.MemorySystem, out_path: epath.Path,
    agent_info: dict[str, str] | None = None, trajectory_critique: str | None =
    None, progress_log: list[memory_lib.ProgressEntry] | None = None)` —
    [`L535`](../../../../../../raw/code/simply/simply/agent/visualizer.py#L535)
    — Generates a holistic HTML overview of the conversation.
-   `generate_progress_section(progress_log: list[memory_lib.ProgressEntry])` —
    [`L359`](../../../../../../raw/code/simply/simply/agent/visualizer.py#L359)
    — Generates an HTML section with a Chart.js plot and table for progress.
-   `generate_token_usage_plot(snapshots: list[memory_lib.MemorySnapshot])` —
    [`L297`](../../../../../../raw/code/simply/simply/agent/visualizer.py#L297)
    — Generates a Chart.js plot for token usage against steps.

## Module values

-   `_BASH_CMD_REGEX` —
    [`L52`](../../../../../../raw/code/simply/simply/agent/visualizer.py#L52)
-   `_BASH_COMMANDS` —
    [`L25`](../../../../../../raw/code/simply/simply/agent/visualizer.py#L25)
-   `_CSS` —
    [`L57`](../../../../../../raw/code/simply/simply/agent/visualizer.py#L57)
-   `_JS` —
    [`L206`](../../../../../../raw/code/simply/simply/agent/visualizer.py#L206)
