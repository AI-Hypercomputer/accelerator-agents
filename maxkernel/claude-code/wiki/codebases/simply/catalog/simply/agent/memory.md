---
title: 'Module: simply/agent/memory.py'
type: catalog
provenance: extracted
module: simply/agent/memory.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.agent.memory`/
symbols:
  MemorySystem.__post_init__: MemorySystem#__post_init__().
  MemorySystem.compress_history: MemorySystem#compress_history().
  get_memory_tools: get_memory_tools().
  MemorySystem.record_llm_output: MemorySystem#record_llm_output().
  MemorySystem.write: MemorySystem#write().
  MemorySystem.record_tool_call: MemorySystem#record_tool_call().
  MemorySystem.record_progress: MemorySystem#record_progress().
  MemoryFile.to_llm: MemoryFile#to_llm().
  MemorySystem._record_memory_snapshot: MemorySystem#_record_memory_snapshot().
  MemorySystem.capture_snapshot: MemorySystem#capture_snapshot().
  MemorySystem.memory: MemorySystem#memory.
  MemorySystem.get_display_mode: MemorySystem#get_display_mode().
  DisplayMode: DisplayMode#
  MemoryFile: MemoryFile#
  Memory.files: Memory#files.
  MemorySystem.llm_view: MemorySystem#llm_view().
  MemorySystem._make_system_status: MemorySystem#_make_system_status().
  MemoryFile.uri: MemoryFile#uri.
  MemoryFile.content: MemoryFile#content.
  MemorySystem.memory_snapshots: MemorySystem#memory_snapshots.
  MemorySnapshot.system_status: MemorySnapshot#system_status.
  memory_compress_history: memory_compress_history().
  MemorySystem.last_snapshot: MemorySystem#last_snapshot().
  MemorySystem.get_events_for_step: MemorySystem#get_events_for_step().
  MemorySystem: MemorySystem#
  MemorySystem._memory_to_llm_view: MemorySystem#_memory_to_llm_view().
  MemorySystem._log_to_llm_view: MemorySystem#_log_to_llm_view().
  memory_fold: memory_fold().
  memory_unfold: memory_unfold().
  memory_write: memory_write().
  MemoryFile.summary: MemoryFile#summary.
  SystemStatus.status_step: SystemStatus#status_step.
  MemoryFile.display: MemoryFile#display.
  MemorySystem.set_display: MemorySystem#set_display().
  MemorySnapshot.memory: MemorySnapshot#memory.
  MemorySystem._user_files_to_llm_view: MemorySystem#_user_files_to_llm_view().
  record_progress: record_progress().
  MemoryFile.metadata: MemoryFile#metadata.
  MemoryFile.update_step: MemoryFile#update_step.
  MemorySystem.generate_summary: MemorySystem#generate_summary().
  memory_delete: memory_delete().
  MemorySystem.delete: MemorySystem#delete().
  DisplayMode.SUMMARY: DisplayMode#SUMMARY.
  DisplayMode.FULL: DisplayMode#FULL.
  MemorySystem.task: MemorySystem#task.
  MemorySystem.progress_log: MemorySystem#progress_log.
  SystemStatus: SystemStatus#
  CONTEXT_ERROR_MSG: CONTEXT_ERROR_MSG.
  Memory: Memory#
  MemorySystem.max_token_budget: MemorySystem#max_token_budget.
  MemoryDisplayAction: MemoryDisplayAction#
  SystemStatus.max_token_budget: SystemStatus#max_token_budget.
  SystemStatus.approximate_token_usage: SystemStatus#approximate_token_usage.
  Memory.log_order: Memory#log_order.
  MemorySystem.replacable: MemorySystem#replacable().
  is_valid_memory_uri: is_valid_memory_uri().
  ProgressEntry.metrics: ProgressEntry#metrics.
  MemoryFile.__post_init__: MemoryFile#__post_init__().
  SystemStatus.elapsed_seconds: SystemStatus#elapsed_seconds.
  MemorySystem.system_prompt: MemorySystem#system_prompt.
  MemorySystem.predefined_knowledge: MemorySystem#predefined_knowledge.
  MemoryWriteAction: MemoryWriteAction#
  MemoryDeleteAction: MemoryDeleteAction#
  MemoryCompressHistoryAction: MemoryCompressHistoryAction#
  RecordProgressAction: RecordProgressAction#
  ProgressEntry: ProgressEntry#
  MemorySnapshot: MemorySnapshot#
  MemorySystem.memory_system_description: MemorySystem#memory_system_description.
  RecordProgressAction.metrics: RecordProgressAction#metrics.
  REGEX_MEMORY_URI: REGEX_MEMORY_URI.
  approximate_token_count: approximate_token_count().
  MetricEntry: MetricEntry#
  ProgressEntry.step: ProgressEntry#step.
  MetricEntry.variant: MetricEntry#variant.
  MetricEntry.metric: MetricEntry#metric.
  MetricEntry.value: MetricEntry#value.
  ProgressEntry.description: ProgressEntry#description.
  MemorySnapshot.llm_view: MemorySnapshot#llm_view.
  MemorySystem.default_summary_length: MemorySystem#default_summary_length.
  MemorySystem.start_time: MemorySystem#start_time.
  MemoryDisplayAction.uri: MemoryDisplayAction#uri.
  SCHEME_LABELS.SCHEME_LABELS: SCHEME_LABELS.SCHEME_LABELS.
  SYSTEM_PROMPT: SYSTEM_PROMPT.
  _MODULE_DIR: _MODULE_DIR.
  escape_xml: escape_xml.
  escape_cdata: escape_cdata.
  SystemStatus.current_time: SystemStatus#current_time.
  MemorySystem.display_memory_tool_calls: MemorySystem#display_memory_tool_calls.
  MemorySystem.n_full_display_recent_steps: MemorySystem#n_full_display_recent_steps.
  MemoryWriteAction.uri: MemoryWriteAction#uri.
  MemoryWriteAction.content: MemoryWriteAction#content.
  MemoryWriteAction.summary: MemoryWriteAction#summary.
  MemoryDeleteAction.uri: MemoryDeleteAction#uri.
  MemoryCompressHistoryAction.start_step: MemoryCompressHistoryAction#start_step.
  MemoryCompressHistoryAction.end_step: MemoryCompressHistoryAction#end_step.
  MemoryCompressHistoryAction.content: MemoryCompressHistoryAction#content.
  MemoryCompressHistoryAction.summary: MemoryCompressHistoryAction#summary.
  RecordProgressAction.description: RecordProgressAction#description.
  SystemStatus.model_config: SystemStatus#model_config.
  MetricEntry.model_config: MetricEntry#model_config.
  ProgressEntry.model_config: ProgressEntry#model_config.
---

# Module: [`simply/agent/memory.py`](../../../../../../raw/code/simply/simply/agent/memory.py)

## Classes

### `DisplayMode` · implements/extends StrEnum

-   def:
    [`simply/agent/memory.py:74`](../../../../../../raw/code/simply/simply/agent/memory.py#L74)
-   signature: `class DisplayMode(enum.StrEnum):`
-   members:
    -   `FULL` —
        [`L76`](../../../../../../raw/code/simply/simply/agent/memory.py#L76) —
        documented in
        [simply-agent-memory](../../../concepts/simply-agent-memory.md)
    -   `SUMMARY` —
        [`L75`](../../../../../../raw/code/simply/simply/agent/memory.py#L75)
-   used by: [`load_skills`](skill_loader.md#load_skills),
    [`__post_init__`](memory.md#MemorySystem.__post_init__),
    [`compress_history`](memory.md#MemorySystem.compress_history),
    [`record_llm_output`](memory.md#MemorySystem.record_llm_output),
    [`write`](memory.md#MemorySystem.write),
    [`record_tool_call`](memory.md#MemorySystem.record_tool_call),
    [`load_skills_cached`](skill_loader.md#load_skills_cached),
    [`to_llm`](memory.md#MemoryFile.to_llm),
    [`test_memory_file_to_llm`](memory_test.md#MemoryTest.test_memory_file_to_llm),
    [`get_display_mode`](memory.md#MemorySystem.get_display_mode),
    [`format_memory_file`](visualizer.md#format_memory_file),
    [`memory_fold`](memory.md#memory_fold),
    [`memory_unfold`](memory.md#memory_unfold),
    [`display`](memory.md#MemoryFile.display),
    [`set_display`](memory.md#MemorySystem.set_display),
    [`_format_step_events`](critique.md#_format_step_events),
    [`parse_predefined_knowledge`](main.md#parse_predefined_knowledge)

### `Memory`

-   def:
    [`simply/agent/memory.py:175`](../../../../../../raw/code/simply/simply/agent/memory.py#L175)
    — documented in
    [simply-agent-memory](../../../concepts/simply-agent-memory.md)
-   doc: The memory of the agent.
-   signature: `class Memory:`
-   members:
    -   `files` —
        [`L177`](../../../../../../raw/code/simply/simply/agent/memory.py#L177)
    -   `log_order` —
        [`L180`](../../../../../../raw/code/simply/simply/agent/memory.py#L180)
-   uses (calls/refs, reference-scoped): [`MemoryFile`](memory.md#MemoryFile)
-   used by: [`__post_init__`](memory.md#MemorySystem.__post_init__),
    [`compress_history`](memory.md#MemorySystem.compress_history),
    [`record_llm_output`](memory.md#MemorySystem.record_llm_output),
    [`write`](memory.md#MemorySystem.write),
    [`record_tool_call`](memory.md#MemorySystem.record_tool_call),
    [`test_write_knowledge`](memory_test.md#MemoryTest.test_write_knowledge),
    [`test_tool_call_error`](memory_test.md#MemoryTest.test_tool_call_error),
    [`_record_memory_snapshot`](memory.md#MemorySystem._record_memory_snapshot),
    [`memory`](memory.md#MemorySystem.memory),
    [`test_update_step`](memory_test.md#MemoryTest.test_update_step),
    [`get_events_for_step`](memory.md#MemorySystem.get_events_for_step),
    [`_log_to_llm_view`](memory.md#MemorySystem._log_to_llm_view),
    [`_memory_to_llm_view`](memory.md#MemorySystem._memory_to_llm_view),
    [`set_display`](memory.md#MemorySystem.set_display),
    [`_user_files_to_llm_view`](memory.md#MemorySystem._user_files_to_llm_view),
    [`memory`](memory.md#MemorySnapshot.memory),
    [`delete`](memory.md#MemorySystem.delete)

### `MemoryCompressHistoryAction` · implements/extends Action

-   def:
    [`simply/agent/memory.py:737`](../../../../../../raw/code/simply/simply/agent/memory.py#L737)
-   doc: An action to compress the history of memory.
-   signature: `class MemoryCompressHistoryAction(tools_lib.Action):`
-   members:
    -   `content` —
        [`L747`](../../../../../../raw/code/simply/simply/agent/memory.py#L747)
    -   `end_step` —
        [`L743`](../../../../../../raw/code/simply/simply/agent/memory.py#L743)
    -   `start_step` —
        [`L739`](../../../../../../raw/code/simply/simply/agent/memory.py#L739)
    -   `summary` —
        [`L750`](../../../../../../raw/code/simply/simply/agent/memory.py#L750)
-   uses (calls/refs, reference-scoped): [`Action`](tools.md#Action)
-   used by: [`get_memory_tools`](memory.md#get_memory_tools),
    [`Action`](tools.md#Action),
    [`memory_compress_history`](memory.md#memory_compress_history)

### `MemoryDeleteAction` · implements/extends Action

-   def:
    [`simply/agent/memory.py:724`](../../../../../../raw/code/simply/simply/agent/memory.py#L724)
-   doc: An action to delete a memory file.
-   signature: `class MemoryDeleteAction(tools_lib.Action):`
-   members:
    -   `uri` —
        [`L726`](../../../../../../raw/code/simply/simply/agent/memory.py#L726)
-   uses (calls/refs, reference-scoped): [`Action`](tools.md#Action)
-   used by: [`get_memory_tools`](memory.md#get_memory_tools),
    [`Action`](tools.md#Action), [`memory_delete`](memory.md#memory_delete)

### `MemoryDisplayAction` · implements/extends Action

-   def:
    [`simply/agent/memory.py:682`](../../../../../../raw/code/simply/simply/agent/memory.py#L682)
-   doc: An action to fold a memory file.
-   signature: `class MemoryDisplayAction(tools_lib.Action):`
-   members:
    -   `uri` —
        [`L684`](../../../../../../raw/code/simply/simply/agent/memory.py#L684)
-   uses (calls/refs, reference-scoped): [`Action`](tools.md#Action)
-   used by: [`get_memory_tools`](memory.md#get_memory_tools),
    [`Action`](tools.md#Action), [`memory_fold`](memory.md#memory_fold),
    [`memory_unfold`](memory.md#memory_unfold)

### `MemoryFile`

-   def:
    [`simply/agent/memory.py:80`](../../../../../../raw/code/simply/simply/agent/memory.py#L80)
    — documented in
    [simply-agent-memory](../../../concepts/simply-agent-memory.md)
-   doc: A file in the memory.
-   signature: `class MemoryFile:`
-   members:
    -   `to_llm(self, force_display: DisplayMode | None = None)` —
        [`L95`](../../../../../../raw/code/simply/simply/agent/memory.py#L95) —
        Returns the string representation of the file for LLM consumption. —
        documented in
        [simply-agent-memory](../../../concepts/simply-agent-memory.md)
    -   `content` —
        [`L84`](../../../../../../raw/code/simply/simply/agent/memory.py#L84)
    -   `display` —
        [`L83`](../../../../../../raw/code/simply/simply/agent/memory.py#L83)
    -   `metadata` —
        [`L87`](../../../../../../raw/code/simply/simply/agent/memory.py#L87)
    -   `summary` —
        [`L85`](../../../../../../raw/code/simply/simply/agent/memory.py#L85)
    -   `update_step` —
        [`L86`](../../../../../../raw/code/simply/simply/agent/memory.py#L86)
    -   `uri` —
        [`L82`](../../../../../../raw/code/simply/simply/agent/memory.py#L82)
-   protocol/private:
    `__post_init__`[`L91`](../../../../../../raw/code/simply/simply/agent/memory.py#L91)
-   uses (calls/refs, reference-scoped): [`DisplayMode`](memory.md#DisplayMode),
    [`FULL`](memory.md#DisplayMode.FULL),
    [`is_valid_memory_uri`](memory.md#is_valid_memory_uri),
    [`escape_cdata`](memory.md#escape_cdata),
    [`escape_xml`](memory.md#escape_xml)
-   used by: [`load_skills`](skill_loader.md#load_skills),
    [`__post_init__`](memory.md#MemorySystem.__post_init__),
    [`generate_html`](visualizer.md#generate_html),
    [`compress_history`](memory.md#MemorySystem.compress_history),
    [`record_llm_output`](memory.md#MemorySystem.record_llm_output),
    [`write`](memory.md#MemorySystem.write),
    [`record_tool_call`](memory.md#MemorySystem.record_tool_call),
    [`test_write_knowledge`](memory_test.md#MemoryTest.test_write_knowledge),
    [`load_skills_cached`](skill_loader.md#load_skills_cached),
    [`test_tool_call_error`](memory_test.md#MemoryTest.test_tool_call_error),
    [`test_memory_file_to_llm`](memory_test.md#MemoryTest.test_memory_file_to_llm),
    [`get_display_mode`](memory.md#MemorySystem.get_display_mode),
    [`add_step`](critique.md#TrajectoryCritique.add_step),
    [`files`](memory.md#Memory.files),
    [`test_update_step`](memory_test.md#MemoryTest.test_update_step),
    [`format_memory_file`](visualizer.md#format_memory_file),
    [`get_events_for_step`](memory.md#MemorySystem.get_events_for_step),
    [`_log_to_llm_view`](memory.md#MemorySystem._log_to_llm_view),
    [`_user_files_to_llm_view`](memory.md#MemorySystem._user_files_to_llm_view),
    [`_format_step_events`](critique.md#_format_step_events),
    [`format_event`](visualizer.md#format_event),
    [`parse_predefined_knowledge`](main.md#parse_predefined_knowledge),
    [`replacable`](memory.md#MemorySystem.replacable),
    [`predefined_knowledge`](agent.md#Agent.predefined_knowledge),
    [`predefined_knowledge`](memory.md#MemorySystem.predefined_knowledge)

### `MemorySnapshot`

-   def:
    [`simply/agent/memory.py:184`](../../../../../../raw/code/simply/simply/agent/memory.py#L184)
-   doc: The memory snapshot.
-   signature: `class MemorySnapshot:`
-   members:
    -   `llm_view` —
        [`L188`](../../../../../../raw/code/simply/simply/agent/memory.py#L188)
    -   `memory` —
        [`L186`](../../../../../../raw/code/simply/simply/agent/memory.py#L186)
    -   `system_status` —
        [`L187`](../../../../../../raw/code/simply/simply/agent/memory.py#L187)
-   uses (calls/refs, reference-scoped): [`Memory`](memory.md#Memory),
    [`SystemStatus`](memory.md#SystemStatus)
-   used by: [`generate_html`](visualizer.md#generate_html),
    [`step`](agent.md#Agent.step), [`run_agent_loop`](main.md#run_agent_loop),
    [`compress_history`](memory.md#MemorySystem.compress_history),
    [`record_llm_output`](memory.md#MemorySystem.record_llm_output),
    [`write`](memory.md#MemorySystem.write),
    [`record_tool_call`](memory.md#MemorySystem.record_tool_call),
    [`test_write_knowledge`](memory_test.md#MemoryTest.test_write_knowledge),
    [`record_progress`](memory.md#MemorySystem.record_progress),
    [`test_tool_call_error`](memory_test.md#MemoryTest.test_tool_call_error),
    [`_record_memory_snapshot`](memory.md#MemorySystem._record_memory_snapshot),
    [`capture_snapshot`](memory.md#MemorySystem.capture_snapshot),
    [`test_update_step`](memory_test.md#MemoryTest.test_update_step),
    [`llm_view`](memory.md#MemorySystem.llm_view),
    [`memory_snapshots`](memory.md#MemorySystem.memory_snapshots),
    [`save_memory_snapshot`](agent.md#Agent.save_memory_snapshot),
    [`get_events_for_step`](memory.md#MemorySystem.get_events_for_step),
    [`last_snapshot`](memory.md#MemorySystem.last_snapshot),
    [`generate_token_usage_plot`](visualizer.md#generate_token_usage_plot)

### `MemorySystem`

-   def:
    [`simply/agent/memory.py:192`](../../../../../../raw/code/simply/simply/agent/memory.py#L192)
    — documented in
    [simply-agent-memory](../../../concepts/simply-agent-memory.md)
-   doc: The memory system.
-   signature: `class MemorySystem:`
-   members:
    -   `_log_to_llm_view(self, memory: Memory, snapshot_step: int)` —
        [`L385`](../../../../../../raw/code/simply/simply/agent/memory.py#L385)
        — Returns the LLM view of the event log in memory. — documented in
        [simply-agent-memory](../../../concepts/simply-agent-memory.md)
    -   `_make_system_status(self, approximate_token_usage: int, status_step:
        int)` —
        [`L413`](../../../../../../raw/code/simply/simply/agent/memory.py#L413)
        — Gets the current system status.
    -   `_memory_to_llm_view(self, memory: Memory, snapshot_step: int)` —
        [`L369`](../../../../../../raw/code/simply/simply/agent/memory.py#L369)
        — Returns the LLM view of the memory. — documented in
        [simply-agent-memory](../../../concepts/simply-agent-memory.md)
    -   `_record_memory_snapshot(self, memory: Memory, snapshot_step: int,
        token_counter: Callable[[str], int] = approximate_token_count)` —
        [`L335`](../../../../../../raw/code/simply/simply/agent/memory.py#L335)
        — Makes a memory snapshot and push to `self.memory_snapshots`. —
        documented in
        [simply-agent-memory](../../../concepts/simply-agent-memory.md)
    -   `_user_files_to_llm_view(self, memory: Memory, scheme: str,
        snapshot_step: int)` —
        [`L394`](../../../../../../raw/code/simply/simply/agent/memory.py#L394)
        — Returns the LLM view of the user files in memory. — documented in
        [simply-agent-memory](../../../concepts/simply-agent-memory.md)
    -   `capture_snapshot(self, token_counter: Callable[[str], int] =
        approximate_token_count)` —
        [`L440`](../../../../../../raw/code/simply/simply/agent/memory.py#L440)
        — Enter the context of capturing a memory snapshot.
    -   `compress_history(self, start_step: int, end_step: int, content: str,
        summary: str | None = None)` —
        [`L605`](../../../../../../raw/code/simply/simply/agent/memory.py#L605)
        — Replace the history from start_step to end_step with a summary step. —
        documented in
        [simply-agent-memory](../../../concepts/simply-agent-memory.md)
    -   `delete(self, uri: str)` —
        [`L594`](../../../../../../raw/code/simply/simply/agent/memory.py#L594)
        — Deletes a memory file. — documented in
        [simply-agent-memory](../../../concepts/simply-agent-memory.md)
    -   `generate_summary(self, content: str)` —
        [`L409`](../../../../../../raw/code/simply/simply/agent/memory.py#L409)
        — Generates the summary for the content.
    -   `get_display_mode(self, file: MemoryFile, display_step: int)` —
        [`L298`](../../../../../../raw/code/simply/simply/agent/memory.py#L298)
        — Returns the display mode of the file for a given step. — documented in
        [simply-agent-memory](../../../concepts/simply-agent-memory.md)
    -   `get_events_for_step(self, step: int)` —
        [`L470`](../../../../../../raw/code/simply/simply/agent/memory.py#L470)
        — Returns all log events belonging to a particular step.
    -   `last_snapshot(self)` —
        [`L287`](../../../../../../raw/code/simply/simply/agent/memory.py#L287)
        — Returns the last memory snapshot.
    -   `llm_view(self)` —
        [`L294`](../../../../../../raw/code/simply/simply/agent/memory.py#L294)
        — The LLM view of the last memory snapshot.
    -   `record_llm_output(self, content: str)` —
        [`L490`](../../../../../../raw/code/simply/simply/agent/memory.py#L490)
        — Records the LLM response in the memory. — documented in
        [simply-agent-memory](../../../concepts/simply-agent-memory.md)
    -   `record_progress(self, description: str, metrics: list[MetricEntry])` —
        [`L658`](../../../../../../raw/code/simply/simply/agent/memory.py#L658)
        — Records a progress entry. — documented in
        [simply-agent-memory](../../../concepts/simply-agent-memory.md)
    -   `record_tool_call(self, tool_name: str, index: int, inputs: str,
        outputs: str)` —
        [`L511`](../../../../../../raw/code/simply/simply/agent/memory.py#L511)
        — Records the tool call in the memory.
    -   `replacable(file: MemoryFile)` —
        [`L618`](../../../../../../raw/code/simply/simply/agent/memory.py#L618)
    -   `set_display(self, uri: str, display: DisplayMode)` —
        [`L557`](../../../../../../raw/code/simply/simply/agent/memory.py#L557)
        — Sets the display mode of a memory file. — documented in
        [simply-agent-memory](../../../concepts/simply-agent-memory.md)
    -   `write(self, uri: str, content: str, summary: str | None = None)` —
        [`L568`](../../../../../../raw/code/simply/simply/agent/memory.py#L568)
        — Writes a memory file. — documented in
        [simply-agent-memory](../../../concepts/simply-agent-memory.md)
    -   `default_summary_length` —
        [`L217`](../../../../../../raw/code/simply/simply/agent/memory.py#L217)
    -   `display_memory_tool_calls` —
        [`L220`](../../../../../../raw/code/simply/simply/agent/memory.py#L220)
    -   `max_token_budget` —
        [`L209`](../../../../../../raw/code/simply/simply/agent/memory.py#L209)
    -   `memory` —
        [`L236`](../../../../../../raw/code/simply/simply/agent/memory.py#L236)
    -   `memory_snapshots` —
        [`L230`](../../../../../../raw/code/simply/simply/agent/memory.py#L230)
        — documented in
        [simply-agent-memory](../../../concepts/simply-agent-memory.md)
    -   `memory_system_description` —
        [`L241`](../../../../../../raw/code/simply/simply/agent/memory.py#L241)
    -   `n_full_display_recent_steps` —
        [`L224`](../../../../../../raw/code/simply/simply/agent/memory.py#L224)
    -   `predefined_knowledge` —
        [`L214`](../../../../../../raw/code/simply/simply/agent/memory.py#L214)
    -   `progress_log` —
        [`L244`](../../../../../../raw/code/simply/simply/agent/memory.py#L244)
    -   `start_time` —
        [`L233`](../../../../../../raw/code/simply/simply/agent/memory.py#L233)
    -   `system_prompt` —
        [`L211`](../../../../../../raw/code/simply/simply/agent/memory.py#L211)
    -   `task` —
        [`L210`](../../../../../../raw/code/simply/simply/agent/memory.py#L210)
-   protocol/private:
    `__post_init__`[`L248`](../../../../../../raw/code/simply/simply/agent/memory.py#L248)
-   uses (calls/refs, reference-scoped):
    [`to_llm`](memory.md#MemoryFile.to_llm),
    [`DisplayMode`](memory.md#DisplayMode),
    [`MemoryFile`](memory.md#MemoryFile), [`files`](memory.md#Memory.files),
    [`content`](memory.md#MemoryFile.content),
    [`uri`](memory.md#MemoryFile.uri),
    [`system_status`](memory.md#MemorySnapshot.system_status),
    [`status_step`](memory.md#SystemStatus.status_step),
    [`summary`](memory.md#MemoryFile.summary),
    [`display`](memory.md#MemoryFile.display),
    [`memory`](memory.md#MemorySnapshot.memory),
    [`metadata`](memory.md#MemoryFile.metadata),
    [`update_step`](memory.md#MemoryFile.update_step),
    [`SUMMARY`](memory.md#DisplayMode.SUMMARY),
    [`FULL`](memory.md#DisplayMode.FULL), [`Memory`](memory.md#Memory),
    [`SystemStatus`](memory.md#SystemStatus),
    [`CONTEXT_ERROR_MSG`](memory.md#CONTEXT_ERROR_MSG),
    [`approximate_token_usage`](memory.md#SystemStatus.approximate_token_usage),
    [`log_order`](memory.md#Memory.log_order),
    [`max_token_budget`](memory.md#SystemStatus.max_token_budget),
    [`metrics`](memory.md#ProgressEntry.metrics),
    [`elapsed_seconds`](memory.md#SystemStatus.elapsed_seconds),
    [`MemorySnapshot`](memory.md#MemorySnapshot),
    [`ProgressEntry`](memory.md#ProgressEntry),
    [`MetricEntry`](memory.md#MetricEntry),
    [`REGEX_MEMORY_URI`](memory.md#REGEX_MEMORY_URI),
    [`approximate_token_count`](memory.md#approximate_token_count),
    [`step`](memory.md#ProgressEntry.step),
    [`SCHEME_LABELS`](memory.md#SCHEME_LABELS.SCHEME_LABELS),
    [`description`](memory.md#ProgressEntry.description),
    [`llm_view`](memory.md#MemorySnapshot.llm_view),
    [`truncate_text`](tools.md#truncate_text),
    [`SYSTEM_PROMPT`](memory.md#SYSTEM_PROMPT),
    [`_MODULE_DIR`](memory.md#_MODULE_DIR),
    [`current_time`](memory.md#SystemStatus.current_time)
-   used by: [`generate_html`](visualizer.md#generate_html),
    [`step`](agent.md#Agent.step), [`run_agent_loop`](main.md#run_agent_loop),
    [`get_memory_tools`](memory.md#get_memory_tools),
    [`__post_init__`](agent.md#Agent.__post_init__),
    [`test_write_knowledge`](memory_test.md#MemoryTest.test_write_knowledge),
    [`test_tool_call_error`](memory_test.md#MemoryTest.test_tool_call_error),
    [`test_update_step`](memory_test.md#MemoryTest.test_update_step),
    [`test_compress_history`](memory_test.md#MemoryTest.test_compress_history),
    [`memory_system`](agent.md#Agent.memory_system),
    [`save_memory_snapshot`](agent.md#Agent.save_memory_snapshot),
    [`make_tool_call`](agent.md#Agent.make_tool_call),
    [`memory_compress_history`](memory.md#memory_compress_history),
    [`memory_fold`](memory.md#memory_fold),
    [`memory_unfold`](memory.md#memory_unfold),
    [`memory_write`](memory.md#memory_write),
    [`record_progress`](memory.md#record_progress),
    [`restore_memory_system`](agent.md#Agent.restore_memory_system),
    [`test_llm_view`](memory_test.md#MemoryTest.test_llm_view),
    [`memory_delete`](memory.md#memory_delete)

### `MemoryWriteAction` · implements/extends Action

-   def:
    [`simply/agent/memory.py:695`](../../../../../../raw/code/simply/simply/agent/memory.py#L695)
-   doc: An action to write a memory file.
-   signature: `class MemoryWriteAction(tools_lib.Action):`
-   members:
    -   `content` —
        [`L706`](../../../../../../raw/code/simply/simply/agent/memory.py#L706)
    -   `summary` —
        [`L709`](../../../../../../raw/code/simply/simply/agent/memory.py#L709)
    -   `uri` —
        [`L697`](../../../../../../raw/code/simply/simply/agent/memory.py#L697)
-   uses (calls/refs, reference-scoped): [`Action`](tools.md#Action)
-   used by: [`get_memory_tools`](memory.md#get_memory_tools),
    [`Action`](tools.md#Action), [`memory_write`](memory.md#memory_write)

### `MetricEntry` · implements/extends BaseModel

-   def:
    [`simply/agent/memory.py:132`](../../../../../../raw/code/simply/simply/agent/memory.py#L132)
-   doc: A single metric measurement for a specific variant.
-   signature: `class MetricEntry(pydantic.BaseModel):`
-   members:
    -   `metric` —
        [`L142`](../../../../../../raw/code/simply/simply/agent/memory.py#L142)
    -   `model_config` —
        [`L134`](../../../../../../raw/code/simply/simply/agent/memory.py#L134)
    -   `value` —
        [`L151`](../../../../../../raw/code/simply/simply/agent/memory.py#L151)
    -   `variant` —
        [`L136`](../../../../../../raw/code/simply/simply/agent/memory.py#L136)
-   used by: [`record_progress`](memory.md#MemorySystem.record_progress),
    [`generate_progress_section`](visualizer.md#generate_progress_section),
    [`metrics`](memory.md#ProgressEntry.metrics),
    [`metrics`](memory.md#RecordProgressAction.metrics)

### `ProgressEntry` · implements/extends BaseModel

-   def:
    [`simply/agent/memory.py:157`](../../../../../../raw/code/simply/simply/agent/memory.py#L157)
-   doc: A progress log entry recorded by the agent.
-   signature: `class ProgressEntry(pydantic.BaseModel):`
-   members:
    -   `description` —
        [`L162`](../../../../../../raw/code/simply/simply/agent/memory.py#L162)
    -   `metrics` —
        [`L163`](../../../../../../raw/code/simply/simply/agent/memory.py#L163)
    -   `model_config` —
        [`L159`](../../../../../../raw/code/simply/simply/agent/memory.py#L159)
    -   `step` —
        [`L161`](../../../../../../raw/code/simply/simply/agent/memory.py#L161)
-   uses (calls/refs, reference-scoped): [`MetricEntry`](memory.md#MetricEntry)
-   used by: [`generate_html`](visualizer.md#generate_html),
    [`record_progress`](memory.md#MemorySystem.record_progress),
    [`generate_progress_section`](visualizer.md#generate_progress_section),
    [`progress_log`](memory.md#MemorySystem.progress_log)

### `RecordProgressAction` · implements/extends Action

-   def:
    [`simply/agent/memory.py:769`](../../../../../../raw/code/simply/simply/agent/memory.py#L769)
-   doc: An action to record a progress entry with metrics for the current step.
-   signature: `class RecordProgressAction(tools_lib.Action):`
-   members:
    -   `description` —
        [`L772`](../../../../../../raw/code/simply/simply/agent/memory.py#L772)
    -   `metrics` —
        [`L781`](../../../../../../raw/code/simply/simply/agent/memory.py#L781)
-   uses (calls/refs, reference-scoped): [`Action`](tools.md#Action),
    [`MetricEntry`](memory.md#MetricEntry)
-   used by: [`get_memory_tools`](memory.md#get_memory_tools),
    [`Action`](tools.md#Action), [`record_progress`](memory.md#record_progress)

### `SystemStatus` · implements/extends BaseModel

-   def:
    [`simply/agent/memory.py:114`](../../../../../../raw/code/simply/simply/agent/memory.py#L114)
    — documented in
    [simply-agent-memory](../../../concepts/simply-agent-memory.md)
-   doc: The status of the agent system.
-   signature: `class SystemStatus(pydantic.BaseModel):`
-   members:
    -   `approximate_token_usage` —
        [`L118`](../../../../../../raw/code/simply/simply/agent/memory.py#L118)
    -   `current_time` —
        [`L126`](../../../../../../raw/code/simply/simply/agent/memory.py#L126)
    -   `elapsed_seconds` —
        [`L127`](../../../../../../raw/code/simply/simply/agent/memory.py#L127)
    -   `max_token_budget` —
        [`L117`](../../../../../../raw/code/simply/simply/agent/memory.py#L117)
    -   `model_config` —
        [`L116`](../../../../../../raw/code/simply/simply/agent/memory.py#L116)
    -   `status_step` —
        [`L125`](../../../../../../raw/code/simply/simply/agent/memory.py#L125)
-   used by: [`generate_html`](visualizer.md#generate_html),
    [`step`](agent.md#Agent.step), [`run_agent_loop`](main.md#run_agent_loop),
    [`compress_history`](memory.md#MemorySystem.compress_history),
    [`record_llm_output`](memory.md#MemorySystem.record_llm_output),
    [`write`](memory.md#MemorySystem.write),
    [`record_tool_call`](memory.md#MemorySystem.record_tool_call),
    [`record_progress`](memory.md#MemorySystem.record_progress),
    [`update_system_status`](tui.md#StatusDisplay.update_system_status),
    [`capture_snapshot`](memory.md#MemorySystem.capture_snapshot),
    [`_make_system_status`](memory.md#MemorySystem._make_system_status),
    [`update_system_status`](tui.md#FullDisplay.update_system_status),
    [`save_memory_snapshot`](agent.md#Agent.save_memory_snapshot),
    [`system_status`](memory.md#MemorySnapshot.system_status),
    [`update_system_status`](tui.md#PrintDisplay.update_system_status),
    [`generate_token_usage_plot`](visualizer.md#generate_token_usage_plot),
    [`update_system_status`](tui.md#DisplayBase.update_system_status)

## Functions

-   `approximate_token_count(text: str)` —
    [`L66`](../../../../../../raw/code/simply/simply/agent/memory.py#L66) —
    Returns the approximate token count of the text.
-   `get_memory_tools(memory_system: MemorySystem)` —
    [`L798`](../../../../../../raw/code/simply/simply/agent/memory.py#L798) —
    Returns the memory tools. — documented in
    [simply-agent-memory](../../../concepts/simply-agent-memory.md)
-   `is_valid_memory_uri(uri: str)` —
    [`L61`](../../../../../../raw/code/simply/simply/agent/memory.py#L61) —
    Checks if the memory URI is valid.
-   `memory_compress_history(action: MemoryCompressHistoryAction, mem_system:
    MemorySystem)` —
    [`L761`](../../../../../../raw/code/simply/simply/agent/memory.py#L761) —
    documented in
    [simply-agent-memory](../../../concepts/simply-agent-memory.md)
-   `memory_delete(action: MemoryDeleteAction, mem_system: MemorySystem)` —
    [`L731`](../../../../../../raw/code/simply/simply/agent/memory.py#L731)
-   `memory_fold(action: MemoryDisplayAction, mem_system: MemorySystem)` —
    [`L687`](../../../../../../raw/code/simply/simply/agent/memory.py#L687) —
    documented in
    [simply-agent-memory](../../../concepts/simply-agent-memory.md)
-   `memory_unfold(action: MemoryDisplayAction, mem_system: MemorySystem)` —
    [`L691`](../../../../../../raw/code/simply/simply/agent/memory.py#L691)
-   `memory_write(action: MemoryWriteAction, mem_system: MemorySystem)` —
    [`L720`](../../../../../../raw/code/simply/simply/agent/memory.py#L720) —
    documented in
    [simply-agent-memory](../../../concepts/simply-agent-memory.md)
-   `record_progress(action: RecordProgressAction, mem_system: MemorySystem)` —
    [`L791`](../../../../../../raw/code/simply/simply/agent/memory.py#L791) —
    documented in
    [simply-agent-memory](../../../concepts/simply-agent-memory.md)

## Module values

-   `CONTEXT_ERROR_MSG` —
    [`L169`](../../../../../../raw/code/simply/simply/agent/memory.py#L169) —
    documented in
    [simply-agent-memory](../../../concepts/simply-agent-memory.md)
-   `REGEX_MEMORY_URI` —
    [`L58`](../../../../../../raw/code/simply/simply/agent/memory.py#L58)
-   `SCHEME_LABELS` —
    [`L52`](../../../../../../raw/code/simply/simply/agent/memory.py#L52)
-   `SYSTEM_PROMPT` —
    [`L32`](../../../../../../raw/code/simply/simply/agent/memory.py#L32)
-   `_MODULE_DIR` —
    [`L41`](../../../../../../raw/code/simply/simply/agent/memory.py#L41)
-   `escape_cdata` —
    [`L48`](../../../../../../raw/code/simply/simply/agent/memory.py#L48)
-   `escape_xml` —
    [`L47`](../../../../../../raw/code/simply/simply/agent/memory.py#L47)
