---
title: 'Module: easydel/trainers/agentic_moshpit/tools.py'
type: catalog
provenance: extracted
module: easydel/trainers/agentic_moshpit/tools.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.agentic_moshpit.tools`/
symbols:
  Tool.call: Tool#call().
  Tool: Tool#
  register_tool: register_tool().
  Tool.schema: Tool#schema().
  FunctionTool.schema: FunctionTool#schema().
  _TOOL_REGISTRY._TOOL_REGISTRY: _TOOL_REGISTRY._TOOL_REGISTRY.
  make_tool: make_tool().
  PythonCodeTool.call: PythonCodeTool#call().
  BashTool.call: BashTool#call().
  Tool.name: Tool#name().
  register_tool.decorator: register_tool().decorator().
  Tool.execute: Tool#execute().
  PythonCodeTool: PythonCodeTool#
  CalculatorTool: CalculatorTool#
  BashTool: BashTool#
  RegexTool: RegexTool#
  JSONProcessorTool: JSONProcessorTool#
  WikipediaTool: WikipediaTool#
  WebFetchTool: WebFetchTool#
  FileReadTool: FileReadTool#
  UnitConverterTool: UnitConverterTool#
  NotepadTool: NotepadTool#
  NotepadTool._content: NotepadTool#_content.
  function_to_json: function_to_json().
  WikipediaTool.call: WikipediaTool#call().
  WebFetchTool.call: WebFetchTool#call().
  FileReadTool.call: FileReadTool#call().
  FunctionTool.func: FunctionTool#func.
  _TYPE_MAP._TYPE_MAP: _TYPE_MAP._TYPE_MAP.
  list_tools: list_tools().
  Tool.chat_schema: Tool#chat_schema().
  PythonCodeTool._timeout_handler: PythonCodeTool#_timeout_handler().
  FunctionTool: FunctionTool#
  FunctionTool.name: FunctionTool#name().
  UnitConverterTool._CONVERSIONS: UnitConverterTool#_CONVERSIONS.
  JSONProcessorTool.call: JSONProcessorTool#call().
  UnitConverterTool.call: UnitConverterTool#call().
  NotepadTool.call: NotepadTool#call().
  FunctionTool.call: FunctionTool#call().
  FunctionTool.execute: FunctionTool#execute().
  PythonCodeTool._timeout: PythonCodeTool#_timeout.
  PythonCodeTool._max_output_length: PythonCodeTool#_max_output_length.
  BashTool._timeout: BashTool#_timeout.
  BashTool._max_output_length: BashTool#_max_output_length.
  WikipediaTool._max_length: WikipediaTool#_max_length.
  WebFetchTool._max_length: WebFetchTool#_max_length.
  FileReadTool._allowed_dirs: FileReadTool#_allowed_dirs.
  FileReadTool._max_length: FileReadTool#_max_length.
  FunctionTool.tool_schema: FunctionTool#tool_schema.
  BashTool._shell: BashTool#_shell.
  JSONProcessorTool._query: JSONProcessorTool#_query().
  WikipediaTool._language: WikipediaTool#_language.
  WebFetchTool._timeout: WebFetchTool#_timeout.
  FunctionTool.tool_name: FunctionTool#tool_name.
  PythonCodeTool.__init__: PythonCodeTool#__init__().
  CalculatorTool.call: CalculatorTool#call().
  BashTool.__init__: BashTool#__init__().
  RegexTool.call: RegexTool#call().
  WikipediaTool.__init__: WikipediaTool#__init__().
  WebFetchTool.__init__: WebFetchTool#__init__().
  FileReadTool.__init__: FileReadTool#__init__().
  NotepadTool.__init__: NotepadTool#__init__().
---
# Module: [`easydel/trainers/agentic_moshpit/tools.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py)

## Classes
### `BashTool`  ·  implements/extends ABC
- def: [`easydel/trainers/agentic_moshpit/tools.py:486`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L486)
- doc: Shell command execution.
- signature: `class BashTool(Tool):`
- members:
  - `call(self, command: str)` — [`L508`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L508) — Run a bash shell command and return its stdout and stderr output. Supports pipes, redirects, environment variables, and all standard Unix utilities (grep, awk, sed, curl, find, etc.).
- protocol/private: `__init__`[`L498`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L498), `_max_output_length`[`L505`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L505), `_shell`[`L506`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L506), `_timeout`[`L504`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L504)
- uses (calls/refs, reference-scoped): [`Tool`](tools.md#Tool), [`register_tool`](tools.md#register_tool)
- used by: [`call`](tools.md#Tool.call)

### `CalculatorTool`  ·  implements/extends ABC
- def: [`easydel/trainers/agentic_moshpit/tools.py:415`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L415)
- doc: Mathematical expression evaluator with full math support.
- signature: `class CalculatorTool(Tool):`
- members:
  - `call(self, expression: str)` — [`L423`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L423) — Evaluate a mathematical expression and return the numeric result. Supports all standard arithmetic (+, -, *, /, //, %, **), comparisons, and math functions: abs, round, min, max, pow, sqrt, sin, cos, tan, asin, acos, atan, atan2, log, log2, log10, exp, ceil, floor, factorial, gcd, lcm, comb, perm, degrees, radians, hypot, isqrt. Constants: pi, e, tau, inf, nan.
- uses (calls/refs, reference-scoped): [`Tool`](tools.md#Tool), [`register_tool`](tools.md#register_tool)
- used by: [`call`](tools.md#Tool.call)

### `FileReadTool`  ·  implements/extends ABC
- def: [`easydel/trainers/agentic_moshpit/tools.py:676`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L676)
- doc: Read local file contents.
- signature: `class FileReadTool(Tool):`
- members:
  - `call(self, path: str)` — [`L693`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L693) — Read and return the contents of a local file. Supports any text file format. Tilde (~) is expanded to the home directory.
- protocol/private: `__init__`[`L685`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L685), `_allowed_dirs`[`L690`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L690), `_max_length`[`L691`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L691)
- uses (calls/refs, reference-scoped): [`Tool`](tools.md#Tool), [`register_tool`](tools.md#register_tool)
- used by: [`call`](tools.md#Tool.call)

### `FunctionTool`  ·  implements/extends Tool
- def: [`easydel/trainers/agentic_moshpit/tools.py:806`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L806)
- doc: Wrap a plain Python function as a Tool.
- signature: `class FunctionTool(Tool):`
- members:
  - `call(self, **kwargs: tp.Any)` — [`L829`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L829) — Delegate to the wrapped function.
  - `execute(self, arguments: str)` — [`L833`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L833) — Parse JSON and call the wrapped function.
  - `name(self)` — [`L826`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L826)
  - `schema(self)` — [`L844`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L844) — Return schema from explicit override or auto-extract from ``func``.
  - `func` — [`L821`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L821)
  - `tool_name` — [`L822`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L822)
  - `tool_schema` — [`L823`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L823)
- uses (calls/refs, reference-scoped): [`Tool`](tools.md#Tool), [`function_to_json`](tools.md#function_to_json)
- used by: [`call`](tools.md#Tool.call), [`Tool`](tools.md#Tool), [`schema`](tools.md#Tool.schema), [`name`](tools.md#Tool.name), [`execute`](tools.md#Tool.execute)

### `JSONProcessorTool`  ·  implements/extends ABC
- def: [`easydel/trainers/agentic_moshpit/tools.py:557`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L557)
- doc: JSON parsing, querying, and inspection.
- signature: `class JSONProcessorTool(Tool):`
- members:
  - `call(self, operation: str, data: str, path: str = "")` — [`L560`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L560) — Process JSON data. Three operations are available: 'parse' validates and pretty-prints the JSON; 'query' extracts a nested value by dot-separated key path (use numeric indices for arrays, e.g. 'results.0.name'); 'keys' lists all top-level keys of a JSON object.
- protocol/private: `_query`[`L584`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L584)
- uses (calls/refs, reference-scoped): [`Tool`](tools.md#Tool), [`register_tool`](tools.md#register_tool)
- used by: [`call`](tools.md#Tool.call)

### `NotepadTool`  ·  implements/extends ABC
- def: [`easydel/trainers/agentic_moshpit/tools.py:778`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L778)
- doc: Persistent scratch-pad that survives across turns within an episode.
- signature: `class NotepadTool(Tool):`
- members:
  - `call(self, action: str, text: str = "")` — [`L784`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L784) — A persistent notepad for storing intermediate results, plans, or reasoning across multiple turns. Use 'write' to replace all content, 'append' to add to existing content, 'read' to retrieve current content, or 'clear' to erase everything. The notepad persists for the entire episode.
- protocol/private: `__init__`[`L781`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L781), `_content`[`L782`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L782)
- uses (calls/refs, reference-scoped): [`Tool`](tools.md#Tool), [`register_tool`](tools.md#register_tool)
- used by: [`call`](tools.md#Tool.call)

### `PythonCodeTool`  ·  implements/extends ABC
- def: [`easydel/trainers/agentic_moshpit/tools.py:365`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L365)
- doc: Sandboxed Python code execution.
- signature: `class PythonCodeTool(Tool):`
- members:
  - `call(self, code: str)` — [`L381`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L381) — Execute a Python code snippet and return everything printed to stdout/stderr. Use print() to produce output. You have access to the full Python standard library including math, json, re, itertools, collections, fractions, decimal, statistics, etc. For example: 'import math; print(math.factorial(10))' or 'print(sum(range(1, 101)))'.
- protocol/private: `__init__`[`L377`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L377), `_max_output_length`[`L379`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L379), `_timeout`[`L378`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L378), `_timeout_handler`[`L393`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L393)
- uses (calls/refs, reference-scoped): [`Tool`](tools.md#Tool), [`register_tool`](tools.md#register_tool)
- used by: [`call`](tools.md#Tool.call)

### `RegexTool`  ·  implements/extends ABC
- def: [`easydel/trainers/agentic_moshpit/tools.py:538`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L538)
- doc: Regular expression search and extraction.
- signature: `class RegexTool(Tool):`
- members:
  - `call(self, pattern: str, text: str)` — [`L541`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L541) — Search text with a Python regular expression and return all matches as a JSON array. Uses re.findall(), so if the pattern contains capture groups, only the captured portions are returned. Supports the full Python regex syntax including lookaheads, named groups, etc.
- uses (calls/refs, reference-scoped): [`Tool`](tools.md#Tool), [`register_tool`](tools.md#register_tool)
- used by: [`call`](tools.md#Tool.call)

### `Tool`
- def: [`easydel/trainers/agentic_moshpit/tools.py:253`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L253)
- members:
  - `call(self, **kwargs: tp.Any)` — [`L293`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L293) — The tool's callable entry point.
  - `chat_schema(self)` — [`L352`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L352) — Return the bare function dict for ``apply_chat_template``.
  - `execute(self, arguments: str)` — [`L314`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L314) — Parse JSON arguments and delegate to ``call``.
  - `name(self)` — [`L303`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L303) — Tool name, derived from the registry or class name.
  - `schema(self)` — [`L332`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L332) — Return the full OpenAI function-calling schema, auto-extracted from ``call``.
- uses (calls/refs, reference-scoped): [`schema`](tools.md#FunctionTool.schema), [`_TOOL_REGISTRY`](tools.md#_TOOL_REGISTRY._TOOL_REGISTRY), [`call`](tools.md#BashTool.call), [`call`](tools.md#PythonCodeTool.call), [`function_to_json`](tools.md#function_to_json), [`call`](tools.md#FileReadTool.call), [`call`](tools.md#WebFetchTool.call), [`call`](tools.md#WikipediaTool.call), [`FunctionTool`](tools.md#FunctionTool), [`name`](tools.md#FunctionTool.name), [`call`](tools.md#FunctionTool.call), [`call`](tools.md#JSONProcessorTool.call), [`call`](tools.md#NotepadTool.call), [`call`](tools.md#UnitConverterTool.call), [`execute`](tools.md#FunctionTool.execute), [`call`](tools.md#CalculatorTool.call), [`call`](tools.md#RegexTool.call)
- used by: [`_execute_tool_calls`](environment.md#ToolEnvWrapper._execute_tool_calls), [`__init__`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer.__init__), [`_tools`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._tools), [`register_tool`](tools.md#register_tool), [`_TOOL_REGISTRY`](tools.md#_TOOL_REGISTRY._TOOL_REGISTRY), [`make_tool`](tools.md#make_tool), [`BashTool`](tools.md#BashTool), [`CalculatorTool`](tools.md#CalculatorTool), [`FileReadTool`](tools.md#FileReadTool), [`JSONProcessorTool`](tools.md#JSONProcessorTool), [`NotepadTool`](tools.md#NotepadTool), [`PythonCodeTool`](tools.md#PythonCodeTool), [`RegexTool`](tools.md#RegexTool), [`UnitConverterTool`](tools.md#UnitConverterTool), [`WebFetchTool`](tools.md#WebFetchTool), [`WikipediaTool`](tools.md#WikipediaTool), [`decorator`](tools.md#register_tool.decorator), [`tools`](environment.md#ToolEnvWrapper.tools), [`__init__`](environment.md#ToolEnvWrapper.__init__), [`FunctionTool`](tools.md#FunctionTool)

### `UnitConverterTool`  ·  implements/extends ABC
- def: [`easydel/trainers/agentic_moshpit/tools.py:718`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L718)
- doc: Physical unit converter.
- signature: `class UnitConverterTool(Tool):`
- members:
  - `call(self, value: float, from_unit: str, to_unit: str)` — [`L758`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L758) — Convert a numeric value from one unit to another. Supported units: length (km, mi, m, ft, cm, in), weight (kg, lb, g, oz), temperature (c, f, k), volume (l, gal, ml). Unit names are case-insensitive.
- protocol/private: `_CONVERSIONS`[`L725`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L725)
- uses (calls/refs, reference-scoped): [`Tool`](tools.md#Tool), [`register_tool`](tools.md#register_tool)
- used by: [`call`](tools.md#Tool.call)

### `WebFetchTool`  ·  implements/extends ABC
- def: [`easydel/trainers/agentic_moshpit/tools.py:637`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L637)
- doc: Fetch web page content via HTTP GET.
- signature: `class WebFetchTool(Tool):`
- members:
  - `call(self, url: str)` — [`L649`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L649) — Fetch a URL via HTTP GET and return the page content as plain text with HTML tags stripped. Useful for reading web pages, API endpoints that return HTML, documentation pages, etc.
- protocol/private: `__init__`[`L645`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L645), `_max_length`[`L647`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L647), `_timeout`[`L646`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L646)
- uses (calls/refs, reference-scoped): [`Tool`](tools.md#Tool), [`register_tool`](tools.md#register_tool)
- used by: [`call`](tools.md#Tool.call)

### `WikipediaTool`  ·  implements/extends ABC
- def: [`easydel/trainers/agentic_moshpit/tools.py:601`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L601)
- doc: Wikipedia article lookup.
- signature: `class WikipediaTool(Tool):`
- members:
  - `call(self, query: str)` — [`L613`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L613) — Look up a topic on Wikipedia and return a summary of the article. Returns the article title and the first few paragraphs (the extract). Useful for factual questions about people, places, events, science, history, etc.
- protocol/private: `__init__`[`L609`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L609), `_language`[`L610`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L610), `_max_length`[`L611`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L611)
- uses (calls/refs, reference-scoped): [`Tool`](tools.md#Tool), [`register_tool`](tools.md#register_tool)
- used by: [`call`](tools.md#Tool.call)

## Functions
- `decorator(cls: type[Tool])` — [`L222`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L222)
- `function_to_json(func: tp.Callable)` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L81) — Convert a Python function into an OpenAI function-calling schema.
- `list_tools()` — [`L248`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L248) — Return names of all registered tools.
- `make_tool(name: str, **kwargs: tp.Any)` — [`L229`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L229) — Create a tool instance by registered name.
- `register_tool(name: str)` — [`L203`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L203) — Register a tool class in the global registry.

## Module values
- `_TOOL_REGISTRY` — [`L65`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L65)
- `_TYPE_MAP` — [`L67`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/tools.py#L67)

