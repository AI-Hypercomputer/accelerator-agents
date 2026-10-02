---
title: 'Module: simply/tool_lib.py'
type: catalog
provenance: extracted
module: simply/tool_lib.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.tool_lib`/
symbols:
  ToolExecutor.sample_with_tool: ToolExecutor#sample_with_tool().
  ToolExecutor._encode: ToolExecutor#_encode().
  CalculatorToolExecutor.make_single_tool_call: CalculatorToolExecutor#make_single_tool_call().
  CalculatorToolExecutor.extract_tool_inputs_from_sample: CalculatorToolExecutor#extract_tool_inputs_from_sample().
  CalculatorToolExecutor: CalculatorToolExecutor#
  ToolExecutor.execute_tool_batch_with_sharding: ToolExecutor#execute_tool_batch_with_sharding().
  CalculatorToolExecutor.parse_tool_output: CalculatorToolExecutor#parse_tool_output().
  create_tool_executor: create_tool_executor().
  CalculatorToolExecutor.make_tool_calls: CalculatorToolExecutor#make_tool_calls().
  CalculatorToolExecutor.execute_tool_from_sample: CalculatorToolExecutor#execute_tool_from_sample().
  ToolExecutor.make_tool_calls: ToolExecutor#make_tool_calls().
  ToolInput: ToolInput#
  ToolOutput: ToolOutput#
  ToolOutput.metadata: ToolOutput#metadata.
  SamplingOutputToolUse: SamplingOutputToolUse#
  ToolExecutor.extract_tool_inputs_from_sample: ToolExecutor#extract_tool_inputs_from_sample().
  ToolExecutor.parse_tool_output: ToolExecutor#parse_tool_output().
  ChunkSequence: ChunkSequence.
  ToolRegistry: ToolRegistry#
  ToolExecutor.execute_tool_batch: ToolExecutor#execute_tool_batch().
  SamplingParams: SamplingParams.
  SamplingOutput: SamplingOutput.
  PyTree: PyTree.
  get_answer_mask: get_answer_mask().
  RawT: RawT.
  SamplingInput: SamplingInput.
  ToolExecutor.execute_tool_from_sample: ToolExecutor#execute_tool_from_sample().
  CalculatorToolExecutor.execute_tool_batch: CalculatorToolExecutor#execute_tool_batch().
  ToolInput.input_text: ToolInput#input_text.
  ToolInput.tool_name: ToolInput#tool_name.
  ToolOutput.output_text: ToolOutput#output_text.
  ToolExecutor: ToolExecutor#
  _find_sublist_index: _find_sublist_index().
  PRNGKey: PRNGKey.
  ToolOutput.tool_name: ToolOutput#tool_name.
  ToolExecutor.result_start_marker: ToolExecutor#result_start_marker.
  ToolExecutor.result_end_marker: ToolExecutor#result_end_marker.
  CalculatorToolExecutor.tool_start_marker: CalculatorToolExecutor#tool_start_marker.
  CalculatorToolExecutor.tool_end_marker: CalculatorToolExecutor#tool_end_marker.
  CalculatorToolExecutor.result_start_marker: CalculatorToolExecutor#result_start_marker.
  CalculatorToolExecutor.result_end_marker: CalculatorToolExecutor#result_end_marker.
  repeat_elements: repeat_elements().
  ToolRegistry.namespace: ToolRegistry#namespace.
  SamplingOutputToolUse.answer_mask: SamplingOutputToolUse#answer_mask.
  SamplingOutputToolUse.turns: SamplingOutputToolUse#turns.
  SamplingOutputToolUse.is_throttled: SamplingOutputToolUse#is_throttled().
  ToolExecutor.tool_start_marker: ToolExecutor#tool_start_marker.
  ToolExecutor.tool_end_marker: ToolExecutor#tool_end_marker.
---
# Module: [`simply/tool_lib.py`](../../../../../raw/code/simply/simply/tool_lib.py)

## Classes
### `CalculatorToolExecutor`
- def: [`simply/tool_lib.py:391`](../../../../../raw/code/simply/simply/tool_lib.py#L391)
- doc: A simple calculator tool executor for demonstration purposes.
- signature: `class CalculatorToolExecutor(ToolExecutor):`
- members:
  - `execute_tool_batch(self, sample_texts: Sequence[str])` — [`L475`](../../../../../raw/code/simply/simply/tool_lib.py#L475) — Executes tool batch.
  - `execute_tool_from_sample(self, sample_text: str)` — [`L467`](../../../../../raw/code/simply/simply/tool_lib.py#L467) — Executes the calculator tool on a sample text.
  - `extract_tool_inputs_from_sample(self, sample_text: str)` — [`L403`](../../../../../raw/code/simply/simply/tool_lib.py#L403) — Extracts arithmetic expressions from the sample text.
  - `make_single_tool_call(self, tool_input: ToolInput)` — [`L418`](../../../../../raw/code/simply/simply/tool_lib.py#L418) — Evaluates a single arithmetic expression.
  - `make_tool_calls(self, tool_inputs: list[ToolInput])` — [`L454`](../../../../../raw/code/simply/simply/tool_lib.py#L454) — Evaluates a batch of arithmetic expressions.
  - `parse_tool_output(self, tool_output: ToolOutput)` — [`L460`](../../../../../raw/code/simply/simply/tool_lib.py#L460) — Formats the tool output.
  - `result_end_marker` — [`L401`](../../../../../raw/code/simply/simply/tool_lib.py#L401)
  - `result_start_marker` — [`L400`](../../../../../raw/code/simply/simply/tool_lib.py#L400)
  - `tool_end_marker` — [`L399`](../../../../../raw/code/simply/simply/tool_lib.py#L399)
  - `tool_start_marker` — [`L398`](../../../../../raw/code/simply/simply/tool_lib.py#L398)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`ToolInput`](tool_lib.md#ToolInput), [`ToolOutput`](tool_lib.md#ToolOutput), [`metadata`](tool_lib.md#ToolOutput.metadata), [`ToolRegistry`](tool_lib.md#ToolRegistry), [`ToolExecutor`](tool_lib.md#ToolExecutor), [`input_text`](tool_lib.md#ToolInput.input_text), [`output_text`](tool_lib.md#ToolOutput.output_text), [`tool_name`](tool_lib.md#ToolInput.tool_name), [`tool_name`](tool_lib.md#ToolOutput.tool_name)
- used by: [`make_tool_calls`](tool_lib.md#ToolExecutor.make_tool_calls), [`extract_tool_inputs_from_sample`](tool_lib.md#ToolExecutor.extract_tool_inputs_from_sample), [`parse_tool_output`](tool_lib.md#ToolExecutor.parse_tool_output), [`execute_tool_batch`](tool_lib.md#ToolExecutor.execute_tool_batch), [`execute_tool_from_sample`](tool_lib.md#ToolExecutor.execute_tool_from_sample), [`test_calculator_tool_executor`](tool_lib_test.md#ToolLibTest.test_calculator_tool_executor), [`test_calculator_tool_executor_batch`](tool_lib_test.md#ToolLibTest.test_calculator_tool_executor_batch), [`test_calculator_tool_executor_error`](tool_lib_test.md#ToolLibTest.test_calculator_tool_executor_error), [`test_calculator_tool_executor_safe_eval`](tool_lib_test.md#ToolLibTest.test_calculator_tool_executor_safe_eval)

### `SamplingOutputToolUse`
- def: [`simply/tool_lib.py:75`](../../../../../raw/code/simply/simply/tool_lib.py#L75)
- signature: `class SamplingOutputToolUse(SamplingOutput):`
- members:
  - `is_throttled(self)` — [`L80`](../../../../../raw/code/simply/simply/tool_lib.py#L80)
  - `answer_mask` — [`L76`](../../../../../raw/code/simply/simply/tool_lib.py#L76)
  - `turns` — [`L77`](../../../../../raw/code/simply/simply/tool_lib.py#L77)
- uses (calls/refs, reference-scoped): [`SamplingOutput`](tool_lib.md#SamplingOutput)
- used by: [`sample_with_tool`](tool_lib.md#ToolExecutor.sample_with_tool)

### `ToolExecutor`
- def: [`simply/tool_lib.py:85`](../../../../../raw/code/simply/simply/tool_lib.py#L85)
- doc: Tool executor base class.
- signature: `class ToolExecutor:`
- members:
  - `execute_tool_batch(self, sample_texts: Sequence[str])` — [`L111`](../../../../../raw/code/simply/simply/tool_lib.py#L111) — Execute tool batch.
  - `execute_tool_batch_with_sharding(self, tool_inputs_str: list[str], lm_interface: model_lib.LMInterface, tool_sync_length: int = 2048)` — [`L115`](../../../../../raw/code/simply/simply/tool_lib.py#L115) — Executes tool batch with sharding.
  - `execute_tool_from_sample(self, sample_text: str)` — [`L107`](../../../../../raw/code/simply/simply/tool_lib.py#L107) — Execute tool from text.
  - `extract_tool_inputs_from_sample(self, sample_text: str)` — [`L93`](../../../../../raw/code/simply/simply/tool_lib.py#L93) — Extract tool inputs from sample_text.
  - `make_tool_calls(self, tool_inputs: list[ToolInput])` — [`L99`](../../../../../raw/code/simply/simply/tool_lib.py#L99) — Make tool calls.
  - `parse_tool_output(self, tool_output: ToolOutput)` — [`L103`](../../../../../raw/code/simply/simply/tool_lib.py#L103) — Parse tool output to text.
  - `sample_with_tool(self, lm_interface: model_lib.LMInterface, lm_format: lm_format_lib.LMFormat, input_text: Sequence[sampling_lib.SamplingInput], sampling_params: SamplingParams | None = None, prng_key: int | PRNGKey | None = None, params: PyTree = None, prefill_size: int = -1, max_turns: int = 1, max_tool_response_len: int = 1024)` — [`L159`](../../../../../raw/code/simply/simply/tool_lib.py#L159) — Sample generations with tool use.
  - `result_end_marker` — [`L91`](../../../../../raw/code/simply/simply/tool_lib.py#L91)
  - `result_start_marker` — [`L90`](../../../../../raw/code/simply/simply/tool_lib.py#L90)
  - `tool_end_marker` — [`L89`](../../../../../raw/code/simply/simply/tool_lib.py#L89)
  - `tool_start_marker` — [`L88`](../../../../../raw/code/simply/simply/tool_lib.py#L88)
- protocol/private: `_encode`[`L229`](../../../../../raw/code/simply/simply/tool_lib.py#L229)
- uses (calls/refs, reference-scoped): [`generate`](model_lib.md#LMInterface.generate), [`input_processor`](model_lib.md#LMInterface.input_processor), [`input_as_chunks`](utils/sampling_lib.md#input_as_chunks), [`LMInterface`](model_lib.md#LMInterface), [`SamplingInput`](utils/sampling_lib.md#SamplingInput), [`encode`](utils/sampling_lib.md#InputProcessorInterface.encode), [`chunks_as_text`](utils/sampling_lib.md#chunks_as_text), [`extract_tool_inputs_from_sample`](tool_lib.md#CalculatorToolExecutor.extract_tool_inputs_from_sample), [`max_seq_len`](utils/sampling_lib.md#SamplingParams.max_seq_len), [`eos_ids`](utils/sampling_lib.md#InputProcessorInterface.eos_ids), [`parse_tool_output`](tool_lib.md#CalculatorToolExecutor.parse_tool_output), [`num_samples`](utils/sampling_lib.md#SamplingParams.num_samples), [`LMFormat`](utils/lm_format.md#LMFormat), [`decode`](utils/sampling_lib.md#InputProcessorInterface.decode), [`multihost_sharded`](utils/sharding.md#multihost_sharded), [`execute_tool_from_sample`](tool_lib.md#CalculatorToolExecutor.execute_tool_from_sample), [`make_tool_calls`](tool_lib.md#CalculatorToolExecutor.make_tool_calls), [`default_sampling_params`](model_lib.md#LMInterface.default_sampling_params), [`ToolInput`](tool_lib.md#ToolInput), [`ToolOutput`](tool_lib.md#ToolOutput), [`SamplingOutputToolUse`](tool_lib.md#SamplingOutputToolUse), [`ChunkSequence`](tool_lib.md#ChunkSequence), [`PyTree`](tool_lib.md#PyTree), [`SamplingParams`](tool_lib.md#SamplingParams), [`get_answer_mask`](tool_lib.md#get_answer_mask), [`execute_tool_batch`](tool_lib.md#CalculatorToolExecutor.execute_tool_batch), [`PRNGKey`](tool_lib.md#PRNGKey), [`repeat_elements`](tool_lib.md#repeat_elements)
- used by: [`run_experiment`](rl_lib.md#run_experiment), [`CalculatorToolExecutor`](tool_lib.md#CalculatorToolExecutor), [`create_tool_executor`](tool_lib.md#create_tool_executor)

### `ToolInput`
- def: [`simply/tool_lib.py:57`](../../../../../raw/code/simply/simply/tool_lib.py#L57)
- doc: Tool input.
- signature: `class ToolInput:`
- members:
  - `input_text` — [`L60`](../../../../../raw/code/simply/simply/tool_lib.py#L60)
  - `tool_name` — [`L61`](../../../../../raw/code/simply/simply/tool_lib.py#L61)
- used by: [`make_single_tool_call`](tool_lib.md#CalculatorToolExecutor.make_single_tool_call), [`extract_tool_inputs_from_sample`](tool_lib.md#CalculatorToolExecutor.extract_tool_inputs_from_sample), [`make_tool_calls`](tool_lib.md#CalculatorToolExecutor.make_tool_calls), [`make_tool_calls`](tool_lib.md#ToolExecutor.make_tool_calls), [`extract_tool_inputs_from_sample`](tool_lib.md#ToolExecutor.extract_tool_inputs_from_sample)

### `ToolOutput`
- def: [`simply/tool_lib.py:65`](../../../../../raw/code/simply/simply/tool_lib.py#L65)
- doc: Tool output.
- signature: `class ToolOutput:`
- members:
  - `metadata` — [`L70`](../../../../../raw/code/simply/simply/tool_lib.py#L70)
  - `output_text` — [`L68`](../../../../../raw/code/simply/simply/tool_lib.py#L68)
  - `tool_name` — [`L69`](../../../../../raw/code/simply/simply/tool_lib.py#L69)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`SamplingRegistry`](utils/sampling_lib.md#SamplingRegistry)
- used by: [`make_single_tool_call`](tool_lib.md#CalculatorToolExecutor.make_single_tool_call), [`parse_tool_output`](tool_lib.md#CalculatorToolExecutor.parse_tool_output), [`make_tool_calls`](tool_lib.md#CalculatorToolExecutor.make_tool_calls), [`make_tool_calls`](tool_lib.md#ToolExecutor.make_tool_calls), [`parse_tool_output`](tool_lib.md#ToolExecutor.parse_tool_output)

### `ToolRegistry`  ·  implements/extends RootRegistry
- def: [`simply/tool_lib.py:50`](../../../../../raw/code/simply/simply/tool_lib.py#L50)
- doc: Tool registry.
- signature: `class ToolRegistry(registry.RootRegistry):`
- members:
  - `namespace` — [`L53`](../../../../../raw/code/simply/simply/tool_lib.py#L53)
- uses (calls/refs, reference-scoped): [`RootRegistry`](utils/registry.md#RootRegistry)
- used by: [`RootRegistry`](utils/registry.md#RootRegistry), [`CalculatorToolExecutor`](tool_lib.md#CalculatorToolExecutor), [`create_tool_executor`](tool_lib.md#create_tool_executor)

## Functions
- `_find_sublist_index(main_list: list[int], sub_list: list[int], start_index: int = 0)` — [`L484`](../../../../../raw/code/simply/simply/tool_lib.py#L484) — Finds the starting index of the first sub_list after start_index.
- `create_tool_executor(config: config_lib.ExperimentConfig)` — [`L543`](../../../../../raw/code/simply/simply/tool_lib.py#L543)
- `get_answer_mask(tokens: list[int], start_tokens: list[int], end_tokens: list[int])` — [`L502`](../../../../../raw/code/simply/simply/tool_lib.py#L502) — Generates a token mask for multiple non-overlapping turns.
- `repeat_elements(input_list, n)` — [`L534`](../../../../../raw/code/simply/simply/tool_lib.py#L534) — [1, 2, 3], n = 2 -> [1, 1, 2, 2, 3, 3].

## Module values
- `ChunkSequence` — [`L42`](../../../../../raw/code/simply/simply/tool_lib.py#L42)
- `PRNGKey` — [`L37`](../../../../../raw/code/simply/simply/tool_lib.py#L37)
- `PyTree` — [`L41`](../../../../../raw/code/simply/simply/tool_lib.py#L41)
- `RawT` — [`L40`](../../../../../raw/code/simply/simply/tool_lib.py#L40)
- `SamplingInput` — [`L43`](../../../../../raw/code/simply/simply/tool_lib.py#L43)
- `SamplingOutput` — [`L39`](../../../../../raw/code/simply/simply/tool_lib.py#L39)
- `SamplingParams` — [`L38`](../../../../../raw/code/simply/simply/tool_lib.py#L38)

