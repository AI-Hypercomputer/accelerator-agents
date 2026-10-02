---
title: 'Module: easydel/trainers/agentic_moshpit/environment.py'
type: catalog
provenance: extracted
module: easydel/trainers/agentic_moshpit/environment.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.agentic_moshpit.environment`/
symbols:
  ToolEnvWrapper._execute_tool_calls: ToolEnvWrapper#_execute_tool_calls().
  AgenticEnvironment: AgenticEnvironment#
  ToolEnvWrapper.step_with_tool_calls: ToolEnvWrapper#step_with_tool_calls().
  StepResult: StepResult#
  ResetResult: ResetResult#
  create_tool_call_parser: create_tool_call_parser().
  StepResult.observation: StepResult#observation.
  ToolEnvWrapper.reset: ToolEnvWrapper#reset().
  AgenticEnvironment.reset: AgenticEnvironment#reset().
  AgenticEnvironment.step: AgenticEnvironment#step().
  ToolEnvWrapper: ToolEnvWrapper#
  StepResult.reward: StepResult#reward.
  _make_inference_tool_parser: _make_inference_tool_parser().
  StepResult.terminated: StepResult#terminated.
  StepResult.info: StepResult#info.
  ToolEnvWrapper._tool_calls_this_step: ToolEnvWrapper#_tool_calls_this_step.
  AgenticEnvironment.max_steps: AgenticEnvironment#max_steps().
  ToolEnvWrapper.tools: ToolEnvWrapper#tools.
  ToolEnvWrapper.env: ToolEnvWrapper#env.
  ResetResult.observation: ResetResult#observation.
  ResetResult.info: ResetResult#info.
  ToolEnvWrapper.__init__: ToolEnvWrapper#__init__().
  ToolEnvWrapper.step: ToolEnvWrapper#step().
  ToolEnvWrapper.close: ToolEnvWrapper#close().
  ToolEnvWrapper.max_steps: ToolEnvWrapper#max_steps().
  AgenticEnvironment.close: AgenticEnvironment#close().
  _make_regex_parser: _make_regex_parser().
  ToolEnvWrapper.tool_call_parser: ToolEnvWrapper#tool_call_parser.
  ToolEnvWrapper._normalize_structured_tool_calls: ToolEnvWrapper#_normalize_structured_tool_calls().
  StepResult.truncated: StepResult#truncated.
  _default_tool_call_parser: _default_tool_call_parser().
  ToolEnvWrapper.max_tool_calls_per_step: ToolEnvWrapper#max_tool_calls_per_step.
  _make_regex_parser.parser: _make_regex_parser().parser().
  _make_inference_tool_parser.parser: _make_inference_tool_parser().parser().
  ToolEnvWrapper._coerce_tool_args_json: ToolEnvWrapper#_coerce_tool_args_json().
---
# Module: [`easydel/trainers/agentic_moshpit/environment.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py)

## Classes
### `AgenticEnvironment`  ·  implements/extends ABC
- def: [`easydel/trainers/agentic_moshpit/environment.py:73`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L73)
- doc: Abstract base class for agentic training environments.
- signature: `class AgenticEnvironment(abc.ABC):`
- members:
  - `close(self)` — [`L121`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L121) — Clean up any resources held by the environment.
  - `max_steps(self)` — [`L125`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L125) — Optional maximum number of steps per episode.
  - `reset(self, seed: int | None = None)` — [`L99`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L99) — Reset the environment and return an initial observation.
  - `step(self, action: str)` — [`L111`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L111) — Process the agent's action and return the next state.
- uses (calls/refs, reference-scoped): [`step`](self_play.md#SelfPlayEnvironment.step), [`reset`](self_play.md#SelfPlayEnvironment.reset), [`StepResult`](environment.md#StepResult), [`ResetResult`](environment.md#ResetResult), [`reset`](environment.md#ToolEnvWrapper.reset), [`ToolEnvWrapper`](environment.md#ToolEnvWrapper), [`SelfPlayEnvironment`](self_play.md#SelfPlayEnvironment), [`close`](environment.md#ToolEnvWrapper.close), [`max_steps`](environment.md#ToolEnvWrapper.max_steps), [`step`](environment.md#ToolEnvWrapper.step), [`max_steps`](self_play.md#SelfPlayEnvironment.max_steps)
- used by: [`run_grouped_episodes`](env_manager.md#RolloutManager.run_grouped_episodes), [`__init__`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer.__init__), [`step_with_tool_calls`](environment.md#ToolEnvWrapper.step_with_tool_calls), [`_wrap_env_with_tools`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._wrap_env_with_tools), [`reset`](environment.md#ToolEnvWrapper.reset), [`ToolEnvWrapper`](environment.md#ToolEnvWrapper), [`SelfPlayEnvironment`](self_play.md#SelfPlayEnvironment), [`_unwrap`](env_manager.md#RolloutManager._unwrap), [`env`](env_manager.md#_EpisodeState.env), [`__init__`](environment.md#ToolEnvWrapper.__init__), [`close`](environment.md#ToolEnvWrapper.close), [`max_steps`](environment.md#ToolEnvWrapper.max_steps)

### `ResetResult`
- def: [`easydel/trainers/agentic_moshpit/environment.py:61`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L61)
- doc: Result from an environment reset.
- signature: `class ResetResult:`
- members:
  - `info` — [`L70`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L70)
  - `observation` — [`L69`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L69)
- used by: [`run_grouped_episodes`](env_manager.md#RolloutManager.run_grouped_episodes), [`reset`](self_play.md#SelfPlayEnvironment.reset), [`reset_with_question`](self_play.md#SelfPlayEnvironment.reset_with_question), [`reset`](environment.md#ToolEnvWrapper.reset), [`reset`](environment.md#AgenticEnvironment.reset)

### `StepResult`
- def: [`easydel/trainers/agentic_moshpit/environment.py:42`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L42)
- doc: Result from a single environment step.
- signature: `class StepResult:`
- members:
  - `info` — [`L57`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L57)
  - `observation` — [`L53`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L53)
  - `reward` — [`L54`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L54)
  - `terminated` — [`L55`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L55)
  - `truncated` — [`L56`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L56)
- used by: [`run_grouped_episodes`](env_manager.md#RolloutManager.run_grouped_episodes), [`step`](self_play.md#SelfPlayEnvironment.step), [`_execute_tool_calls`](environment.md#ToolEnvWrapper._execute_tool_calls), [`step_with_tool_calls`](environment.md#ToolEnvWrapper.step_with_tool_calls), [`step`](environment.md#AgenticEnvironment.step), [`step`](environment.md#ToolEnvWrapper.step)

### `ToolEnvWrapper`  ·  implements/extends AgenticEnvironment
- def: [`easydel/trainers/agentic_moshpit/environment.py:236`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L236)
- doc: Wraps an environment with tool-calling capabilities.
- signature: `class ToolEnvWrapper(AgenticEnvironment):`
- members:
  - `close(self)` — [`L377`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L377)
  - `max_steps(self)` — [`L381`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L381)
  - `reset(self, seed: int | None = None)` — [`L277`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L277)
  - `step(self, action: str)` — [`L374`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L374)
  - `step_with_tool_calls(self, action: str, *, tool_calls: tp.Any | None = None)` — [`L355`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L355)
  - `env` — [`L271`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L271)
  - `max_tool_calls_per_step` — [`L274`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L274)
  - `tool_call_parser` — [`L273`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L273)
  - `tools` — [`L272`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L272)
- protocol/private: `__init__`[`L264`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L264), `_coerce_tool_args_json`[`L282`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L282), `_execute_tool_calls`[`L332`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L332), `_normalize_structured_tool_calls`[`L292`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L292), `_tool_calls_this_step`[`L275`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L275)
- uses (calls/refs, reference-scoped): [`Tool`](tools.md#Tool), [`AgenticEnvironment`](environment.md#AgenticEnvironment), [`StepResult`](environment.md#StepResult), [`ResetResult`](environment.md#ResetResult), [`observation`](environment.md#StepResult.observation), [`reset`](environment.md#AgenticEnvironment.reset), [`step`](environment.md#AgenticEnvironment.step), [`reward`](environment.md#StepResult.reward), [`info`](environment.md#StepResult.info), [`name`](tools.md#Tool.name), [`terminated`](environment.md#StepResult.terminated), [`execute`](tools.md#Tool.execute), [`max_steps`](environment.md#AgenticEnvironment.max_steps), [`close`](environment.md#AgenticEnvironment.close), [`_default_tool_call_parser`](environment.md#_default_tool_call_parser), [`truncated`](environment.md#StepResult.truncated)
- used by: [`run_grouped_episodes`](env_manager.md#RolloutManager.run_grouped_episodes), [`AgenticEnvironment`](environment.md#AgenticEnvironment), [`_wrap_env_with_tools`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._wrap_env_with_tools), [`reset`](environment.md#AgenticEnvironment.reset), [`step`](environment.md#AgenticEnvironment.step), [`_unwrap`](env_manager.md#RolloutManager._unwrap), [`max_steps`](environment.md#AgenticEnvironment.max_steps), [`close`](environment.md#AgenticEnvironment.close)

## Functions
- `_default_tool_call_parser(action: str)` — [`L168`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L168) — Parse tool calls in hermes format: <tool_call>{"name": "...", "arguments": ...}</tool_call>.
- `_make_inference_tool_parser(parser_name: str, tokenizer: tp.Any | None)` — [`L208`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L208) — Create a parser using easydel.inference.tools registered parsers.
- `_make_regex_parser(pattern: str)` — [`L185`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L185) — Create a parser from a regex pattern.
- `create_tool_call_parser(tool_caller: str | None, tokenizer: tp.Any | None = None)` — [`L134`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L134) — Create a tool call parser from a tool_caller specification.
- `parser(action: str)` — [`L192`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L192)
- `parser(action: str)` — [`L222`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/environment.py#L222)

