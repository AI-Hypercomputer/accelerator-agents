---
title: 'Module: easydel/trainers/agentic_moshpit/env_manager.py'
type: catalog
provenance: extracted
module: easydel/trainers/agentic_moshpit/env_manager.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.agentic_moshpit.env_manager`/
symbols:
  RolloutManager.run_grouped_episodes: RolloutManager#run_grouped_episodes().
  RolloutManager._build_trajectory: RolloutManager#_build_trajectory().
  turn_record_to_message: turn_record_to_message().
  RolloutManager.collate_trajectories: RolloutManager#collate_trajectories().
  _EpisodeState.turns: _EpisodeState#turns.
  TurnRecord: TurnRecord#
  RolloutManager._unwrap: RolloutManager#_unwrap().
  TurnRecord.role: TurnRecord#role.
  TurnRecord.content: TurnRecord#content.
  _EpisodeState.env: _EpisodeState#env.
  RolloutManager.tokenizer: RolloutManager#tokenizer.
  TurnRecord.tool_calls: TurnRecord#tool_calls.
  TrajectoryResult: TrajectoryResult#
  _EpisodeState.step_rewards: _EpisodeState#step_rewards.
  TrajectoryResult.turns: TrajectoryResult#turns.
  TrajectoryResult.input_ids: TrajectoryResult#input_ids.
  _EpisodeState.episode_reward: _EpisodeState#episode_reward.
  RolloutManager.tool_schemas: RolloutManager#tool_schemas.
  _normalize_tool_call_payloads: _normalize_tool_call_payloads().
  TurnRecord.is_response: TurnRecord#is_response.
  TurnRecord.visible_content: TurnRecord#visible_content.
  TurnRecord.reasoning: TurnRecord#reasoning.
  TrajectoryResult.response_mask: TrajectoryResult#response_mask.
  _EpisodeState: _EpisodeState#
  _EpisodeState.num_steps: _EpisodeState#num_steps.
  _EpisodeState.info: _EpisodeState#info.
  RolloutManager: RolloutManager#
  RolloutManager.__init__: RolloutManager#__init__().
  _coerce_mapping_like: _coerce_mapping_like().
  TrajectoryResult.episode_reward: TrajectoryResult#episode_reward.
  TrajectoryResult.step_rewards: TrajectoryResult#step_rewards.
  TrajectoryResult.num_steps: TrajectoryResult#num_steps.
  _EpisodeState.done: _EpisodeState#done.
  RolloutManager.max_seq_length: RolloutManager#max_seq_length.
  RolloutManager.system_prompt: RolloutManager#system_prompt.
  RolloutManager._pad_token_id: RolloutManager#_pad_token_id.
  TurnRecord.raw_content: TurnRecord#raw_content.
  TrajectoryResult.attention_mask: TrajectoryResult#attention_mask.
  TrajectoryResult.prompt_mask: TrajectoryResult#prompt_mask.
  TrajectoryResult.info: TrajectoryResult#info.
  RolloutManager.max_steps: RolloutManager#max_steps.
  logger: logger.
  TurnRecord.token_ids: TurnRecord#token_ids.
  TurnRecord.reward: TurnRecord#reward.
---
# Module: [`easydel/trainers/agentic_moshpit/env_manager.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py)

## Classes
### `RolloutManager`
- def: [`easydel/trainers/agentic_moshpit/env_manager.py:217`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L217)
- doc: Manages batched multi-turn rollouts for agentic MoshPit training.
- signature: `class RolloutManager:`
- members:
  - `_build_trajectory(self, ep: _EpisodeState)` — [`L529`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L529) — Tokenize turns and build a TrajectoryResult.
  - `collate_trajectories(self, trajectories: list[TrajectoryResult], max_prompt_length: int, max_completion_length: int)` — [`L445`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L445) — Collate trajectories into batched arrays for training.
  - `run_grouped_episodes(self, env_factory: tp.Callable[[], AgenticEnvironment], generate_fn: tp.Callable[[list[str]], list[str]], group_size: int, base_seed: int = 0, num_groups: int = 1)` — [`L252`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L252) — Run all episodes with batched parallel generation.
  - `max_seq_length` — [`L246`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L246)
  - `max_steps` — [`L245`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L245)
  - `system_prompt` — [`L247`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L247)
  - `tokenizer` — [`L244`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L244)
  - `tool_schemas` — [`L248`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L248)
- protocol/private: `__init__`[`L236`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L236), `_pad_token_id`[`L250`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L250), `_unwrap`[`L290`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L290)
- uses (calls/refs, reference-scoped): [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`reset_with_question`](self_play.md#SelfPlayEnvironment.reset_with_question), [`AgenticEnvironment`](environment.md#AgenticEnvironment), [`step_with_tool_calls`](environment.md#ToolEnvWrapper.step_with_tool_calls), [`turn_record_to_message`](env_manager.md#turn_record_to_message), [`_make_info`](self_play.md#SelfPlayEnvironment._make_info), [`_question`](self_play.md#SelfPlayEnvironment._question), [`ResetResult`](environment.md#ResetResult), [`turns`](env_manager.md#_EpisodeState.turns), [`TurnRecord`](env_manager.md#TurnRecord), [`observation`](environment.md#StepResult.observation), [`ToolEnvWrapper`](environment.md#ToolEnvWrapper), [`reset`](environment.md#AgenticEnvironment.reset), [`step`](environment.md#AgenticEnvironment.step), [`reward`](environment.md#StepResult.reward), [`SelfPlayEnvironment`](self_play.md#SelfPlayEnvironment), [`generate_batch`](self_play.md#QuestionGenerator.generate_batch), [`_metadata`](self_play.md#SelfPlayEnvironment._metadata), [`content`](env_manager.md#TurnRecord.content), [`role`](env_manager.md#TurnRecord.role), [`env`](env_manager.md#_EpisodeState.env), [`_topic`](self_play.md#SelfPlayEnvironment._topic), [`info`](environment.md#StepResult.info), [`question`](self_play.md#GeneratedQuestion.question), [`terminated`](environment.md#StepResult.terminated), [`_tool_calls_this_step`](environment.md#ToolEnvWrapper._tool_calls_this_step), [`verify_batch`](self_play.md#QuestionGenerator.verify_batch), [`TrajectoryResult`](env_manager.md#TrajectoryResult), [`_generator`](self_play.md#SelfPlayEnvironment._generator), [`env`](environment.md#ToolEnvWrapper.env), [`info`](environment.md#ResetResult.info), [`metadata`](self_play.md#GeneratedQuestion.metadata), [`observation`](environment.md#ResetResult.observation), [`step_rewards`](env_manager.md#_EpisodeState.step_rewards), [`tool_calls`](env_manager.md#TurnRecord.tool_calls), [`close`](environment.md#AgenticEnvironment.close), [`turns`](env_manager.md#TrajectoryResult.turns), [`episode_reward`](env_manager.md#_EpisodeState.episode_reward), [`input_ids`](env_manager.md#TrajectoryResult.input_ids), [`_EpisodeState`](env_manager.md#_EpisodeState)  (+17 more)
- used by: [`_preprocess_batch_input`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._preprocess_batch_input), [`_rollout_manager`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._rollout_manager)

### `TrajectoryResult`
- def: [`easydel/trainers/agentic_moshpit/env_manager.py:178`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L178)
- doc: Complete trajectory from a single episode rollout.
- signature: `class TrajectoryResult:`
- members:
  - `attention_mask` — [`L194`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L194)
  - `episode_reward` — [`L197`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L197)
  - `info` — [`L201`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L201)
  - `input_ids` — [`L193`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L193)
  - `num_steps` — [`L199`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L199)
  - `prompt_mask` — [`L195`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L195)
  - `response_mask` — [`L196`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L196)
  - `step_rewards` — [`L198`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L198)
  - `turns` — [`L200`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L200)
- uses (calls/refs, reference-scoped): [`TurnRecord`](env_manager.md#TurnRecord)
- used by: [`run_grouped_episodes`](env_manager.md#RolloutManager.run_grouped_episodes), [`_preprocess_batch_input`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._preprocess_batch_input), [`_build_trajectory`](env_manager.md#RolloutManager._build_trajectory), [`collate_trajectories`](env_manager.md#RolloutManager.collate_trajectories)

### `TurnRecord`
- def: [`easydel/trainers/agentic_moshpit/env_manager.py:130`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L130)
- doc: Record of a single interaction turn.
- signature: `class TurnRecord:`
- members:
  - `content` — [`L148`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L148)
  - `is_response` — [`L150`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L150)
  - `raw_content` — [`L153`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L153)
  - `reasoning` — [`L154`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L154)
  - `reward` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L151)
  - `role` — [`L147`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L147)
  - `token_ids` — [`L149`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L149)
  - `tool_calls` — [`L155`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L155)
  - `visible_content` — [`L152`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L152)
- used by: [`run_grouped_episodes`](env_manager.md#RolloutManager.run_grouped_episodes), [`_preprocess_batch_input`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer._preprocess_batch_input), [`_build_trajectory`](env_manager.md#RolloutManager._build_trajectory), [`turn_record_to_message`](env_manager.md#turn_record_to_message), [`turns`](env_manager.md#_EpisodeState.turns), [`turns`](env_manager.md#TrajectoryResult.turns)

### `_EpisodeState`
- def: [`easydel/trainers/agentic_moshpit/env_manager.py:205`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L205)
- doc: Mutable state for a single in-flight episode.
- signature: `class _EpisodeState:`
- members:
  - `done` — [`L213`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L213)
  - `env` — [`L208`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L208)
  - `episode_reward` — [`L211`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L211)
  - `info` — [`L214`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L214)
  - `num_steps` — [`L212`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L212)
  - `step_rewards` — [`L210`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L210)
  - `turns` — [`L209`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L209)
- uses (calls/refs, reference-scoped): [`AgenticEnvironment`](environment.md#AgenticEnvironment), [`TurnRecord`](env_manager.md#TurnRecord)
- used by: [`run_grouped_episodes`](env_manager.md#RolloutManager.run_grouped_episodes), [`_build_trajectory`](env_manager.md#RolloutManager._build_trajectory)

## Functions
- `_coerce_mapping_like(value: tp.Any)` — [`L55`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L55) — Coerce JSON-string payloads into mapping-like objects when possible.
- `_normalize_tool_call_payloads(tool_calls: tp.Any)` — [`L66`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L66) — Normalize tool calls for HF/Jinja chat-template compatibility.
- `turn_record_to_message(turn: TurnRecord)` — [`L158`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L158) — Convert a stored turn into a chat-template message payload.

## Module values
- `logger` — [`L52`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/env_manager.py#L52)

