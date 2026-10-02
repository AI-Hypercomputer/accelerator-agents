---
title: 'Module: easydel/trainers/agentic_moshpit/self_play.py'
type: catalog
provenance: extracted
module: easydel/trainers/agentic_moshpit/self_play.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.agentic_moshpit.self_play`/
symbols:
  SelfPlayEnvironment.step: SelfPlayEnvironment#step().
  SelfPlayEnvironment.reset: SelfPlayEnvironment#reset().
  SelfPlayEnvironment.reset_with_question: SelfPlayEnvironment#reset_with_question().
  LocalQuestionGenerator.generate_batch: LocalQuestionGenerator#generate_batch().
  LocalQuestionGenerator.verify_batch: LocalQuestionGenerator#verify_batch().
  OpenAIQuestionGenerator._chat: OpenAIQuestionGenerator#_chat().
  OpenAIQuestionGenerator.generate: OpenAIQuestionGenerator#generate().
  SelfPlayEnvironment._do_verify: SelfPlayEnvironment#_do_verify().
  SelfPlayEnvironment._make_info: SelfPlayEnvironment#_make_info().
  QuestionGenerator: QuestionGenerator#
  OpenAIQuestionGenerator.verify: OpenAIQuestionGenerator#verify().
  SelfPlayEnvironment._question: SelfPlayEnvironment#_question.
  QuestionGenerator.generate: QuestionGenerator#generate().
  GeneratedQuestion: GeneratedQuestion#
  QuestionGenerator.verify: QuestionGenerator#verify().
  QuestionGenerator.generate_batch: QuestionGenerator#generate_batch().
  SelfPlayEnvironment: SelfPlayEnvironment#
  SelfPlayEnvironment._metadata: SelfPlayEnvironment#_metadata.
  CallableQuestionGenerator.generate: CallableQuestionGenerator#generate().
  SelfPlayEnvironment._topic: SelfPlayEnvironment#_topic.
  GeneratedQuestion.question: GeneratedQuestion#question.
  LocalQuestionGenerator._format_prompt: LocalQuestionGenerator#_format_prompt().
  LocalQuestionGenerator: LocalQuestionGenerator#
  QuestionGenerator.verify_batch: QuestionGenerator#verify_batch().
  SelfPlayEnvironment._solver_turns: SelfPlayEnvironment#_solver_turns.
  SelfPlayEnvironment.set_generate_fn: SelfPlayEnvironment#set_generate_fn().
  SelfPlayEnvironment._generator: SelfPlayEnvironment#_generator.
  LocalQuestionGenerator._tokenizer: LocalQuestionGenerator#_tokenizer.
  GeneratedQuestion.metadata: GeneratedQuestion#metadata.
  LocalQuestionGenerator._generate_fn: LocalQuestionGenerator#_generate_fn.
  LocalQuestionGenerator.generate: LocalQuestionGenerator#generate().
  SelfPlayEnvironment._step_count: SelfPlayEnvironment#_step_count.
  QuestionGenerator.set_generate_fn: QuestionGenerator#set_generate_fn().
  OpenAIQuestionGenerator: OpenAIQuestionGenerator#
  CallableQuestionGenerator: CallableQuestionGenerator#
  SelfPlayEnvironment._defer_verify: SelfPlayEnvironment#_defer_verify.
  SelfPlayEnvironment._final_answer: SelfPlayEnvironment#_final_answer.
  LocalQuestionGenerator.set_generate_fn: LocalQuestionGenerator#set_generate_fn().
  LocalQuestionGenerator.verify: LocalQuestionGenerator#verify().
  LocalQuestionGenerator._last_verify_responses: LocalQuestionGenerator#_last_verify_responses.
  CallableQuestionGenerator.verify: CallableQuestionGenerator#verify().
  SelfPlayEnvironment.__init__: SelfPlayEnvironment#__init__().
  SelfPlayEnvironment.max_steps: SelfPlayEnvironment#max_steps().
  LocalQuestionGenerator._strip_reasoning: LocalQuestionGenerator#_strip_reasoning.
  _deterministic_variety: _deterministic_variety().
  LocalQuestionGenerator._questioner_prompt: LocalQuestionGenerator#_questioner_prompt.
  LocalQuestionGenerator._parse_score: LocalQuestionGenerator#_parse_score().
  OpenAIQuestionGenerator._questioner_prompt: OpenAIQuestionGenerator#_questioner_prompt.
  CallableQuestionGenerator._ver_fn: CallableQuestionGenerator#_ver_fn.
  SelfPlayEnvironment._answer_pattern: SelfPlayEnvironment#_answer_pattern.
  SelfPlayEnvironment._max_steps_override: SelfPlayEnvironment#_max_steps_override.
  LocalQuestionGenerator._verifier_prompt: LocalQuestionGenerator#_verifier_prompt.
  LocalQuestionGenerator._tool_schemas: LocalQuestionGenerator#_tool_schemas.
  LocalQuestionGenerator._q_temperature: LocalQuestionGenerator#_q_temperature.
  LocalQuestionGenerator._q_top_p: LocalQuestionGenerator#_q_top_p.
  LocalQuestionGenerator._q_top_k: LocalQuestionGenerator#_q_top_k.
  LocalQuestionGenerator._v_temperature: LocalQuestionGenerator#_v_temperature.
  LocalQuestionGenerator._v_top_p: LocalQuestionGenerator#_v_top_p.
  LocalQuestionGenerator._v_top_k: LocalQuestionGenerator#_v_top_k.
  OpenAIQuestionGenerator._base_url: OpenAIQuestionGenerator#_base_url.
  OpenAIQuestionGenerator._api_key: OpenAIQuestionGenerator#_api_key.
  OpenAIQuestionGenerator._model: OpenAIQuestionGenerator#_model.
  OpenAIQuestionGenerator._verifier_prompt: OpenAIQuestionGenerator#_verifier_prompt.
  OpenAIQuestionGenerator._temperature: OpenAIQuestionGenerator#_temperature.
  OpenAIQuestionGenerator._max_tokens: OpenAIQuestionGenerator#_max_tokens.
  OpenAIQuestionGenerator._timeout: OpenAIQuestionGenerator#_timeout.
  CallableQuestionGenerator._gen_fn: CallableQuestionGenerator#_gen_fn.
  SelfPlayEnvironment._verify: SelfPlayEnvironment#_verify.
  LocalQuestionGenerator.__init__: LocalQuestionGenerator#__init__().
  OpenAIQuestionGenerator.__init__: OpenAIQuestionGenerator#__init__().
  CallableQuestionGenerator.__init__: CallableQuestionGenerator#__init__().
---
# Module: [`easydel/trainers/agentic_moshpit/self_play.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py)

## Classes
### `CallableQuestionGenerator`  ·  implements/extends QuestionGenerator
- def: [`easydel/trainers/agentic_moshpit/self_play.py:468`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L468)
- doc: Wrap plain functions as a QuestionGenerator.
- signature: `class CallableQuestionGenerator(QuestionGenerator):`
- members:
  - `generate(self, topic: str, seed: int | None = None)` — [`L485`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L485)
  - `verify(self, question: str, answer: str, metadata: dict[str, tp.Any] | None = None)` — [`L488`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L488)
- protocol/private: `__init__`[`L477`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L477), `_gen_fn`[`L482`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L482), `_ver_fn`[`L483`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L483)
- uses (calls/refs, reference-scoped): [`QuestionGenerator`](self_play.md#QuestionGenerator), [`GeneratedQuestion`](self_play.md#GeneratedQuestion), [`question`](self_play.md#GeneratedQuestion.question)
- used by: [`QuestionGenerator`](self_play.md#QuestionGenerator), [`generate`](self_play.md#QuestionGenerator.generate), [`verify`](self_play.md#QuestionGenerator.verify)

### `GeneratedQuestion`
- def: [`easydel/trainers/agentic_moshpit/self_play.py:70`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L70)
- doc: A question produced by a QuestionGenerator.
- signature: `class GeneratedQuestion:`
- members:
  - `metadata` — [`L79`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L79)
  - `question` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L78)
- used by: [`run_grouped_episodes`](env_manager.md#RolloutManager.run_grouped_episodes), [`reset`](self_play.md#SelfPlayEnvironment.reset), [`reset_with_question`](self_play.md#SelfPlayEnvironment.reset_with_question), [`generate_batch`](self_play.md#LocalQuestionGenerator.generate_batch), [`generate`](self_play.md#OpenAIQuestionGenerator.generate), [`generate`](self_play.md#QuestionGenerator.generate), [`generate_batch`](self_play.md#QuestionGenerator.generate_batch), [`generate`](self_play.md#CallableQuestionGenerator.generate), [`generate`](self_play.md#LocalQuestionGenerator.generate)

### `LocalQuestionGenerator`  ·  implements/extends QuestionGenerator
- def: [`easydel/trainers/agentic_moshpit/self_play.py:174`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L174)
- doc: Generate questions and verify answers using a local LLM.
- signature: `class LocalQuestionGenerator(QuestionGenerator):`
- members:
  - `generate(self, topic: str, seed: int | None = None)` — [`L236`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L236)
  - `generate_batch(self, topics: list[str], seeds: list[int | None])` — [`L239`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L239) — Batch-generate questions with a single ``generate_fn`` call.
  - `set_generate_fn(self, generate_fn: tp.Callable[[list[str]], list[str]])` — [`L233`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L233)
  - `verify(self, question: str, answer: str, metadata: dict[str, tp.Any] | None = None)` — [`L281`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L281)
  - `verify_batch(self, questions: list[str], answers: list[str], metadatas: list[dict[str, tp.Any] | None])` — [`L289`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L289) — Batch-verify answers with a single ``generate_fn`` call.
- protocol/private: `__init__`[`L199`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L199), `_format_prompt`[`L327`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L327), `_generate_fn`[`L223`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L223), `_last_verify_responses`[`L324`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L324), `_parse_score`[`L355`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L355), `_q_temperature`[`L217`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L217), `_q_top_k`[`L219`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L219), `_q_top_p`[`L218`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L218), `_questioner_prompt`[`L213`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L213), `_strip_reasoning`[`L231`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L231), `_tokenizer`[`L215`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L215), `_tool_schemas`[`L216`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L216), `_v_temperature`[`L220`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L220), `_v_top_k`[`L222`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L222), `_v_top_p`[`L221`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L221), `_verifier_prompt`[`L214`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L214)
- uses (calls/refs, reference-scoped): [`QuestionGenerator`](self_play.md#QuestionGenerator), [`GeneratedQuestion`](self_play.md#GeneratedQuestion), [`question`](self_play.md#GeneratedQuestion.question), [`metadata`](self_play.md#GeneratedQuestion.metadata), [`_deterministic_variety`](self_play.md#_deterministic_variety)
- used by: [`wrapped_env_factory`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer.wrapped_env_factory), [`QuestionGenerator`](self_play.md#QuestionGenerator), [`verify`](self_play.md#OpenAIQuestionGenerator.verify), [`generate`](self_play.md#QuestionGenerator.generate), [`verify`](self_play.md#QuestionGenerator.verify), [`generate_batch`](self_play.md#QuestionGenerator.generate_batch), [`verify_batch`](self_play.md#QuestionGenerator.verify_batch), [`set_generate_fn`](self_play.md#QuestionGenerator.set_generate_fn)

### `OpenAIQuestionGenerator`  ·  implements/extends QuestionGenerator
- def: [`easydel/trainers/agentic_moshpit/self_play.py:367`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L367)
- doc: Generate questions and verify answers via an OpenAI-compatible API.
- signature: `class OpenAIQuestionGenerator(QuestionGenerator):`
- members:
  - `generate(self, topic: str, seed: int | None = None)` — [`L405`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L405)
  - `verify(self, question: str, answer: str, metadata: dict[str, tp.Any] | None = None)` — [`L419`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L419)
- protocol/private: `__init__`[`L385`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L385), `_api_key`[`L397`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L397), `_base_url`[`L396`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L396), `_chat`[`L435`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L435), `_max_tokens`[`L402`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L402), `_model`[`L398`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L398), `_questioner_prompt`[`L399`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L399), `_temperature`[`L401`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L401), `_timeout`[`L403`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L403), `_verifier_prompt`[`L400`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L400)
- uses (calls/refs, reference-scoped): [`QuestionGenerator`](self_play.md#QuestionGenerator), [`GeneratedQuestion`](self_play.md#GeneratedQuestion), [`question`](self_play.md#GeneratedQuestion.question), [`LocalQuestionGenerator`](self_play.md#LocalQuestionGenerator), [`metadata`](self_play.md#GeneratedQuestion.metadata), [`_deterministic_variety`](self_play.md#_deterministic_variety), [`_parse_score`](self_play.md#LocalQuestionGenerator._parse_score)
- used by: [`QuestionGenerator`](self_play.md#QuestionGenerator), [`generate`](self_play.md#QuestionGenerator.generate), [`verify`](self_play.md#QuestionGenerator.verify)

### `QuestionGenerator`
- def: [`easydel/trainers/agentic_moshpit/self_play.py:82`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L82)
- doc: Abstract interface for question generation backends.
- signature: `class QuestionGenerator:`
- members:
  - `generate(self, topic: str, seed: int | None = None)` — [`L95`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L95) — Generate a single question on the given topic.
  - `generate_batch(self, topics: list[str], seeds: list[int | None])` — [`L107`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L107) — Generate questions for multiple topics/seeds in one batched call.
  - `set_generate_fn(self, generate_fn: tp.Callable[[list[str]], list[str]])` — [`L165`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L165) — Inject the batched generation function from the trainer.
  - `verify(self, question: str, answer: str, metadata: dict[str, tp.Any] | None = None)` — [`L126`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L126) — Verify a solver's answer and return a reward.
  - `verify_batch(self, questions: list[str], answers: list[str], metadatas: list[dict[str, tp.Any] | None])` — [`L144`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L144) — Verify multiple answers in one batched call.
- uses (calls/refs, reference-scoped): [`generate_batch`](self_play.md#LocalQuestionGenerator.generate_batch), [`verify_batch`](self_play.md#LocalQuestionGenerator.verify_batch), [`generate`](self_play.md#OpenAIQuestionGenerator.generate), [`verify`](self_play.md#OpenAIQuestionGenerator.verify), [`GeneratedQuestion`](self_play.md#GeneratedQuestion), [`generate`](self_play.md#CallableQuestionGenerator.generate), [`LocalQuestionGenerator`](self_play.md#LocalQuestionGenerator), [`generate`](self_play.md#LocalQuestionGenerator.generate), [`CallableQuestionGenerator`](self_play.md#CallableQuestionGenerator), [`OpenAIQuestionGenerator`](self_play.md#OpenAIQuestionGenerator), [`set_generate_fn`](self_play.md#LocalQuestionGenerator.set_generate_fn), [`verify`](self_play.md#CallableQuestionGenerator.verify), [`verify`](self_play.md#LocalQuestionGenerator.verify)
- used by: [`run_grouped_episodes`](env_manager.md#RolloutManager.run_grouped_episodes), [`reset`](self_play.md#SelfPlayEnvironment.reset), [`_do_verify`](self_play.md#SelfPlayEnvironment._do_verify), [`LocalQuestionGenerator`](self_play.md#LocalQuestionGenerator), [`set_generate_fn`](self_play.md#SelfPlayEnvironment.set_generate_fn), [`CallableQuestionGenerator`](self_play.md#CallableQuestionGenerator), [`OpenAIQuestionGenerator`](self_play.md#OpenAIQuestionGenerator), [`__init__`](self_play.md#SelfPlayEnvironment.__init__)

### `SelfPlayEnvironment`  ·  implements/extends AgenticEnvironment
- def: [`easydel/trainers/agentic_moshpit/self_play.py:499`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L499)
- doc: Self-play environment where questions are generated by an LLM.
- signature: `class SelfPlayEnvironment(AgenticEnvironment):`
- members:
  - `max_steps(self)` — [`L642`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L642)
  - `reset(self, seed: int | None = None)` — [`L548`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L548) — Reset the environment.
  - `reset_with_question(self, question: GeneratedQuestion)` — [`L571`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L571) — Reset with a pre-generated question (used by batched rollout).
  - `set_generate_fn(self, generate_fn: tp.Callable[[list[str]], list[str]])` — [`L621`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L621) — Propagate the trainer's generate_fn to the question generator.
  - `step(self, action: str)` — [`L582`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L582)
- protocol/private: `__init__`[`L528`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L528), `_answer_pattern`[`L539`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L539), `_defer_verify`[`L545`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L545), `_do_verify`[`L625`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L625), `_final_answer`[`L546`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L546), `_generator`[`L537`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L537), `_make_info`[`L630`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L630), `_max_steps_override`[`L540`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L540), `_metadata`[`L542`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L542), `_question`[`L541`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L541), `_solver_turns`[`L543`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L543), `_step_count`[`L544`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L544), `_topic`[`L536`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L536), `_verify`[`L538`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L538)
- uses (calls/refs, reference-scoped): [`AgenticEnvironment`](environment.md#AgenticEnvironment), [`StepResult`](environment.md#StepResult), [`QuestionGenerator`](self_play.md#QuestionGenerator), [`generate`](self_play.md#QuestionGenerator.generate), [`GeneratedQuestion`](self_play.md#GeneratedQuestion), [`ResetResult`](environment.md#ResetResult), [`observation`](environment.md#StepResult.observation), [`verify`](self_play.md#QuestionGenerator.verify), [`reward`](environment.md#StepResult.reward), [`info`](environment.md#StepResult.info), [`question`](self_play.md#GeneratedQuestion.question), [`terminated`](environment.md#StepResult.terminated), [`info`](environment.md#ResetResult.info), [`metadata`](self_play.md#GeneratedQuestion.metadata), [`observation`](environment.md#ResetResult.observation), [`set_generate_fn`](self_play.md#QuestionGenerator.set_generate_fn)
- used by: [`run_grouped_episodes`](env_manager.md#RolloutManager.run_grouped_episodes), [`wrapped_env_factory`](agentic_moshpit_trainer.md#AgenticMoshPitTrainer.wrapped_env_factory), [`AgenticEnvironment`](environment.md#AgenticEnvironment), [`reset`](environment.md#AgenticEnvironment.reset), [`step`](environment.md#AgenticEnvironment.step), [`max_steps`](environment.md#AgenticEnvironment.max_steps)

## Functions
- `_deterministic_variety(seed: int | None, index: int = 0)` — [`L58`](../../../../../../../raw/code/EasyDeL/easydel/trainers/agentic_moshpit/self_play.py#L58) — Generate a deterministic variety integer from seed and index.

