---
title: 'Module: easydel/trainers/rlvr_trainer/reward_verifiers.py'
type: catalog
provenance: extracted
module: easydel/trainers/rlvr_trainer/reward_verifiers.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.rlvr_trainer.reward_verifiers`/
symbols:
  _extract_answer: _extract_answer().
  MathVerifier._get_gold_answers: MathVerifier#_get_gold_answers().
  MathVerifier.__call__: MathVerifier#__call__().
  CodeVerifier.__call__: CodeVerifier#__call__().
  LengthPenaltyVerifier.__call__: LengthPenaltyVerifier#__call__().
  CodeVerifier._run_tests: CodeVerifier#_run_tests().
  _normalize_number: _normalize_number().
  FormatVerifier.__call__: FormatVerifier#__call__().
  MathVerifier: MathVerifier#
  CodeVerifier._get_tests: CodeVerifier#_get_tests().
  FormatVerifier: FormatVerifier#
  LengthPenaltyVerifier: LengthPenaltyVerifier#
  _HASH_ANSWER_RE: _HASH_ANSWER_RE.
  FormatVerifier._pattern: FormatVerifier#_pattern.
  LengthPenaltyVerifier._target: LengthPenaltyVerifier#_target.
  _BOXED_RE: _BOXED_RE.
  MathVerifier._answer_key: MathVerifier#_answer_key.
  MathVerifier._tolerance: MathVerifier#_tolerance.
  CodeVerifier: CodeVerifier#
  CodeVerifier._timeout: CodeVerifier#_timeout.
  CodeVerifier._test_key: CodeVerifier#_test_key.
  CodeVerifier._extract_code: CodeVerifier#_extract_code().
  CodeVerifier._handler: CodeVerifier#_handler().
  FormatVerifier._full: FormatVerifier#_full.
  LengthPenaltyVerifier._min: LengthPenaltyVerifier#_min.
  LengthPenaltyVerifier._max: LengthPenaltyVerifier#_max.
  MathVerifier.__init__: MathVerifier#__init__().
  CodeVerifier.__init__: CodeVerifier#__init__().
  FormatVerifier.__init__: FormatVerifier#__init__().
  LengthPenaltyVerifier.__init__: LengthPenaltyVerifier#__init__().
---
# Module: [`easydel/trainers/rlvr_trainer/reward_verifiers.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py)

## Classes
### `CodeVerifier`
- def: [`easydel/trainers/rlvr_trainer/reward_verifiers.py:170`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L170)
- doc: Code execution verifier.
- signature: `class CodeVerifier:`
- members:
  - `__call__(self, *, prompts: tp.Any = None, completions: tp.Any = None, batch: dict[str, tp.Any] | None = None, **kwargs: tp.Any)` — [`L188`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L188) — Score completions by executing code against test cases.
  - `_extract_code(text: str)` — [`L225`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L225) — Extract code from markdown or XML code blocks.
  - `_run_tests(self, code: str, test_code: str)` — [`L235`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L235) — Execute code + test assertions in a sandboxed environment.
- protocol/private: `__init__`[`L184`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L184), `_get_tests`[`L260`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L260), `_handler`[`L244`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L244), `_test_key`[`L186`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L186), `_timeout`[`L185`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L185)

### `FormatVerifier`
- def: [`easydel/trainers/rlvr_trainer/reward_verifiers.py:274`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L274)
- doc: Format compliance verifier.
- signature: `class FormatVerifier:`
- members:
  - `__call__(self, *, prompts: tp.Any = None, completions: tp.Any = None, **kwargs: tp.Any)` — [`L293`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L293) — Score completions by format compliance.
- protocol/private: `__init__`[`L289`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L289), `_full`[`L291`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L291), `_pattern`[`L290`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L290)
- used by: [`_build_verifiers`](rlvr_trainer.md#RLVRTrainer._build_verifiers)

### `LengthPenaltyVerifier`
- def: [`easydel/trainers/rlvr_trainer/reward_verifiers.py:323`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L323)
- doc: Length-based reward modifier.
- signature: `class LengthPenaltyVerifier:`
- members:
  - `__call__(self, *, prompts: tp.Any = None, completions: tp.Any = None, **kwargs: tp.Any)` — [`L346`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L346) — Score completions by length proximity to target.
- protocol/private: `__init__`[`L336`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L336), `_max`[`L344`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L344), `_min`[`L343`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L343), `_target`[`L342`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L342)
- used by: [`_build_verifiers`](rlvr_trainer.md#RLVRTrainer._build_verifiers)

### `MathVerifier`
- def: [`easydel/trainers/rlvr_trainer/reward_verifiers.py:75`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L75)
- doc: Rule-based math answer verifier.
- signature: `class MathVerifier:`
- members:
  - `__call__(self, *, prompts: tp.Any = None, completions: tp.Any = None, batch: dict[str, tp.Any] | None = None, **kwargs: tp.Any)` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L94) — Score completions against gold math answers.
  - `_get_gold_answers(self, batch: dict[str, tp.Any] | None, n: int)` — [`L138`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L138) — Extract gold answers from the batch dict.
- protocol/private: `__init__`[`L90`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L90), `_answer_key`[`L91`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L91), `_tolerance`[`L92`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L92)
- uses (calls/refs, reference-scoped): [`_extract_answer`](reward_verifiers.md#_extract_answer), [`_normalize_number`](reward_verifiers.md#_normalize_number), [`_HASH_ANSWER_RE`](reward_verifiers.md#_HASH_ANSWER_RE)
- used by: [`__init__`](rlvr_trainer.md#RLVRTrainer.__init__), [`_build_verifiers`](rlvr_trainer.md#RLVRTrainer._build_verifiers)

## Functions
- `_extract_answer(text: str)` — [`L57`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L57) — Extract a numeric answer from model output.
- `_normalize_number(text: str)` — [`L48`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L48) — Strip formatting and return a canonical number string.

## Module values
- `_BOXED_RE` — [`L44`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L44)
- `_HASH_ANSWER_RE` — [`L45`](../../../../../../../raw/code/EasyDeL/easydel/trainers/rlvr_trainer/reward_verifiers.py#L45)

