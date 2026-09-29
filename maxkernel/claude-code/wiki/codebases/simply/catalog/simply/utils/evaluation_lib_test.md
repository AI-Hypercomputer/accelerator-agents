---
title: 'Module: simply/utils/evaluation_lib_test.py'
type: catalog
provenance: extracted
module: simply/utils/evaluation_lib_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.evaluation_lib_test`/
symbols:
  TEST_EXAMPLE: TEST_EXAMPLE.
  EvaluationLibTest.test_few_shot_gsm8k_evaluation: EvaluationLibTest#test_few_shot_gsm8k_evaluation().
  EvaluationLibTest.test_zero_shot_boxed_in_question_evaluation: EvaluationLibTest#test_zero_shot_boxed_in_question_evaluation().
  EvaluationLibTest.test_zero_shot_cot_boxed_in_question_evaluation: EvaluationLibTest#test_zero_shot_cot_boxed_in_question_evaluation().
  EvaluationLibTest.test_zero_shot_deep_seek_qwen_r1_cot_boxed: EvaluationLibTest#test_zero_shot_deep_seek_qwen_r1_cot_boxed().
  EvaluationLibTest.test_zero_shot_system_cot_boxed: EvaluationLibTest#test_zero_shot_system_cot_boxed().
  AnswerNormalizerTest.test_match: AnswerNormalizerTest#test_match().
  GSM8K_8_SHOTS_PROMPT: GSM8K_8_SHOTS_PROMPT.
  TEST_MMLU_EXAMPLE_1: TEST_MMLU_EXAMPLE_1.
  TEST_MMLU_EXAMPLE_2: TEST_MMLU_EXAMPLE_2.
  TEST_MMLU_EXAMPLE_3: TEST_MMLU_EXAMPLE_3.
  TEST_GPQA_EXAMPLE_1: TEST_GPQA_EXAMPLE_1.
  TEST_MMLU_2_SHOTS_PROMPT: TEST_MMLU_2_SHOTS_PROMPT.
  EvaluationLibTest: EvaluationLibTest#
  AnswerNormalizerTest: AnswerNormalizerTest#
---
# Module: [`simply/utils/evaluation_lib_test.py`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py)

## Classes
### `AnswerNormalizerTest`  ·  implements/extends TestCase
- def: [`simply/utils/evaluation_lib_test.py:185`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py#L185)
- signature: `class AnswerNormalizerTest(parameterized.TestCase):`
- members:
  - `test_match(self, answer, ground_truth, expected)` — [`L208`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py#L208)
- uses (calls/refs, reference-scoped): [`match`](math_eval.md#match)

### `EvaluationLibTest`  ·  implements/extends TestCase
- def: [`simply/utils/evaluation_lib_test.py:106`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py#L106)
- signature: `class EvaluationLibTest(absltest.TestCase):`
- members:
  - `test_few_shot_gsm8k_evaluation(self)` — [`L108`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py#L108)
  - `test_zero_shot_boxed_in_question_evaluation(self)` — [`L119`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py#L119)
  - `test_zero_shot_cot_boxed_in_question_evaluation(self)` — [`L135`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py#L135)
  - `test_zero_shot_deep_seek_qwen_r1_cot_boxed(self)` — [`L149`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py#L149)
  - `test_zero_shot_system_cot_boxed(self)` — [`L163`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py#L163)
- uses (calls/refs, reference-scoped): [`TEST_EXAMPLE`](evaluation_lib_test.md#TEST_EXAMPLE), [`ZeroShotBoxedInQuestionEvaluation`](evaluation_lib.md#ZeroShotBoxedInQuestionEvaluation), [`ZeroShotCoTBoxedInQuestionEvaluation`](evaluation_lib.md#ZeroShotCoTBoxedInQuestionEvaluation), [`ZeroShotDeepSeekQwenR1CoTBoxed`](evaluation_lib.md#ZeroShotDeepSeekQwenR1CoTBoxed), [`FewShotGSM8KEvaluation`](evaluation_lib.md#FewShotGSM8KEvaluation), [`ZeroShotSystemCoTBoxed`](evaluation_lib.md#ZeroShotSystemCoTBoxed), [`GSM8K_8_SHOTS_PROMPT`](evaluation_lib_test.md#GSM8K_8_SHOTS_PROMPT)

## Module values
- `GSM8K_8_SHOTS_PROMPT` — [`L67`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py#L67)
- `TEST_EXAMPLE` — [`L26`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py#L26)
- `TEST_GPQA_EXAMPLE_1` — [`L57`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py#L57)
- `TEST_MMLU_2_SHOTS_PROMPT` — [`L95`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py#L95)
- `TEST_MMLU_EXAMPLE_1` — [`L36`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py#L36)
- `TEST_MMLU_EXAMPLE_2` — [`L42`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py#L42)
- `TEST_MMLU_EXAMPLE_3` — [`L48`](../../../../../../raw/code/simply/simply/utils/evaluation_lib_test.py#L48)

