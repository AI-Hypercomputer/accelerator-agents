---
title: 'Module: simply/utils/evaluation_lib.py'
type: catalog
provenance: extracted
module: simply/utils/evaluation_lib.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.evaluation_lib`/
symbols:
  FewShotGSM8KEvaluation.get_messages: FewShotGSM8KEvaluation#get_messages().
  ZeroShotBoxedInQuestionEvaluation: ZeroShotBoxedInQuestionEvaluation#
  Evaluation.get_sampling_input: Evaluation#get_sampling_input().
  ZeroShotCoTBoxedInQuestionEvaluation: ZeroShotCoTBoxedInQuestionEvaluation#
  EvaluationRegistry: EvaluationRegistry#
  ZeroShotDeepSeekQwenR1CoTBoxed: ZeroShotDeepSeekQwenR1CoTBoxed#
  FewShotGSM8KEvaluation: FewShotGSM8KEvaluation#
  ZeroShotBoxedInQuestionEvaluation.get_prompt: ZeroShotBoxedInQuestionEvaluation#get_prompt().
  ZeroShotBoxedInQuestionEvaluation.evaluate: ZeroShotBoxedInQuestionEvaluation#evaluate().
  Evaluation.get_messages: Evaluation#get_messages().
  FewShotGSM8KEvaluation.shots: FewShotGSM8KEvaluation#shots().
  maybe_remove_comma: maybe_remove_comma.
  Evaluation: Evaluation#
  Evaluation.evaluate: Evaluation#evaluate().
  FewShotGSM8KEvaluation.evaluate: FewShotGSM8KEvaluation#evaluate().
  ZeroShotBoxedInQuestionEvaluation.get_messages: ZeroShotBoxedInQuestionEvaluation#get_messages().
  ZeroShotCoTBoxedInQuestionEvaluation.answer_start: ZeroShotCoTBoxedInQuestionEvaluation#answer_start.
  ZeroShotDeepSeekQwenR1CoTBoxed.answer_start: ZeroShotDeepSeekQwenR1CoTBoxed#answer_start.
  find_number: find_number.
  extract_boxed_answer: extract_boxed_answer.
  match: match.
  FewShotGSM8KEvaluation.turn_template: FewShotGSM8KEvaluation#turn_template.
  FewShotGSM8KEvaluation.default_8_shots_txt: FewShotGSM8KEvaluation#default_8_shots_txt.
  QAToolUseEvaluation: QAToolUseEvaluation#
  ZeroShotSystemCoTBoxed: ZeroShotSystemCoTBoxed#
  FewShotGSM8KEvaluation.n_shots: FewShotGSM8KEvaluation#n_shots.
  FewShotGSM8KEvaluation.input_marker: FewShotGSM8KEvaluation#input_marker.
  FewShotGSM8KEvaluation.output_marker: FewShotGSM8KEvaluation#output_marker.
  FewShotGSM8KEvaluation.input_end: FewShotGSM8KEvaluation#input_end.
  ZeroShotBoxedInQuestionEvaluation.system_message: ZeroShotBoxedInQuestionEvaluation#system_message.
  GSM8K_8_SHOTS_TXT: GSM8K_8_SHOTS_TXT.
  FewShotGSM8KEvaluation.prompt_template: FewShotGSM8KEvaluation#prompt_template.
  FewShotGSM8KEvaluation.partial_turn_template: FewShotGSM8KEvaluation#partial_turn_template.
  FewShotGSM8KEvaluation.question_template: FewShotGSM8KEvaluation#question_template.
  FewShotGSM8KEvaluation.output_end: FewShotGSM8KEvaluation#output_end.
  FewShotGSM8KEvaluation.system_marker: FewShotGSM8KEvaluation#system_marker.
  FewShotGSM8KEvaluation.system_message: FewShotGSM8KEvaluation#system_message.
  ZeroShotBoxedInQuestionEvaluation.question_start: ZeroShotBoxedInQuestionEvaluation#question_start.
  ZeroShotBoxedInQuestionEvaluation.question_end: ZeroShotBoxedInQuestionEvaluation#question_end.
  ZeroShotBoxedInQuestionEvaluation.answer_start: ZeroShotBoxedInQuestionEvaluation#answer_start.
  EvaluationRegistry.namespace: EvaluationRegistry#namespace.
  QAToolUseEvaluation.question_start: QAToolUseEvaluation#question_start.
  ZeroShotDeepSeekQwenR1CoTBoxed.question_start: ZeroShotDeepSeekQwenR1CoTBoxed#question_start.
  ZeroShotDeepSeekQwenR1CoTBoxed.question_end: ZeroShotDeepSeekQwenR1CoTBoxed#question_end.
  ZeroShotSystemCoTBoxed.system_message: ZeroShotSystemCoTBoxed#system_message.
  ZeroShotSystemCoTBoxed.question_start: ZeroShotSystemCoTBoxed#question_start.
  ZeroShotSystemCoTBoxed.question_end: ZeroShotSystemCoTBoxed#question_end.
  ZeroShotSystemCoTBoxed.answer_start: ZeroShotSystemCoTBoxed#answer_start.
---
# Module: [`simply/utils/evaluation_lib.py`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py)

## Classes
### `Evaluation`  ·  implements/extends ABC
- def: [`simply/utils/evaluation_lib.py:66`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L66)
- doc: Base class for evaluation.
- signature: `class Evaluation(abc.ABC):`
- members:
  - `evaluate(self, example: Mapping[str, Any], response: Any)` — [`L70`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L70) — Evaluates the response in for the given example.
  - `get_messages(self, example: Mapping[str, Any])` — [`L75`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L75) — API that is compatible with HuggingFace apply_chat_template.
  - `get_sampling_input(self, example: Mapping[str, Any], lm_format: lm_format_lib.LMFormat)` — [`L89`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L89) — Converts raw example to SamplingInput.
- uses (calls/refs, reference-scoped): [`get_messages`](evaluation_lib.md#FewShotGSM8KEvaluation.get_messages), [`Chunk`](sampling_lib.md#Chunk), [`input_as_chunks`](sampling_lib.md#input_as_chunks), [`SamplingInput`](sampling_lib.md#SamplingInput), [`format`](lm_format.md#LMFormat.format), [`LMFormat`](lm_format.md#LMFormat), [`evaluate`](evaluation_lib.md#ZeroShotBoxedInQuestionEvaluation.evaluate), [`evaluate`](evaluation_lib.md#FewShotGSM8KEvaluation.evaluate), [`get_messages`](evaluation_lib.md#ZeroShotBoxedInQuestionEvaluation.get_messages)
- used by: [`compute_stats`](../rl_lib.md#compute_stats), [`ZeroShotBoxedInQuestionEvaluation`](evaluation_lib.md#ZeroShotBoxedInQuestionEvaluation), [`FewShotGSM8KEvaluation`](evaluation_lib.md#FewShotGSM8KEvaluation), [`evaluation`](../config_lib.md#RLExperimentConfig.evaluation), [`validation_evaluation`](../config_lib.md#RLExperimentConfig.validation_evaluation)

### `EvaluationRegistry`  ·  implements/extends RootRegistry
- def: [`simply/utils/evaluation_lib.py:60`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L60)
- doc: Evaluation registry.
- signature: `class EvaluationRegistry(registry.RootRegistry):`
- members:
  - `namespace` — [`L63`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L63)
- uses (calls/refs, reference-scoped): [`RootRegistry`](registry.md#RootRegistry)
- used by: [`main`](../eval/page_decode_eval.md#main), [`main`](../eval/decode_eval.md#main), [`RootRegistry`](registry.md#RootRegistry), [`main`](../eval/remote_decode_eval.md#main), [`ZeroShotBoxedInQuestionEvaluation`](evaluation_lib.md#ZeroShotBoxedInQuestionEvaluation), [`ZeroShotCoTBoxedInQuestionEvaluation`](evaluation_lib.md#ZeroShotCoTBoxedInQuestionEvaluation), [`ZeroShotDeepSeekQwenR1CoTBoxed`](evaluation_lib.md#ZeroShotDeepSeekQwenR1CoTBoxed), [`FewShotGSM8KEvaluation`](evaluation_lib.md#FewShotGSM8KEvaluation), [`answer_start`](evaluation_lib.md#ZeroShotCoTBoxedInQuestionEvaluation.answer_start), [`answer_start`](evaluation_lib.md#ZeroShotDeepSeekQwenR1CoTBoxed.answer_start)

### `FewShotGSM8KEvaluation`
- def: [`simply/utils/evaluation_lib.py:124`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L124)
- doc: Few-shot GSM8K evaluation.
- signature: `class FewShotGSM8KEvaluation(Evaluation):`
- members:
  - `evaluate(self, example: Mapping[str, Any], response: str)` — [`L192`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L192) — Evaluate the response in for the given example.
  - `get_messages(self, example: Mapping[str, Any])` — [`L163`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L163) — Returns the prompt for the given example.
  - `shots(self)` — [`L149`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L149) — Returns the shots.
  - `default_8_shots_txt` — [`L135`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L135)
  - `input_end` — [`L140`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L140)
  - `input_marker` — [`L138`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L138)
  - `n_shots` — [`L136`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L136)
  - `output_end` — [`L141`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L141)
  - `output_marker` — [`L139`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L139)
  - `partial_turn_template` — [`L128`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L128)
  - `prompt_template` — [`L127`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L127)
  - `question_template` — [`L132`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L132)
  - `system_marker` — [`L142`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L142)
  - `system_message` — [`L143`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L143)
  - `turn_template` — [`L131`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L131)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`EvaluationRegistry`](evaluation_lib.md#EvaluationRegistry), [`maybe_remove_comma`](evaluation_lib.md#maybe_remove_comma), [`Evaluation`](evaluation_lib.md#Evaluation), [`find_number`](evaluation_lib.md#find_number), [`GSM8K_8_SHOTS_TXT`](evaluation_lib.md#GSM8K_8_SHOTS_TXT)
- used by: [`test_few_shot_gsm8k_evaluation`](evaluation_lib_test.md#EvaluationLibTest.test_few_shot_gsm8k_evaluation), [`get_messages`](evaluation_lib.md#Evaluation.get_messages), [`evaluate`](evaluation_lib.md#Evaluation.evaluate)

### `QAToolUseEvaluation`
- def: [`simply/utils/evaluation_lib.py:277`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L277)
- signature: `class QAToolUseEvaluation(ZeroShotBoxedInQuestionEvaluation):`
- members:
  - `question_start` — [`L278`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L278)
- uses (calls/refs, reference-scoped): [`ZeroShotBoxedInQuestionEvaluation`](evaluation_lib.md#ZeroShotBoxedInQuestionEvaluation)
- used by: [`gemma3_4b_it_simple_qa_number_only_tool_use_rl`](../config_lib.md#gemma3_4b_it_simple_qa_number_only_tool_use_rl)

### `ZeroShotBoxedInQuestionEvaluation`
- def: [`simply/utils/evaluation_lib.py:210`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L210)
- doc: 0-shot that asks for \boxed{} in the question part.
- signature: `class ZeroShotBoxedInQuestionEvaluation(Evaluation):`
- members:
  - `evaluate(self, example: Mapping[str, Any], response: str)` — [`L242`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L242) — Rates the response in for the given example.
  - `get_messages(self, example: Mapping[str, Any])` — [`L231`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L231) — Returns the prompt for the given example.
  - `get_prompt(self, example: Mapping[str, Any])` — [`L222`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L222) — Returns the prompt for the given example.
  - `answer_start` — [`L220`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L220)
  - `question_end` — [`L217`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L217)
  - `question_start` — [`L216`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L216)
  - `system_message` — [`L215`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L215)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`EvaluationRegistry`](evaluation_lib.md#EvaluationRegistry), [`maybe_remove_comma`](evaluation_lib.md#maybe_remove_comma), [`Evaluation`](evaluation_lib.md#Evaluation), [`extract_boxed_answer`](evaluation_lib.md#extract_boxed_answer), [`match`](evaluation_lib.md#match)
- used by: [`lm_rl_test`](../config_lib.md#lm_rl_test), [`gemma2_2b_gsm8k_0shot_rl`](../config_lib.md#gemma2_2b_gsm8k_0shot_rl), [`gemma2_2b_it_gsm8k_0shot_rl`](../config_lib.md#gemma2_2b_it_gsm8k_0shot_rl), [`ZeroShotCoTBoxedInQuestionEvaluation`](evaluation_lib.md#ZeroShotCoTBoxedInQuestionEvaluation), [`ZeroShotDeepSeekQwenR1CoTBoxed`](evaluation_lib.md#ZeroShotDeepSeekQwenR1CoTBoxed), [`get_messages`](evaluation_lib.md#Evaluation.get_messages), [`evaluate`](evaluation_lib.md#Evaluation.evaluate), [`evaluation`](../config_lib.md#RLExperimentConfig.evaluation), [`test_zero_shot_boxed_in_question_evaluation`](evaluation_lib_test.md#EvaluationLibTest.test_zero_shot_boxed_in_question_evaluation), [`QAToolUseEvaluation`](evaluation_lib.md#QAToolUseEvaluation)

### `ZeroShotCoTBoxedInQuestionEvaluation`
- def: [`simply/utils/evaluation_lib.py:262`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L262)
- doc: 0-shot that asks for \boxed{} in the question part.
- signature: `class ZeroShotCoTBoxedInQuestionEvaluation(ZeroShotBoxedInQuestionEvaluation):`
- members:
  - `answer_start` — [`L272`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L272)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`ZeroShotBoxedInQuestionEvaluation`](evaluation_lib.md#ZeroShotBoxedInQuestionEvaluation), [`EvaluationRegistry`](evaluation_lib.md#EvaluationRegistry)
- used by: [`gemma2_2b_dsr40k_cot_0shot_rl`](../config_lib.md#gemma2_2b_dsr40k_cot_0shot_rl), [`gemma2_2b_gsm8k_cot_0shot_rl`](../config_lib.md#gemma2_2b_gsm8k_cot_0shot_rl), [`gemma2_2b_it_dsr40k_cot_0shot_rl`](../config_lib.md#gemma2_2b_it_dsr40k_cot_0shot_rl), [`gemma2_2b_it_gsm8k_cot_0shot_rl`](../config_lib.md#gemma2_2b_it_gsm8k_cot_0shot_rl), [`test_zero_shot_cot_boxed_in_question_evaluation`](evaluation_lib_test.md#EvaluationLibTest.test_zero_shot_cot_boxed_in_question_evaluation)

### `ZeroShotDeepSeekQwenR1CoTBoxed`
- def: [`simply/utils/evaluation_lib.py:294`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L294)
- doc: 0-shot that asks to reason step by step and put answer in \boxed{}.
- signature: `class ZeroShotDeepSeekQwenR1CoTBoxed(ZeroShotBoxedInQuestionEvaluation):`
- members:
  - `answer_start` — [`L307`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L307)
  - `question_end` — [`L304`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L304)
  - `question_start` — [`L303`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L303)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`ZeroShotBoxedInQuestionEvaluation`](evaluation_lib.md#ZeroShotBoxedInQuestionEvaluation), [`EvaluationRegistry`](evaluation_lib.md#EvaluationRegistry)
- used by: [`deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl`](../config_lib.md#deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl), [`test_zero_shot_deep_seek_qwen_r1_cot_boxed`](evaluation_lib_test.md#EvaluationLibTest.test_zero_shot_deep_seek_qwen_r1_cot_boxed), [`ZeroShotSystemCoTBoxed`](evaluation_lib.md#ZeroShotSystemCoTBoxed)

### `ZeroShotSystemCoTBoxed`
- def: [`simply/utils/evaluation_lib.py:312`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L312)
- doc: 0-shot that asks to reason step by step and put answer in \boxed{}.
- signature: `class ZeroShotSystemCoTBoxed(ZeroShotDeepSeekQwenR1CoTBoxed):`
- members:
  - `answer_start` — [`L322`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L322)
  - `question_end` — [`L321`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L321)
  - `question_start` — [`L320`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L320)
  - `system_message` — [`L317`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L317)
- uses (calls/refs, reference-scoped): [`ZeroShotDeepSeekQwenR1CoTBoxed`](evaluation_lib.md#ZeroShotDeepSeekQwenR1CoTBoxed)
- used by: [`test_zero_shot_system_cot_boxed`](evaluation_lib_test.md#EvaluationLibTest.test_zero_shot_system_cot_boxed)

## Module values
- `GSM8K_8_SHOTS_TXT` — [`L41`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L41)
- `extract_boxed_answer` — [`L37`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L37)
- `find_number` — [`L36`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L36)
- `match` — [`L38`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L38)
- `maybe_remove_comma` — [`L35`](../../../../../../raw/code/simply/simply/utils/evaluation_lib.py#L35)

