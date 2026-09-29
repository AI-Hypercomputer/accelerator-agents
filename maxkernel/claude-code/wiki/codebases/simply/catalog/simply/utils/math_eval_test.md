---
title: 'Module: simply/utils/math_eval_test.py'
type: catalog
provenance: extracted
module: simply/utils/math_eval_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.math_eval_test`/
symbols:
  FindNumbersTest.test_find_numbers: FindNumbersTest#test_find_numbers().
  FindNumbersTest.test_find_number_prefers_answer_delimiter: FindNumbersTest#test_find_number_prefers_answer_delimiter().
  FindNumbersTest.test_find_number_falls_back_to_last_number: FindNumbersTest#test_find_number_falls_back_to_last_number().
  FindNumbersTest.test_find_number_empty_when_no_numbers: FindNumbersTest#test_find_number_empty_when_no_numbers().
  FindNumbersTest.test_maybe_remove_comma: FindNumbersTest#test_maybe_remove_comma().
  BoxedAnswerTest.test_last_boxed_only_string: BoxedAnswerTest#test_last_boxed_only_string().
  BoxedAnswerTest.test_last_boxed_only_string_returns_none_when_absent: BoxedAnswerTest#test_last_boxed_only_string_returns_none_when_absent().
  BoxedAnswerTest.test_remove_boxed: BoxedAnswerTest#test_remove_boxed().
  BoxedAnswerTest.test_remove_boxed_returns_none_on_bad_input: BoxedAnswerTest#test_remove_boxed_returns_none_on_bad_input().
  BoxedAnswerTest.test_extract_boxed_answer: BoxedAnswerTest#test_extract_boxed_answer().
  BoxedAnswerTest.test_extract_boxed_answer_returns_none_when_absent: BoxedAnswerTest#test_extract_boxed_answer_returns_none_when_absent().
  StrToIntTest.test_str_to_int: StrToIntTest#test_str_to_int().
  StrToIntTest.test_preserves_precision_above_2_pow_53: StrToIntTest#test_preserves_precision_above_2_pow_53().
  StrToIntTest.test_preserves_precision_very_large_integer: StrToIntTest#test_preserves_precision_very_large_integer().
  StrIsIntTest.test_str_is_int: StrIsIntTest#test_str_is_int().
  IsFracTest.test_is_frac: IsFracTest#test_is_frac().
  StripPropertyFormattedCommasTest.test_strip_properly_formatted_commas: StripPropertyFormattedCommasTest#test_strip_properly_formatted_commas().
  CountUnknownLettersTest.test_count_unknown_letters_in_expr: CountUnknownLettersTest#test_count_unknown_letters_in_expr().
  ShouldAllowEvalTest.test_should_allow_eval: ShouldAllowEvalTest#test_should_allow_eval().
  SplitTupleTest.test_split_tuple: SplitTupleTest#test_split_tuple().
  NormalizeTest.test_normalize: NormalizeTest#test_normalize().
  MathdNormalizeTest.test_none_returns_none: MathdNormalizeTest#test_none_returns_none().
  MathdNormalizeTest.test_strips_enclosing_text: MathdNormalizeTest#test_strips_enclosing_text().
  MathdNormalizeTest.test_grade_answer_mathd_exact_match: MathdNormalizeTest#test_grade_answer_mathd_exact_match().
  MathdNormalizeTest.test_grade_answer_mathd_mismatch: MathdNormalizeTest#test_grade_answer_mathd_mismatch().
  MatchTest.test_match: MatchTest#test_match().
  FindNumbersTest: FindNumbersTest#
  BoxedAnswerTest: BoxedAnswerTest#
  StrToIntTest: StrToIntTest#
  StrIsIntTest: StrIsIntTest#
  IsFracTest: IsFracTest#
  StripPropertyFormattedCommasTest: StripPropertyFormattedCommasTest#
  CountUnknownLettersTest: CountUnknownLettersTest#
  ShouldAllowEvalTest: ShouldAllowEvalTest#
  SplitTupleTest: SplitTupleTest#
  NormalizeTest: NormalizeTest#
  MathdNormalizeTest: MathdNormalizeTest#
  MatchTest: MatchTest#
  MatchTest.setUpClass: MatchTest#setUpClass().
---
# Module: [`simply/utils/math_eval_test.py`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py)

## Classes
### `BoxedAnswerTest`  ·  implements/extends TestCase
- def: [`simply/utils/math_eval_test.py:50`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L50)
- signature: `class BoxedAnswerTest(parameterized.TestCase):`
- members:
  - `test_extract_boxed_answer(self)` — [`L74`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L74)
  - `test_extract_boxed_answer_returns_none_when_absent(self)` — [`L79`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L79)
  - `test_last_boxed_only_string(self, text, expected)` — [`L58`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L58)
  - `test_last_boxed_only_string_returns_none_when_absent(self)` — [`L61`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L61)
  - `test_remove_boxed(self, text, expected)` — [`L68`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L68)
  - `test_remove_boxed_returns_none_on_bad_input(self)` — [`L71`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L71)
- uses (calls/refs, reference-scoped): [`extract_boxed_answer`](math_eval.md#extract_boxed_answer), [`last_boxed_only_string`](math_eval.md#last_boxed_only_string), [`remove_boxed`](math_eval.md#remove_boxed)

### `CountUnknownLettersTest`  ·  implements/extends TestCase
- def: [`simply/utils/math_eval_test.py:147`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L147)
- signature: `class CountUnknownLettersTest(parameterized.TestCase):`
- members:
  - `test_count_unknown_letters_in_expr(self, expr, expected)` — [`L157`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L157)
- uses (calls/refs, reference-scoped): [`count_unknown_letters_in_expr`](math_eval.md#count_unknown_letters_in_expr)

### `FindNumbersTest`  ·  implements/extends TestCase
- def: [`simply/utils/math_eval_test.py:21`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L21)
- signature: `class FindNumbersTest(parameterized.TestCase):`
- members:
  - `test_find_number_empty_when_no_numbers(self)` — [`L42`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L42)
  - `test_find_number_falls_back_to_last_number(self)` — [`L38`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L38)
  - `test_find_number_prefers_answer_delimiter(self)` — [`L34`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L34)
  - `test_find_numbers(self, text, expected)` — [`L31`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L31)
  - `test_maybe_remove_comma(self)` — [`L45`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L45)
- uses (calls/refs, reference-scoped): [`find_number`](math_eval.md#find_number), [`find_numbers`](math_eval.md#find_numbers), [`maybe_remove_comma`](math_eval.md#maybe_remove_comma)

### `IsFracTest`  ·  implements/extends TestCase
- def: [`simply/utils/math_eval_test.py:121`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L121)
- signature: `class IsFracTest(parameterized.TestCase):`
- members:
  - `test_is_frac(self, text, expected)` — [`L129`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L129)
- uses (calls/refs, reference-scoped): [`_is_frac`](math_eval.md#_is_frac)

### `MatchTest`  ·  implements/extends TestCase
- def: [`simply/utils/math_eval_test.py:213`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L213)
- doc: End-to-end `match`. These exercise the sympy path; skip if unavailable.
- signature: `class MatchTest(parameterized.TestCase):`
- members:
  - `setUpClass(cls)` — [`L217`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L217)
  - `test_match(self, answer, ground_truth, expected)` — [`L230`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L230)
- uses (calls/refs, reference-scoped): [`match`](math_eval.md#match)

### `MathdNormalizeTest`  ·  implements/extends TestCase
- def: [`simply/utils/math_eval_test.py:198`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L198)
- signature: `class MathdNormalizeTest(parameterized.TestCase):`
- members:
  - `test_grade_answer_mathd_exact_match(self)` — [`L206`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L206)
  - `test_grade_answer_mathd_mismatch(self)` — [`L209`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L209)
  - `test_none_returns_none(self)` — [`L200`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L200)
  - `test_strips_enclosing_text(self)` — [`L203`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L203)
- uses (calls/refs, reference-scoped): [`mathd_normalize_answer`](math_eval.md#mathd_normalize_answer), [`grade_answer_mathd`](math_eval.md#grade_answer_mathd)

### `NormalizeTest`  ·  implements/extends TestCase
- def: [`simply/utils/math_eval_test.py:186`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L186)
- signature: `class NormalizeTest(parameterized.TestCase):`
- members:
  - `test_normalize(self, expr, expected)` — [`L194`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L194)
- uses (calls/refs, reference-scoped): [`_normalize`](math_eval.md#_normalize)

### `ShouldAllowEvalTest`  ·  implements/extends TestCase
- def: [`simply/utils/math_eval_test.py:161`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L161)
- signature: `class ShouldAllowEvalTest(parameterized.TestCase):`
- members:
  - `test_should_allow_eval(self, expr, expected)` — [`L170`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L170)
- uses (calls/refs, reference-scoped): [`should_allow_eval`](math_eval.md#should_allow_eval)

### `SplitTupleTest`  ·  implements/extends TestCase
- def: [`simply/utils/math_eval_test.py:174`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L174)
- signature: `class SplitTupleTest(parameterized.TestCase):`
- members:
  - `test_split_tuple(self, expr, expected)` — [`L182`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L182)
- uses (calls/refs, reference-scoped): [`split_tuple`](math_eval.md#split_tuple)

### `StrIsIntTest`  ·  implements/extends TestCase
- def: [`simply/utils/math_eval_test.py:108`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L108)
- signature: `class StrIsIntTest(parameterized.TestCase):`
- members:
  - `test_str_is_int(self, text, expected)` — [`L117`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L117)
- uses (calls/refs, reference-scoped): [`_str_is_int`](math_eval.md#_str_is_int)

### `StrToIntTest`  ·  implements/extends TestCase
- def: [`simply/utils/math_eval_test.py:83`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L83)
- doc: Tests for `_str_to_int`, including the &gt;2^53 precision regression.
- signature: `class StrToIntTest(parameterized.TestCase):`
- members:
  - `test_preserves_precision_above_2_pow_53(self)` — [`L97`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L97)
  - `test_preserves_precision_very_large_integer(self)` — [`L103`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L103)
  - `test_str_to_int(self, text, expected)` — [`L94`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L94)
- uses (calls/refs, reference-scoped): [`_str_to_int`](math_eval.md#_str_to_int)

### `StripPropertyFormattedCommasTest`  ·  implements/extends TestCase
- def: [`simply/utils/math_eval_test.py:133`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L133)
- signature: `class StripPropertyFormattedCommasTest(parameterized.TestCase):`
- members:
  - `test_strip_properly_formatted_commas(self, text, expected)` — [`L141`](../../../../../../raw/code/simply/simply/utils/math_eval_test.py#L141)
- uses (calls/refs, reference-scoped): [`_strip_properly_formatted_commas`](math_eval.md#_strip_properly_formatted_commas)

