---
title: 'Module: simply/utils/math_eval.py'
type: catalog
provenance: extracted
module: simply/utils/math_eval.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.math_eval`/
symbols:
  _normalize: _normalize().
  grade_answer_sympy: grade_answer_sympy().
  _strip_string: _strip_string().
  match: match().
  should_allow_eval: should_allow_eval().
  extract_boxed_answer: extract_boxed_answer().
  split_tuple: split_tuple().
  find_number: find_number().
  mathd_normalize_answer: mathd_normalize_answer().
  _str_is_int: _str_is_int().
  are_equal_under_sympy: are_equal_under_sympy().
  grade_answer_mathd: grade_answer_mathd().
  _str_to_int: _str_to_int().
  find_numbers: find_numbers().
  maybe_remove_comma: maybe_remove_comma().
  last_boxed_only_string: last_boxed_only_string().
  remove_boxed: remove_boxed().
  TUPLE_CHARS: TUPLE_CHARS.
  _is_frac: _is_frac().
  _strip_properly_formatted_commas: _strip_properly_formatted_commas().
  count_unknown_letters_in_expr: count_unknown_letters_in_expr().
  _strip_string._fix_fracs: _strip_string()._fix_fracs().
  _strip_string._fix_a_slash_b: _strip_string()._fix_a_slash_b().
  _strip_string._remove_right_units: _strip_string()._remove_right_units().
  _strip_string._fix_sqrt: _strip_string()._fix_sqrt().
  BAD_SUBSTRINGS: BAD_SUBSTRINGS.
  BAD_REGEXES: BAD_REGEXES.
  _sympy_parse: _sympy_parse().
  _parse_latex: _parse_latex().
  _is_float: _is_float().
  _is_int: _is_int().
  _inject_implicit_mixed_number: _inject_implicit_mixed_number().
---
# Module: [`simply/utils/math_eval.py`](../../../../../../raw/code/simply/simply/utils/math_eval.py)

## Functions
- `_fix_a_slash_b(string: str)` — [`L186`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L186)
- `_fix_fracs(string: str)` — [`L155`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L155)
- `_fix_sqrt(string: str)` — [`L209`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L209)
- `_inject_implicit_mixed_number(step: str)` — [`L365`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L365) — Automatically make a mixed number evalable, like 7 3/4 => 7+3/4.
- `_is_float(num: str)` — [`L325`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L325)
- `_is_frac(expr: str)` — [`L340`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L340)
- `_is_int(x: float)` — [`L333`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L333)
- `_normalize(expr: str)` — [`L383`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L383) — Normalize answer expressions.
- `_parse_latex(expr: str)` — [`L306`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L306) — Attempts to parse latex to an expression sympy can read.
- `_remove_right_units(string: str)` — [`L200`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L200)
- `_str_is_int(x: str)` — [`L344`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L344)
- `_str_to_int(x: str)` — [`L353`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L353)
- `_strip_properly_formatted_commas(expr: str)` — [`L372`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L372)
- `_strip_string(string: str)` — [`L152`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L152) — Strips string.
- `_sympy_parse(expr: str)` — [`L293`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L293) — Parses an expression with sympy.
- `are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str)` — [`L479`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L479) — Check equality using sympy.
- `count_unknown_letters_in_expr(expr: str)` — [`L456`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L456)
- `extract_boxed_answer(solution: str)` — [`L96`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L96) — Extract the answer from inside a LaTeX \\boxed{} command.
- `find_number(x: str, answer_delimiter: str = 'The answer is')` — [`L32`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L32) — Finds the most relevant number in a string.
- `find_numbers(x: str)` — [`L21`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L21) — Finds all numbers in a string.
- `grade_answer_mathd(given_answer: str, ground_truth: str)` — [`L129`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L129) — Grade answer using mathematical correctness.
- `grade_answer_sympy(given_answer: str, ground_truth: str)` — [`L512`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L512) — Grade the answer using sympy.
- `last_boxed_only_string(string: str)` — [`L56`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L56) — find the last \\boxed{} or \\fbox{} part in a string.
- `match(answer: str, ground_truth: str)` — [`L106`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L106) — Matches answer against the ground-truth.
- `mathd_normalize_answer(answer: str | None)` — [`L138`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L138)
- `maybe_remove_comma(x: str)` — [`L49`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L49)
- `remove_boxed(s: str)` — [`L85`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L85) — Remove the \\boxed{} command from a string.
- `should_allow_eval(expr: str)` — [`L463`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L463) — Whether we should try to eval in the first place: we don't want to try parsing unknown text or functions of more than two variables.
- `split_tuple(expr: str)` — [`L495`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L495) — Split the elements in a tuple/interval, while handling well-formatted commas in large numbers.

## Module values
- `BAD_REGEXES` — [`L289`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L289)
- `BAD_SUBSTRINGS` — [`L288`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L288)
- `TUPLE_CHARS` — [`L290`](../../../../../../raw/code/simply/simply/utils/math_eval.py#L290)

