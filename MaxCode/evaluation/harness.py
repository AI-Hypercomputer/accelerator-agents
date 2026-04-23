"""Evaluation harness for agents."""

import ast
# This includes:
# - Defining dataset formats for single-file, model, and repo agents.
# - Implementing run_evaluation function/class to execute agents on datasets.
# - Defining metrics (e.g., syntax correctness, BLEU score vs reference,
#   unit test pass rate, lint checks).
# - Integrating with different agent types.


def check_syntax(code: str) -> bool:
  """Checks if the given Python code has valid syntax."""
  try:
    ast.parse(code)
    return True
  except SyntaxError:
    return False
