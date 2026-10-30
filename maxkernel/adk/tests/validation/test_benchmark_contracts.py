"""Contract validation tests for adapted benchmark dataset problems."""

import ast
from pathlib import Path
from typing import List

import pytest

EVALUATION_DIR = Path(__file__).resolve().parents[2] / "evaluation"
DATASET_DIR = EVALUATION_DIR / "jaxbench_adapted_dataset"


def discover_benchmark_problem_dirs() -> List[Path]:
  """Discover all problem directories under level1 and level2 datasets."""
  problem_dirs: List[Path] = []
  for level in ("level1", "level2"):
    level_dir = DATASET_DIR / level
    if not level_dir.exists():
      continue
    for child in sorted(level_dir.iterdir()):
      if child.is_dir():
        problem_dirs.append(child)
  return problem_dirs


@pytest.mark.parametrize(
  "problem_dir",
  discover_benchmark_problem_dirs(),
  ids=lambda p: str(p.relative_to(DATASET_DIR)),
)
def test_benchmark_problem_contract(problem_dir: Path):
  """Verify that each benchmark problem provides valid specs and reference functions."""
  kernel_task_path = problem_dir / "kernel_task.yaml"
  reference_path = problem_dir / "reference.py"

  assert kernel_task_path.is_file(), (
    f"Problem directory {problem_dir} is missing required kernel_task.yaml"
  )
  assert reference_path.is_file(), (
    f"Problem directory {problem_dir} is missing required reference.py"
  )

  with open(reference_path, "r", encoding="utf-8") as f:
    source = f.read()

  try:
    tree = ast.parse(source, filename=str(reference_path))
  except SyntaxError as e:
    pytest.fail(f"File {reference_path} failed to parse: {e}")

  top_level_functions = {
    stmt.name
    for stmt in tree.body
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
  }

  assert "computation" in top_level_functions, (
    f"File {reference_path} must define a top-level 'computation' function,"
    f" found: {sorted(top_level_functions)}"
  )
  assert "get_inputs" in top_level_functions, (
    f"File {reference_path} must define a top-level 'get_inputs' function,"
    f" found: {sorted(top_level_functions)}"
  )
