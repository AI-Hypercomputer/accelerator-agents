"""Pre-flight validation tests for kernel task YAML specs and VMEM limits."""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import yaml

EVALUATION_DIR = Path(__file__).resolve().parents[2] / "evaluation"

DEFAULT_MIN_VMEM_LIMIT_BYTES = (
  67_108_864  # 64 MiB for complex attention kernels
)
MIN_VMEM_LIMIT_BYTES = DEFAULT_MIN_VMEM_LIMIT_BYTES  # Backward compatibility

# Architecture-aware minimum VMEM limit mappings (in bytes).
# Scoped vector memory (VMEM) capacity is required to prevent compile-time
# scoped memory OOMs and avoid excessive spill loops for complex attention kernels.
# Memory capacities vary across TPU accelerator generations:
# - v4:  16 MiB per-core default scratchpad
# - v5p: 32 MiB per-core default scratchpad / 64 MiB expanded capacity
# - v6e: 64 MiB for large tile configurations (physical ceiling up to 128 MiB)
# - default: 64 MiB baseline for complex attention benchmarks
TPU_ARCH_MIN_VMEM_LIMIT_BYTES: Dict[str, int] = {
  "v4": 16 * 1024 * 1024,
  "v5p": 32 * 1024 * 1024,
  "v6e": 64 * 1024 * 1024,
  "default": DEFAULT_MIN_VMEM_LIMIT_BYTES,
}


def get_min_vmem_limit_bytes(arch: str = "default") -> int:
  """Retrieve the minimum VMEM allocation threshold for a given accelerator architecture."""
  return TPU_ARCH_MIN_VMEM_LIMIT_BYTES.get(arch, DEFAULT_MIN_VMEM_LIMIT_BYTES)


def discover_kernel_task_yamls() -> List[Path]:
  """Discover all kernel_task.yaml files under the evaluation directory."""
  return sorted(EVALUATION_DIR.glob("**/kernel_task.yaml"))


def discover_reference_files() -> List[Path]:
  """Discover all reference.py files under the evaluation directory."""
  return sorted(EVALUATION_DIR.glob("**/reference.py"))


def _eval_const_expr(
  node: ast.AST, env: Optional[Dict[str, Any]] = None
) -> Optional[int | float]:
  """Evaluate constant arithmetic expressions in AST.

  Design Rationale (AST vs. Runtime Sandbox):
  Static AST evaluation is intentionally selected over sandboxed dynamic execution
  (e.g., executing reference.py in an isolated subprocess). Dynamic execution would
  require JAX, Pallas, and hardware accelerator runtimes/drivers (e.g., libtpu), which
  are unavailable on lightweight CI runners. AST parsing is completely hermetic,
  zero-dependency, safe from arbitrary code execution side effects, and executes
  across the entire benchmark suite in under a second without hardware dependencies.
  """
  if env is None:
    env = {}
  if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
    return node.value
  if isinstance(node, ast.Name) and node.id in env:
    return env[node.id]
  if isinstance(node, ast.UnaryOp):
    operand = _eval_const_expr(node.operand, env)
    if operand is not None:
      if isinstance(node.op, ast.USub):
        return -operand
      if isinstance(node.op, ast.UAdd):
        return operand
  if isinstance(node, ast.BinOp):
    left = _eval_const_expr(node.left, env)
    right = _eval_const_expr(node.right, env)
    if left is not None and right is not None:
      if isinstance(node.op, ast.Mult):
        return left * right
      if isinstance(node.op, ast.Add):
        return left + right
      if isinstance(node.op, ast.Sub):
        return left - right
      if isinstance(node.op, ast.FloorDiv):
        return left // right
      if isinstance(node.op, ast.Div):
        return left / right
      if isinstance(node.op, ast.Pow):
        return left**right
  return None


def _extract_vmem_limits_from_file(
  file_path: Path,
) -> List[Tuple[int, int | float]]:
  """Extract numeric vmem_limit_bytes values and line numbers from reference.py."""
  with open(file_path, "r", encoding="utf-8") as f:
    source = f.read()

  if "vmem_limit_bytes" not in source:
    return []

  tree = ast.parse(source, filename=str(file_path))
  env: Dict[str, Any] = {}

  # Collect module-level constant assignments first
  for stmt in tree.body:
    if isinstance(stmt, ast.Assign):
      val = _eval_const_expr(stmt.value, env)
      if val is not None:
        for target in stmt.targets:
          if isinstance(target, ast.Name):
            env[target.id] = val
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
      val = _eval_const_expr(stmt.value, env)
      if val is not None and isinstance(stmt.target, ast.Name):
        env[stmt.target.id] = val

  limits: List[Tuple[int, int | float]] = []
  for node in ast.walk(tree):
    # Variable assignments: vmem_limit_bytes = ...
    if isinstance(node, ast.Assign):
      for target in node.targets:
        if isinstance(target, ast.Name) and target.id == "vmem_limit_bytes":
          val = _eval_const_expr(node.value, env)
          if val is not None:
            limits.append((node.lineno, val))
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
      if (
        isinstance(node.target, ast.Name)
        and node.target.id == "vmem_limit_bytes"
      ):
        val = _eval_const_expr(node.value, env)
        if val is not None:
          limits.append((node.lineno, val))
    # Dictionary values: {'vmem_limit_bytes': ...}
    elif isinstance(node, ast.Dict):
      for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and key.value == "vmem_limit_bytes":
          val = _eval_const_expr(value, env)
          if val is not None:
            limits.append((node.lineno, val))
    # Call arguments: func(vmem_limit_bytes=...)
    elif isinstance(node, ast.Call):
      for kw in node.keywords:
        if kw.arg == "vmem_limit_bytes":
          val = _eval_const_expr(kw.value, env)
          if val is not None:
            limits.append((node.lineno, val))

  return limits


@pytest.mark.parametrize(
  "yaml_path",
  discover_kernel_task_yamls(),
  ids=lambda p: str(p.relative_to(EVALUATION_DIR)),
)
def test_kernel_task_yaml_structure_and_specs(yaml_path: Path):
  """Validate task_id, tolerance fields, and input generation code."""
  with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

  assert isinstance(data, dict), (
    f"File {yaml_path} must be a valid YAML dictionary, got {type(data).__name__}."
  )

  required_keys = ("task_id", "input_gen_code", "atol", "rtol")
  for key in required_keys:
    assert key in data, f"File {yaml_path} is missing required key '{key}'."

  assert isinstance(data["atol"], (int, float)), (
    f"File {yaml_path}: atol must be a numeric float/int, got"
    f" {type(data['atol']).__name__}."
  )
  assert data["atol"] >= 0, (
    f"File {yaml_path}: atol must be non-negative, got {data['atol']}."
  )

  assert isinstance(data["rtol"], (int, float)), (
    f"File {yaml_path}: rtol must be a numeric float/int, got"
    f" {type(data['rtol']).__name__}."
  )
  assert data["rtol"] >= 0, (
    f"File {yaml_path}: rtol must be non-negative, got {data['rtol']}."
  )

  input_code = data["input_gen_code"]
  assert isinstance(input_code, str) and input_code.strip(), (
    f"File {yaml_path}: input_gen_code must be a non-empty string."
  )

  try:
    tree = ast.parse(input_code, filename=str(yaml_path))
  except SyntaxError as e:
    pytest.fail(f"File {yaml_path}: input_gen_code failed to parse: {e}")

  defined_functions = {
    node.name
    for node in ast.walk(tree)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
  }
  assert "get_inputs" in defined_functions, (
    f"File {yaml_path}: input_gen_code must define a 'get_inputs' function."
  )


@pytest.mark.parametrize(
  "reference_path",
  discover_reference_files(),
  ids=lambda p: str(p.relative_to(EVALUATION_DIR)),
)
def test_reference_vmem_limit_bounds(reference_path: Path):
  """Assert that explicit vmem_limit_bytes in reference.py are >= minimum thresholds."""
  vmem_limits = _extract_vmem_limits_from_file(reference_path)
  min_limit = get_min_vmem_limit_bytes("default")
  for lineno, value in vmem_limits:
    assert value >= min_limit, (
      f"{reference_path}:{lineno} specifies vmem_limit_bytes={value} (<"
      f" {min_limit} bytes / {min_limit // (1024 * 1024)} MiB). This risks compile-time"
      " scoped VMEM OOMs."
    )


def test_arch_aware_vmem_limits():
  """Validate architecture-aware VMEM mappings and fallback behavior."""
  assert TPU_ARCH_MIN_VMEM_LIMIT_BYTES["v4"] == 16 * 1024 * 1024
  assert TPU_ARCH_MIN_VMEM_LIMIT_BYTES["v5p"] == 32 * 1024 * 1024
  assert TPU_ARCH_MIN_VMEM_LIMIT_BYTES["v6e"] == 64 * 1024 * 1024
  assert (
    TPU_ARCH_MIN_VMEM_LIMIT_BYTES["default"] == DEFAULT_MIN_VMEM_LIMIT_BYTES
  )
  assert get_min_vmem_limit_bytes("v4") == 16 * 1024 * 1024
  assert get_min_vmem_limit_bytes("v5p") == 32 * 1024 * 1024
  assert get_min_vmem_limit_bytes("v6e") == 64 * 1024 * 1024
  assert get_min_vmem_limit_bytes("default") == DEFAULT_MIN_VMEM_LIMIT_BYTES
  assert get_min_vmem_limit_bytes("unknown") == DEFAULT_MIN_VMEM_LIMIT_BYTES


def test_eval_const_expr_arithmetic_and_unary():
  """Test constant expression evaluation with various operators."""
  assert _eval_const_expr(ast.parse("10 / 2", mode="eval").body) == 5.0
  assert _eval_const_expr(ast.parse("2 ** 3", mode="eval").body) == 8
  assert _eval_const_expr(ast.parse("7 // 2", mode="eval").body) == 3
  assert _eval_const_expr(ast.parse("2 * 3", mode="eval").body) == 6
  assert _eval_const_expr(ast.parse("1 + 2", mode="eval").body) == 3
  assert _eval_const_expr(ast.parse("5 - 2", mode="eval").body) == 3
  assert _eval_const_expr(ast.parse("-5", mode="eval").body) == -5
  assert _eval_const_expr(ast.parse("+5", mode="eval").body) == 5
  assert _eval_const_expr(ast.parse("-(2 ** 3)", mode="eval").body) == -8


def test_extract_vmem_limits_annotated_assignment(tmp_path: Path):
  """Test extracting vmem limits with type annotations and constants."""
  code = """
VMEM_CAP: int = 128 * 1024 * 1024
vmem_limit_bytes: int = VMEM_CAP // 2
"""
  test_file = tmp_path / "reference.py"
  test_file.write_text(code, encoding="utf-8")
  limits = _extract_vmem_limits_from_file(test_file)
  assert len(limits) == 1
  assert limits[0][1] == 64 * 1024 * 1024
