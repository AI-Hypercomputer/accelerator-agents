"""Shape validation evaluator for JAX conversions."""

import ast
import logging
import re
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions


class ShapeValidator(BaseAgent):
  """Validates that input/output shapes are consistent in converted JAX code."""

  input_key: Optional[str] = None
  output_key: Optional[str] = None

  def __init__(self, name: str, input_key: str, output_key: str):
    super().__init__(name=name)
    self.input_key = input_key
    self.output_key = output_key

  def extract_shapes_from_code(self, code: str) -> Dict[str, List[Tuple]]:
    """Extract shape information from code by analyzing array initializations."""
    shapes = {"inputs": [], "outputs": []}

    # Pattern 1: Shape from random/zeros/ones calls: (N, M, K)
    shape_patterns = [
        r"random\.normal\([^,]+,\s*\(([^)]+)\)\)",
        r"random\.uniform\([^,]+,\s*\(([^)]+)\)\)",
        r"jnp\.zeros\(\(([^)]+)\)\)",
        r"jnp\.ones\(\(([^)]+)\)\)",
        r"np\.zeros\(\(([^)]+)\)\)",
        r"np\.ones\(\(([^)]+)\)\)",
        r"torch\.randn\(\(([^)]+)\)\)",
        r"torch\.zeros\(\(([^)]+)\)\)",
    ]

    for pattern in shape_patterns:
      matches = re.finditer(pattern, code)
      for match in matches:
        shape_str = match.group(1)
        # Try to evaluate as tuple of dimensions
        try:
          # Clean up shape string and convert to tuple
          dims = [d.strip() for d in shape_str.split(",")]
          shapes["inputs"].append(tuple(dims))
        except:
          pass

    # Pattern 2: Explicit shape comments or annotations
    shape_comment_pattern = r"#.*shape[:\s]+\(([^)]+)\)"
    matches = re.finditer(shape_comment_pattern, code, re.IGNORECASE)
    for match in matches:
      try:
        shape_str = match.group(1)
        dims = [d.strip() for d in shape_str.split(",")]
        shapes["inputs"].append(tuple(dims))
      except:
        pass

    return shapes

  def validate_shape_consistency(
      self, code: str) -> Tuple[bool, List[str], List[str]]:
    """Validate shape consistency in the code."""
    errors = []
    warnings = []

    shapes = self.extract_shapes_from_code(code)

    # Check if we found any shape information
    if not shapes["inputs"]:
      warnings.append(
          "Could not extract shape information from code. Manual verification recommended."
      )
      return True, errors, warnings

    # Check for common shape-related issues
    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
      # Check for dimension mismatches in matmul/dot operations
      if "matmul" in line or "dot" in line:
        # Look for common mistakes like (N, M) @ (N, K) instead of (N, M) @ (M, K)
        if "@" in line:
          warnings.append(
              f"Line {i}: Found @ operator - verify matrix dimensions are compatible"
          )

      # Check for reshape operations that might change dimensions
      if "reshape" in line or "view" in line:
        warnings.append(
            f"Line {i}: Found reshape operation - verify shape transformation is correct"
        )

      # Check for reduce operations that change dimensions
      if any(op in line for op in ["sum(", "mean(", "max(", "min(", "reduce("]):
        if "axis=" not in line and "dim=" not in line:
          warnings.append(
              f"Line {i}: Found reduction operation without explicit axis - output shape may differ from expected"
          )

    # Verify computation function signature
    try:
      tree = ast.parse(code)
      for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "computation":
          # Check if function has type hints
          has_type_hints = any(
              arg.annotation is not None for arg in node.args.args)
          if not has_type_hints:
            warnings.append(
                "computation() function missing type hints - add jnp.ndarray annotations for clarity"
            )

          # Check return type
          if node.returns is None:
            warnings.append(
                "computation() function missing return type annotation")
    except:
      pass

    is_valid = len(errors) == 0
    return is_valid, errors, warnings

  async def _run_async_impl(
      self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    code = ctx.session.state.get(self.input_key, "")
    if not code:
      logging.warning(f"[{self.name}] No {self.input_key} found in context")
      yield Event(
          author=self.name,
          actions=EventActions(
              state_delta={self.output_key: "No code to validate"}),
      )
      return

    try:
      is_valid, errors, warnings = self.validate_shape_consistency(code)

      if not is_valid or errors:
        error_msg = "Shape Validation Failed:\n"
        for i, error in enumerate(errors, 1):
          error_msg += f"  {i}. {error}\n"
        if warnings:
          error_msg += "\nWarnings:\n"
          for i, warning in enumerate(warnings, 1):
            error_msg += f"  {i}. {warning}\n"

        logging.error(f"[{self.name}] {error_msg}")
        yield Event(
            author=self.name,
            actions=EventActions(state_delta={self.output_key: error_msg}),
        )
      else:
        success_msg = "Shape Validation Passed"
        shapes = self.extract_shapes_from_code(code)
        if shapes["inputs"]:
          success_msg += f"\n\nDetected input shapes: {shapes['inputs']}"
        if warnings:
          success_msg += "\n\nWarnings:\n"
          for i, warning in enumerate(warnings, 1):
            success_msg += f"  {i}. {warning}\n"

        logging.info(f"[{self.name}] {success_msg}")
        yield Event(
            author=self.name,
            actions=EventActions(state_delta={self.output_key: success_msg}),
        )

    except Exception as e:
      error_msg = f"Exception during shape validation: {str(e)}"
      logging.error(f"[{self.name}] {error_msg}")
      yield Event(
          author=self.name,
          actions=EventActions(state_delta={self.output_key: error_msg}),
      )
