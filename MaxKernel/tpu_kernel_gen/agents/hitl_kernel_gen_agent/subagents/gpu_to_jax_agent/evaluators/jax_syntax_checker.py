"""JAX syntax validation evaluator."""

import ast
import logging
from typing import AsyncGenerator, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions


class JaxSyntaxChecker(BaseAgent):
  """Validates JAX code syntax and checks for common conversion errors."""

  input_key: Optional[str] = None
  output_key: Optional[str] = None

  def __init__(self, name: str, input_key: str, output_key: str):
    super().__init__(name=name)
    self.input_key = input_key
    self.output_key = output_key

  async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
    code = ctx.session.state.get(self.input_key, "")
    if not code:
      logging.warning(f"[{self.name}] No {self.input_key} found in context")
      yield Event(
        author=self.name,
        actions=EventActions(state_delta={self.output_key: "No code to validate"}),
      )
      return

    errors = []
    warnings = []

    # Check 1: Basic Python syntax
    try:
      ast.parse(code)
      logging.info(f"[{self.name}] Python syntax is valid")
    except SyntaxError as e:
      error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
      logging.error(f"[{self.name}] {error_msg}")
      errors.append(error_msg)
      yield Event(
        author=self.name,
        actions=EventActions(state_delta={self.output_key: error_msg}),
      )
      return

    # Check 2: Required JAX imports
    if "import jax" not in code and "from jax" not in code:
      errors.append("Missing 'import jax' statement")

    if "jax.numpy" not in code and "jnp" not in code:
      warnings.append("No JAX numpy import detected (import jax.numpy as jnp)")

    # Check 3: Check for GPU-specific remnants
    gpu_remnants = []
    if ".cuda()" in code:
      gpu_remnants.append(".cuda() call found (should be removed in JAX)")
    if ".to(device)" in code or ".to('cuda')" in code:
      gpu_remnants.append(".to(device) call found (should be removed in JAX)")
    if "torch.Tensor" in code:
      gpu_remnants.append("torch.Tensor reference found (should be jnp.ndarray)")
    if "<<<" in code or ">>>" in code:
      gpu_remnants.append("CUDA kernel launch syntax found (<<< >>>)")
    if "__global__" in code or "__device__" in code:
      gpu_remnants.append("CUDA kernel decorators found (__global__, __device__)")
    if "@triton.jit" in code:
      gpu_remnants.append("Triton decorator found (@triton.jit)")

    if gpu_remnants:
      errors.extend(gpu_remnants)

    # Check 4: Verify JAX idioms
    if "random.normal" in code or "random.uniform" in code:
      if "random.PRNGKey" not in code and "PRNGKey" not in code:
        warnings.append("Using JAX random functions but no PRNGKey initialization found")

    # Check 5: Common API mismatches
    if " dim=" in code:
      warnings.append("Found 'dim=' parameter - JAX uses 'axis=' instead (PyTorch uses 'dim=')")

    if ".numpy()" in code or ".item()" in code:
      errors.append("Found .numpy() or .item() method call - JAX arrays don't have these methods")

    # Check 6: Verify three-section structure
    required_sections = ["# Imports", "# Initialization", "# Computation"]
    for section in required_sections:
      if section not in code:
        warnings.append(f"Missing expected section: {section}")

    # Build result message
    if errors:
      error_msg = "JAX Syntax Validation Failed:\n"
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
      success_msg = "JAX Syntax Validation Passed"
      if warnings:
        success_msg += "\n\nWarnings:\n"
        for i, warning in enumerate(warnings, 1):
          success_msg += f"  {i}. {warning}\n"

      logging.info(f"[{self.name}] {success_msg}")
      yield Event(
        author=self.name,
        actions=EventActions(state_delta={self.output_key: success_msg}),
      )
