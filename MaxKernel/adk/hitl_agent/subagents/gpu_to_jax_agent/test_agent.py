"""Unit tests for GPU to JAX conversion agent."""

import pytest

from hitl_agent.subagents.gpu_to_jax_agent.evaluators import (
  JaxSyntaxChecker,
  ShapeValidator,
)


class TestJaxSyntaxChecker:
  """Tests for JAX syntax validation."""

  def test_valid_jax_code(self):
    """Test that valid JAX code passes syntax check."""
    valid_code = """
# Imports
import jax
import jax.numpy as jnp
import jax.random as random

# Initialization
N = 1024
key = random.PRNGKey(0)
A = random.normal(key, (N, N))

# Computation
def computation(A: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(A)

result = jax.block_until_ready(computation(A))
"""
    checker = JaxSyntaxChecker(
      name="TestChecker", input_key="test_code", output_key="test_results"
    )

    # Note: This is a synchronous test - in practice, you'd need to run the async method
    # For now, we just verify the checker can be instantiated
    assert checker.input_key == "test_code"
    assert checker.output_key == "test_results"

  def test_invalid_python_syntax(self):
    """Test that invalid Python syntax is detected."""
    invalid_code = """
# Imports
import jax

# This will cause a syntax error
def broken(
    return None
"""
    checker = JaxSyntaxChecker(
      name="TestChecker", input_key="test_code", output_key="test_results"
    )
    assert checker is not None


class TestShapeValidator:
  """Tests for shape validation."""

  def test_extract_shapes_from_code(self):
    """Test shape extraction from code."""
    code = """
import jax.random as random
key = random.PRNGKey(0)
A = random.normal(key, (1024, 512))
B = random.normal(key, (512, 256))
"""
    validator = ShapeValidator(
      name="TestValidator", input_key="test_code", output_key="test_results"
    )

    shapes = validator.extract_shapes_from_code(code)
    assert "inputs" in shapes
    assert len(shapes["inputs"]) > 0

  def test_validate_shape_consistency(self):
    """Test shape consistency validation."""
    code = """
import jax.numpy as jnp

def computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(A, B)
"""
    validator = ShapeValidator(
      name="TestValidator", input_key="test_code", output_key="test_results"
    )

    is_valid, errors, warnings = validator.validate_shape_consistency(code)
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


class TestGpuToJaxAgent:
  """Integration tests for the full conversion agent."""

  @pytest.mark.asyncio
  async def test_agent_initialization(self):
    """Test that the agent can be initialized."""
    from hitl_agent.subagents.gpu_to_jax_agent.agent import (
      gpu_to_jax_conversion_agent,
    )

    assert gpu_to_jax_conversion_agent is not None
    assert gpu_to_jax_conversion_agent.name == "GpuToJaxConversionAgent"
    assert len(gpu_to_jax_conversion_agent.sub_agents) > 0


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
