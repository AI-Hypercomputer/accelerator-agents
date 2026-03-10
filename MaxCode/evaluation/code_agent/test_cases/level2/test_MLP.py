"""Compilability and shape checks for CreditScoringMLP."""

import os
import sys

from absl import app
# pylint: disable=g-importing-member
from evaluation.code_agent.pytorch_references.level2.MLP import CreditScoringMLP as CreditScoringMLP_Torch
from generated_code.code_agent.v1.Gemini_2_5_pro.CreditScoringMLP.model import CreditScoringMLP as CreditScoringMLP_Jax
import jax
from jax import config
import jax.numpy as jnp
import torch


os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_default_matmul_precision", "highest")

TEST_CONFIG = {
    "input_features": 20,
    "batch_size": 4,
}


def test_pytorch_independent():
  """Sanity check for PyTorch model execution."""
  # Instantiate model
  model = CreditScoringMLP_Torch(TEST_CONFIG["input_features"])
  model.eval()

  # Create dummy input
  x = torch.randn(TEST_CONFIG["batch_size"], TEST_CONFIG["input_features"])

  # Run inference
  y = model(x)

  # Checks
  expected_shape = (TEST_CONFIG["batch_size"], 1)
  assert (
      y.shape == expected_shape
  ), f"PyTorch Shape Mismatch: Expected {expected_shape}, got {y.shape}"
  assert not torch.isnan(y).any(), "PyTorch Output contains NaNs"


def test_jax_independent():
  """Sanity check for JAX model execution."""
  # Instantiate model
  model = CreditScoringMLP_Jax(TEST_CONFIG["input_features"])

  # Create dummy input
  x = jnp.ones((TEST_CONFIG["batch_size"], TEST_CONFIG["input_features"]))

  # Initialize variables (Note: supplying 'train' kwarg as requested)
  rng = jax.random.PRNGKey(42)
  variables = model.init(rng, x, train=False)

  # Run inference
  y = model.apply(variables, x, train=False)

  # Checks
  expected_shape = (TEST_CONFIG["batch_size"], 1)
  assert (
      y.shape == expected_shape
  ), f"JAX Shape Mismatch: Expected {expected_shape}, got {y.shape}"
  assert not jnp.isnan(y).any(), "JAX Output contains NaNs"


# pylint: disable=unused-argument
def main(argv):
  # PyTorch Test
  try:
    test_pytorch_independent()
    print("PyTorch Model Shape: VALID (True)")
  except AssertionError as e:
    print(f"PyTorch Model FAILED: {e}")

  # JAX Test
  try:
    test_jax_independent()
    print("JAX Model Shape:     VALID (True)")
  except AssertionError as e:
    print(f"JAX Model FAILED: {e}")
    sys.exit(1)


if __name__ == "__main__":
  app.run(main)
