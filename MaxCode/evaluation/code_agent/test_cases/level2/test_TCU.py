"""Compilability and shape checks for TCU (Temporal Convolution Unit)."""

import os
import sys

# pylint: disable=g-importing-member
from absl import app
from evaluation.code_agent.pytorch_references.level2.TCU import Chomp1d as TCU_Torch
from generated_code.code_agent.v1.Gemini_2_5_pro.TCU.layers import Chomp1d as TCU_Jax
import jax
import jax.numpy as jnp
import torch

os.environ["JAX_PLATFORMS"] = "cpu"

config = {
    "n_inputs": 10,
    "n_outputs": 20,
    "batch_size": 4,
    "seq_len": 32,
}


def test_pytorch_independent():
  """Sanity check for PyTorch model execution."""
  model = TCU_Torch(chomp_size=2)
  # PyTorch: (Batch, Channels, Length)
  x = torch.randn(config["batch_size"], config["n_inputs"], config["seq_len"])
  y = model(x)

  # Output length = Length - Chomp Size
  expected_shape = (
      config["batch_size"],
      config["n_inputs"],
      config["seq_len"] - 2,
  )
  assert (
      y.shape == expected_shape
  ), f"PyTorch Shape Mismatch: Expected {expected_shape}, got {y.shape}"
  assert not torch.isnan(y).any(), "PyTorch Output contains NaNs"


def test_jax_independent():
  """Sanity check for JAX model execution."""
  model = TCU_Jax(chomp_size=2)

  # JAX: (Batch, Length, Features/Channels)
  x = jnp.ones((config["batch_size"], config["seq_len"], config["n_inputs"]))

  variables = model.init(jax.random.PRNGKey(42), x)
  y = model.apply(variables, x)

  # Output length = Length - Chomp Size
  expected_shape = (
      config["batch_size"],
      config["seq_len"] - 2,
      config["n_inputs"],
  )

  assert (
      y.shape == expected_shape
  ), f"JAX Shape Mismatch: Expected {expected_shape}, got {y.shape}"
  assert not jnp.isnan(y).any(), "JAX Output contains NaNs"


# pylint: disable=unused-argument
def main(argv):
  try:
    test_pytorch_independent()
    print("PyTorch Model Shape: VALID (True)")
  except AssertionError as e:
    print(f"PyTorch Model FAILED: {e}")
    sys.exit(1)

  try:
    test_jax_independent()
    print("JAX Model Shape: VALID (True)")
  except AssertionError as e:
    print(f"JAX Model FAILED: {e}")
    sys.exit(1)


if __name__ == "__main__":
  app.run(main)
