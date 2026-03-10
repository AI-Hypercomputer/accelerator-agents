"""Compilability and shape checks for LSTMAttentionForecaster."""

import os
import sys

# pylint: disable=g-importing-member
from absl import app
from evaluation.code_agent.pytorch_references.level2.lstm_attention_forecaster import LSTMAttentionForecaster as LSTMAttentionForecaster_Torch
from generated_code.code_agent.v1.Gemini_2_5_pro.LSTMAttentionForecaster.model import LSTMAttentionForecaster as LSTMAttentionForecaster_Jax
import jax
from jax import config as jax_config
import jax.numpy as jnp
import torch


os.environ["JAX_PLATFORMS"] = "cpu"
jax_config.update("jax_default_matmul_precision", "highest")

CONFIG = {
    "input_size": 5,
    "hidden_size": 10,
    "output_size": 3,
    "batch_size": 4,
    "seq_len": 8,
}


def test_pytorch_independent():
  """Sanity check for PyTorch model execution."""
  model = LSTMAttentionForecaster_Torch(
      input_size=CONFIG["input_size"],
      hidden_size=CONFIG["hidden_size"],
      output_size=CONFIG["output_size"],
  )
  model.eval()
  x = torch.randn(CONFIG["batch_size"], CONFIG["seq_len"], CONFIG["input_size"])
  y = model(x)

  expected_shape = (CONFIG["batch_size"], CONFIG["output_size"])
  assert (
      y.shape == expected_shape
  ), f"PyTorch Shape Mismatch: Expected {expected_shape}, got {y.shape}"
  assert not torch.isnan(y).any(), "PyTorch Output contains NaNs"


def test_jax_independent():
  """Sanity check for JAX model execution."""
  model = LSTMAttentionForecaster_Jax(
      input_size=CONFIG["input_size"],
      hidden_size=CONFIG["hidden_size"],
      output_size=CONFIG["output_size"],
  )
  x = jnp.ones((CONFIG["batch_size"], CONFIG["seq_len"], CONFIG["input_size"]))

  # Initialize variables
  rng = jax.random.PRNGKey(42)
  variables = model.init(rng, x)

  # Run inference
  y = model.apply(variables, x)

  expected_shape = (CONFIG["batch_size"], CONFIG["output_size"])
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
    # Don't exit yet, try JAX too

  try:
    test_jax_independent()
    print("JAX Model Shape:     VALID (True)")
  except AssertionError as e:
    print(f"JAX Model FAILED: {e}")
    sys.exit(1)


if __name__ == "__main__":
  app.run(main)
