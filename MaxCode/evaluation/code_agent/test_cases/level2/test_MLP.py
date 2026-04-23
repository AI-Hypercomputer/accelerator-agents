"""Compilability and shape checks for CreditScoringMLP."""

import os
import traceback

from absl import app
from absl import flags
# pylint: disable=g-importing-member
from evaluation.code_agent.pytorch_references.level2.MLP import CreditScoringMLP as CreditScoringMLP_Torch
from evaluation.proto import result_pb2
from generated_code.code_agent.v1.Gemini_2_5_pro.CreditScoringMLP.model import CreditScoringMLP as CreditScoringMLP_Jax
import immutabledict
import jax
from jax import config
import jax.numpy as jnp
import torch

from google3.net.proto2.python.public import text_format


os.environ["JAX_PLATFORMS"] = "cpu"
config.update("jax_default_matmul_precision", "highest")

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "metrics_output_path",
    None,
    "Path to write the evaluation metrics.",
)

TEST_CONFIG = immutabledict.immutabledict({
    "input_features": 20,
    "batch_size": 4,
})


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
    pytorch_result = result_pb2.Result(valid=True)
    print("PyTorch Model Shape: VALID (True)")
  except (AssertionError, TypeError, ValueError) as e:
    error_message = "".join(traceback.format_exception_only(type(e), e))
    pytorch_result = result_pb2.Result(valid=False, error_message=error_message)
    print(f"PyTorch Model FAILED: {e}")

  # JAX Test
  try:
    test_jax_independent()
    jax_result = result_pb2.Result(valid=True)
    print("JAX Model Shape:     VALID (True)")
  except (AssertionError, TypeError, ValueError) as e:
    error_message = "".join(traceback.format_exception_only(type(e), e))
    jax_result = result_pb2.Result(valid=False, error_message=error_message)
    print(f"JAX Model FAILED: {e}")

  if FLAGS.metrics_output_path:
    eval_result = result_pb2.EvaluationResult(
        pytorch_result=pytorch_result,
        jax_result=jax_result,
    )
    with open(FLAGS.metrics_output_path, "w") as f:
      f.write(text_format.MessageToString(eval_result))


if __name__ == "__main__":
  app.run(main)
