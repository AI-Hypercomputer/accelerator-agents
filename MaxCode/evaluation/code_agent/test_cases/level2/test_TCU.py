"""Compilability and shape checks for TCU (Temporal Convolution Unit)."""

import os
import traceback

# pylint: disable=g-importing-member
from absl import app
from absl import flags
from evaluation.code_agent.pytorch_references.level2.TCU import Chomp1d as TCU_Torch
from evaluation.proto import result_pb2
from generated_code.code_agent.v1.Gemini_2_5_pro.TCU.layers import Chomp1d as TCU_Jax
import immutabledict
import jax
import jax.numpy as jnp
import torch

from google3.net.proto2.python.public import text_format


os.environ["JAX_PLATFORMS"] = "cpu"

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "metrics_output_path",
    None,
    "Path to write the evaluation metrics.",
)

config = immutabledict.immutabledict({
    "n_inputs": 10,
    "n_outputs": 20,
    "batch_size": 4,
    "seq_len": 32,
})


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
    pytorch_result = result_pb2.Result(valid=True)
    print("PyTorch Model Shape: VALID (True)")
  except (AssertionError, TypeError, ValueError) as e:
    error_message = "".join(traceback.format_exception_only(type(e), e))
    pytorch_result = result_pb2.Result(valid=False, error_message=error_message)
    print(f"PyTorch Model FAILED: {e}")

  try:
    test_jax_independent()
    jax_result = result_pb2.Result(valid=True)
    print("JAX Model Shape: VALID (True)")
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
