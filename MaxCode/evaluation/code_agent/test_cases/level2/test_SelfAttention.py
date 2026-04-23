"""Compilability, Shape, and Equivalence tests for SelfAttention."""

import os
import traceback

# pylint: disable=g-importing-member
from absl import app
from absl import flags
from absl import logging
from evaluation.code_agent.pytorch_references.level2.self_attention import SelfAttention as SelfAttention_Torch
from evaluation.proto import result_pb2
from generated_code.code_agent.v1.Gemini_2_5_pro.SelfAttention.attention_compiled import SelfAttention as SelfAttention_Jax
from flax import core
import immutabledict
import jax
import jax.numpy as jnp
import numpy as np
import torch

from google3.net.proto2.python.public import text_format


os.environ["JAX_PLATFORMS"] = "cpu"
logging.set_stderrthreshold(logging.ERROR)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "metrics_output_path",
    None,
    "Path to write the evaluation metrics.",
)

config = immutabledict.immutabledict({
    "embed_size": 32,
    "heads": 4,
    "batch_size": 2,
    "seq_len": 10,
})


def test_pytorch_independent():
  """Sanity check for PyTorch model execution."""
  model = SelfAttention_Torch(
      embed_size=config["embed_size"], heads=config["heads"]
  )
  model.eval()

  x = torch.randn(config["batch_size"], config["seq_len"], config["embed_size"])
  mask = torch.ones((config["batch_size"], 1, 1, config["seq_len"]))

  out = model(x, x, x, mask)

  expected_shape = (
      config["batch_size"],
      config["seq_len"],
      config["embed_size"],
  )
  assert (
      out.shape == expected_shape
  ), f"PT Shape Mismatch: Expected {expected_shape}, got {out.shape}"
  assert not torch.isnan(out).any(), "PT Output contains NaNs"
  return True


def test_jax_independent():
  """Sanity check for JAX model execution."""
  model = SelfAttention_Jax(
      embed_size=config["embed_size"], heads=config["heads"]
  )

  rng = jax.random.PRNGKey(0)
  x = jax.random.normal(
      rng, (config["batch_size"], config["seq_len"], config["embed_size"])
  )
  mask = jnp.ones((config["batch_size"], 1, 1, config["seq_len"]))

  variables = model.init(rng, x, x, x, mask)
  out = model.apply(variables, x, x, x, mask)

  expected_shape = (
      config["batch_size"],
      config["seq_len"],
      config["embed_size"],
  )
  assert (
      out.shape == expected_shape
  ), f"JAX Shape Mismatch: Expected {expected_shape}, got {out.shape}"
  assert not jnp.isnan(out).any(), "JAX Output contains NaNs"
  return True


def test_attention_equivalence():
  """Validates numerical equivalence by transferring weights layer-by-layer."""
  # 1. Initialize Models
  pt_model = SelfAttention_Torch(config["embed_size"], config["heads"]).eval()
  jax_model = SelfAttention_Jax(config["embed_size"], config["heads"])

  # 2. Init JAX params
  dummy_input = jnp.ones(
      (config["batch_size"], config["seq_len"], config["embed_size"])
  )
  variables = jax_model.init(
      jax.random.PRNGKey(0), dummy_input, dummy_input, dummy_input, None
  )
  new_params = core.unfreeze(variables["params"])

  # 3. Helper for Weight Transfer (PT -> JAX)
  def copy_layer(pt_layer, jax_name):
    # Transpose Kernel: PyTorch (Out, In) -> JAX (In, Out)
    new_params[jax_name]["kernel"] = pt_layer.weight.detach().numpy().T
    if pt_layer.bias is not None and "bias" in new_params[jax_name]:
      new_params[jax_name]["bias"] = pt_layer.bias.detach().numpy()

  # Mapping based on your Pdb finding
  copy_layer(pt_model.values, "values_projection")
  copy_layer(pt_model.keys, "keys_projection")
  copy_layer(pt_model.queries, "queries_projection")
  copy_layer(pt_model.fc_out, "output_projection")

  final_vars = {"params": core.freeze(new_params)}

  # 4. Shared Inputs
  np_input = np.random.randn(
      config["batch_size"], config["seq_len"], config["embed_size"]
  ).astype(np.float32)
  pt_in = torch.from_numpy(np_input)
  jax_in = jnp.array(np_input)

  # 5. Inference
  with torch.no_grad():
    pt_out = pt_model(pt_in, pt_in, pt_in, None).numpy()

  jax_out = jax_model.apply(final_vars, jax_in, jax_in, jax_in, None)

  # 6. Compare & Detailed Debug
  max_diff = np.max(np.abs(pt_out - np.array(jax_out)))
  print(f"   >> Max Absolute Difference: {max_diff:.2e}")

  # Check intermediate energy scaling (helps detect why it might fail)
  head_dim = config["embed_size"] // config["heads"]
  scale = head_dim**0.5
  print(f"   >> Head Dim: {head_dim}, Scaling Factor: {scale}")

  # Final Assertion
  np.testing.assert_allclose(pt_out, jax_out, rtol=1e-4, atol=1e-5)
  return True


# pylint: disable=unused-argument
def main(argv):
  # Shape Checks
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
    print("JAX Model Shape:     VALID (True)")
  except (AssertionError, TypeError, ValueError) as e:
    error_message = "".join(traceback.format_exception_only(type(e), e))
    jax_result = result_pb2.Result(valid=False, error_message=error_message)
    print(f"JAX Model FAILED: {e}")

  # Equivalence
  try:
    test_attention_equivalence()
    print("Equivalence Test:    VALID (True)")
  except AssertionError as e:
    print(f"Equivalence Test FAILED: {e}")

  if FLAGS.metrics_output_path:
    eval_result = result_pb2.EvaluationResult(
        pytorch_result=pytorch_result,
        jax_result=jax_result,
    )
    with open(FLAGS.metrics_output_path, "w") as f:
      f.write(text_format.MessageToString(eval_result))


if __name__ == "__main__":
  app.run(main)
