"""Compilability and shape checks for Variational Autoencoder (VAE).

Compares PyTorch and JAX models.
"""

import os
import traceback

from absl import app
from absl import flags
import evaluation.code_agent.pytorch_references.level2.VAE as VAE_Torch
from evaluation.proto import result_pb2
import generated_code.code_agent.v1.Gemini_2_5_pro.VAE.model as model_jax
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
    "input_dim": 784,
    "hidden_dim": 128,
    "latent_dim": 10,
    "batch_size": 4,
})


def test_pytorch_independent():
  """Sanity check for PyTorch model execution."""
  model = VAE_Torch.VAE(
      input_dim=config["input_dim"],
      hidden_dim=config["hidden_dim"],
      latent_dim=config["latent_dim"],
  )
  model.eval()

  dummy_input = torch.ones((config["batch_size"], config["input_dim"]))
  recon, mu, log_var = model(dummy_input)

  # Shape and NaN checks
  assert recon.shape == (config["batch_size"], config["input_dim"]), (
      "PyTorch Recon Shape Mismatch: Expected"
      f" {(config['batch_size'], config['input_dim'])}, got {recon.shape}"
  )
  assert mu.shape == (config["batch_size"], config["latent_dim"]), (
      "PyTorch Mu Shape Mismatch: Expected"
      f" {(config['batch_size'], config['latent_dim'])}, got {mu.shape}"
  )
  assert log_var.shape == (config["batch_size"], config["latent_dim"]), (
      "PyTorch LogVar Shape Mismatch: Expected"
      f" {(config['batch_size'], config['latent_dim'])}, got {log_var.shape}"
  )
  assert not torch.isnan(recon).any(), "PyTorch Reconstruction contains NaNs"
  assert not torch.isnan(mu).any(), "PyTorch Mu contains Na Ns"
  assert not torch.isnan(log_var).any(), "PyTorch LogVar contains NaNs"


def test_jax_independent():
  """Sanity check for JAX model execution."""
  model = model_jax.VAE(
      input_dim=config["input_dim"],
      hidden_dim=config["hidden_dim"],
      latent_dim=config["latent_dim"],
  )
  dummy_input = jnp.ones((config["batch_size"], config["input_dim"]))
  keys = {"params": jax.random.PRNGKey(42), "sampling": jax.random.PRNGKey(0)}
  variables = model.init(keys, dummy_input)

  # The .apply() call was already correct using rngs=...
  recon, mu, log_var = model.apply(
      variables, dummy_input, rngs={"sampling": jax.random.PRNGKey(0)}
  )
  # Shape and NaN checks
  assert recon.shape == (config["batch_size"], config["input_dim"]), (
      "JAX Recon Shape Mismatch: Expected"
      f" {(config['batch_size'], config['input_dim'])}, got {recon.shape}"
  )
  assert mu.shape == (config["batch_size"], config["latent_dim"]), (
      "JAX Mu Shape Mismatch: Expected"
      f" {(config['batch_size'], config['latent_dim'])}, got {mu.shape}"
  )
  assert log_var.shape == (config["batch_size"], config["latent_dim"]), (
      "JAX LogVar Shape Mismatch: Expected"
      f" {(config['batch_size'], config['latent_dim'])}, got {log_var.shape}"
  )
  assert not jnp.isnan(recon).any(), "JAX Reconstruction contains NaNs"
  assert not jnp.isnan(mu).any(), "JAX Mu contains NaNs"
  assert not jnp.isnan(log_var).any(), "JAX LogVar contains NaNs"


# pylint: disable=unused-argument
def main(argv):
  # Shape and NaN checks
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
