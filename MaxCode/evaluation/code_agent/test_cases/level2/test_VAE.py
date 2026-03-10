"""Compilability and shape checks for Variational Autoencoder (VAE) to compare PyTorch and JAX models."""

import os
import sys

from absl import app
from evaluation.code_agent.pytorch_references.level2.VAE import VAE as VAE_Torch
from generated_code.code_agent.v1.Gemini_2_5_pro.VAE.models import VAE as VAE_Jax
import jax
import jax.numpy as jnp
import torch


os.environ["JAX_PLATFORMS"] = "cpu"

config = {
    "input_dim": 784,
    "hidden_dim": 128,
    "latent_dim": 10,
    "batch_size": 4,
}


def test_pytorch_independent():
  """Sanity check for PyTorch model execution."""
  model = VAE_Torch(
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
  model = VAE_Jax(
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
