"""Compilability and shape checks for GraphConvolution."""

import os
import sys

from absl import app
from evaluation.code_agent.pytorch_references.level2.graph_convolution import GraphConvolution as GraphConvolution_Torch
from generated_code.code_agent.v1.Gemini_2_5_pro.GraphConvolution.layers import GraphConvolution as GraphConvolution_Jax
import jax
import jax.numpy as jnp
import torch


os.environ["JAX_PLATFORMS"] = "cpu"

CONFIG = {
    "n_nodes": 10,
    "in_features": 5,
    "out_features": 8,
}


def test_pytorch_independent():
  """Sanity check for PyTorch model execution."""
  model = GraphConvolution_Torch(CONFIG["in_features"], CONFIG["out_features"])
  model.eval()

  node_features = torch.ones((CONFIG["n_nodes"], CONFIG["in_features"]))
  adj_matrix = torch.randint(
      0, 2, (CONFIG["n_nodes"], CONFIG["n_nodes"])
  ).float()
  out = model(node_features, adj_matrix)
  expected_shape = (CONFIG["n_nodes"], CONFIG["out_features"])

  assert (
      out.shape == expected_shape
  ), f"PyTorch Shape Mismatch: Expected {expected_shape}, got {out.shape}"
  assert not torch.isnan(out).any(), "PyTorch Output contains NaNs"


def test_jax_independent():
  """Sanity check for JAX model execution."""
  model = GraphConvolution_Jax(CONFIG["out_features"])

  node_features = jnp.ones((CONFIG["n_nodes"], CONFIG["in_features"]))
  adj_matrix = jax.random.randint(
      jax.random.PRNGKey(0), (CONFIG["n_nodes"], CONFIG["n_nodes"]), 0, 2
  ).astype(jnp.float32)

  variables = model.init(jax.random.PRNGKey(42), node_features, adj_matrix)
  out = model.apply(variables, node_features, adj_matrix)
  expected_shape = (CONFIG["n_nodes"], CONFIG["out_features"])

  assert (
      out.shape == expected_shape
  ), f"JAX Shape Mismatch: Expected {expected_shape}, got {out.shape}"
  assert not jnp.isnan(out).any(), "JAX Output contains NaNs"


# pylint: disable=unused-argument
def main(argv):
  try:
    test_pytorch_independent()
    print("PyTorch Model Shape: VALID (True)")
  except (AssertionError, TypeError, ValueError) as e:
    print(f"PyTorch Model FAILED: {e}")
    sys.exit(1)

  try:
    test_jax_independent()
    print("JAX Model Shape: VALID (True)")
  except (AssertionError, TypeError, ValueError) as e:
    print(f"JAX Model FAILED: {e}")
    sys.exit(1)


if __name__ == "__main__":
  app.run(main)
