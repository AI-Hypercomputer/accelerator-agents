import jax
import numpy as np
import torch
import torch.nn as nn


def compare_jax_pytorch_random_init(seed=42, input_dim=3, output_dim=2):
  """
  Compare random initialization in JAX vs PyTorch using the same seed.

  Args:
      seed: Random seed to use for both frameworks
      input_dim: Input dimension for the linear layer
      output_dim: Output dimension for the linear layer
  """
  print(f"Using random seed: {seed}")
  print(f"Creating linear layers with shape: {input_dim} -> {output_dim}")

  # Set random seeds
  np.random.seed(seed)  # For NumPy/JAX
  torch.manual_seed(seed)  # For PyTorch

  # JAX initialization
  key = jax.random.PRNGKey(seed)
  # Initialize weights using the same scheme as PyTorch (uniform distribution)
  k_bound = 1.0 / np.sqrt(input_dim)
  jax_weights = jax.random.uniform(key, (input_dim, output_dim), minval=-k_bound, maxval=k_bound)

  # PyTorch linear layer
  torch_layer = nn.Linear(input_dim, output_dim, bias=False)
  # Reset initialization to use uniform distribution with bounds matching JAX
  nn.init.uniform_(torch_layer.weight, -k_bound, k_bound)
  # Need to transpose PyTorch weights for comparison since they're stored differently
  torch_weights = torch_layer.weight.data.numpy().T

  print("\n--- JAX weights ---")
  print(jax_weights)

  print("\n--- PyTorch weights (transposed for comparison) ---")
  print(torch_weights)

  print("\n--- Difference between JAX and PyTorch weights ---")
  diff = jax_weights - torch_weights
  print(diff)
  print(f"Max absolute difference: {np.max(np.abs(diff))}")

  # Check if weights are close
  are_close = np.allclose(jax_weights, torch_weights)
  print(f"\nAre the weights approximately equal? {are_close}")

  if not are_close:
    print("\nNote: Even with the same seed, JAX and PyTorch may use different")
    print("random number generation algorithms, leading to different values.")


if __name__ == "__main__":
  # Try with different seeds
  compare_jax_pytorch_random_init(seed=42)
  print("\n" + "=" * 50 + "\n")
  compare_jax_pytorch_random_init(seed=123)
