import jax
import jax.numpy as jnp
import numpy as np
import torch

# Results are nearly identical when using cpu backend
jax.config.update("jax_platform_name", "cpu")


def compare_matmul():
  # Set random seed for reproducibility
  np.random.seed(42)

  # Create random matrices
  size = 2048
  A = np.random.randn(size, size).astype(np.float32)
  B = np.random.randn(size, size).astype(np.float32)

  # PyTorch matmul
  torch_A = torch.from_numpy(A)
  torch_B = torch.from_numpy(B)

  torch_result = torch.matmul(torch_A, torch_B).numpy()

  # JAX matmul
  jax_A = jnp.array(A)
  jax_B = jnp.array(B)

  jax_result = np.array(jax_A @ jax_B)

  # Compare results
  max_diff = np.max(np.abs(torch_result - jax_result))
  print(f"Max difference: {max_diff}")
  avg_diff = np.mean(np.abs(torch_result - jax_result))
  print(f"Average difference: {avg_diff}")
  # Print average magnitude of both results
  torch_avg_magnitude = np.mean(np.abs(torch_result))
  jax_avg_magnitude = np.mean(np.abs(jax_result))
  print(f"PyTorch result average magnitude: {torch_avg_magnitude:.4f}")
  print(f"JAX result average magnitude: {jax_avg_magnitude:.4f}")

  are_close = np.allclose(torch_result, jax_result, rtol=1e-2, atol=1e-2)
  print(f"Results are close enough: {are_close}")

  return torch_result, jax_result


if __name__ == "__main__":
  torch_result, jax_result = compare_matmul()
