"""
Expected JAX output for Triton matrix multiplication example.
This shows the simplified algorithm without GPU-specific optimizations.
"""

# Imports
import jax
import jax.numpy as jnp
import jax.random as random

# Initialization
M, N, K = 1024, 1024, 1024
key = random.PRNGKey(0)
key_a, key_b = random.split(key)

A = random.normal(key_a, (M, K))
B = random.normal(key_b, (K, N))


# Computation
def computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
  return jnp.matmul(A, B)


C = jax.block_until_ready(computation(A, B))
