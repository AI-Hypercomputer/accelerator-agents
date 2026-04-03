"""
Expected JAX output for CUDA vector addition example.
This shows what the conversion should produce.
"""

# Imports
import jax
import jax.numpy as jnp

# Initialization
N = 1024
A = jnp.arange(N, dtype=jnp.float32)
B = jnp.arange(N, dtype=jnp.float32) * 2


# Computation
def computation(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
  return A + B


C = jax.block_until_ready(computation(A, B))
