# Imports
import jax
import jax.numpy as jnp


# Initialization
def get_inputs():
  M = 8192
  K = 8192
  N = 28672
  dtype = jnp.bfloat16
  key = jax.random.PRNGKey(42)
  k1, k2 = jax.random.split(key, 2)
  A = jax.random.normal(k1, (M, K), dtype=dtype)
  B = jax.random.normal(k2, (K, N), dtype=dtype) * 0.02
  return [A, B], []


# Computation
def computation(A, B):
  return jnp.dot(A, B)
