# Imports
import jax
import jax.numpy as jnp

# Initialization
Array = jnp.ndarray
DType = jnp.dtype

matrix_dim = 2048
dtype = jnp.bfloat16

X = jax.random.normal(jax.random.PRNGKey(0), (matrix_dim, matrix_dim), dtype=dtype)
Y = jax.random.normal(jax.random.PRNGKey(1), (matrix_dim, matrix_dim), dtype=dtype)


# Computation
def computation(
  X: Array,
  Y: Array,
):
  """Computes ReLU(X + Y)."""
  sum_inputs = X + Y
  return jax.nn.relu(sum_inputs)


output = jax.block_until_ready(computation(X, Y))
