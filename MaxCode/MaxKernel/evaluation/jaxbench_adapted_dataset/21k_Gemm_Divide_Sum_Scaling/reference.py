# pylint: skip-file
# Imports
import jax
import jax.lax as lax
import jax.numpy as jnp


# Initialization
def get_inputs():
  dtype = jnp.float32
  batch_size = 4096
  input_size = 8192
  hidden_size = 8192
  scaling_factor = 1.5

  key = jax.random.key(0)
  k1, k2 = jax.random.split(key)
  x = jax.random.uniform(k1, (batch_size, input_size), dtype=dtype)
  weight = jax.random.normal(k2, (input_size, hidden_size), dtype=dtype)

  dynamic_args = [x, weight]
  static_args = [scaling_factor]

  return dynamic_args, static_args


# Computation
def computation(x, weight, scaling_factor):
  x = lax.dot_general(
      x,
      weight.T,
      dimension_numbers=(((1,), (0,)), ((), ())),
      precision=lax.Precision.HIGHEST,
  )
  x = x / 2.0
  x = jnp.sum(x, axis=1, keepdims=True)
  x = x * scaling_factor
  return x
