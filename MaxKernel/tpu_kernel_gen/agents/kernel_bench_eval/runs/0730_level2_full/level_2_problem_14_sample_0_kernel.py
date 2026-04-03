# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_size = 10
hidden_size = 20
scaling_factor = 1.5
key = random.PRNGKey(0)
key_x, key_weight = random.split(key)
x = random.normal(key_x, (batch_size, input_size))
weight = random.normal(key_weight, (hidden_size, input_size))
b_size = 8


# Computation
def kernel(x_ref, weight_ref, out_ref, scaling_factor):
  """Pallas kernel for the given JAX computation."""
  # Perform the matrix multiplication between the input slice and the transposed weight matrix.
  # x_ref shape: (b_size, input_size)
  # weight_ref.T shape: (input_size, hidden_size)
  # y shape: (b_size, hidden_size)
  y = jnp.matmul(x_ref[...], weight_ref[...].T)

  # Apply the element-wise division.
  y = y / 2.0

  # Sum the results along the hidden dimension.
  # The output shape becomes (b_size, 1).
  y = jnp.sum(y, axis=1, keepdims=True)

  # Apply the final scaling factor.
  y = y * scaling_factor

  # Write the final result to the output buffer.
  out_ref[...] = y


result = pl.pallas_call(
  lambda x_ref, weight_ref, out_ref: kernel(x_ref, weight_ref, out_ref, scaling_factor),
  out_shape=jax.ShapeDtypeStruct((batch_size, 1), x.dtype),
  grid=(batch_size // b_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(b_size, input_size), index_map=lambda i: (i * b_size, 0)),
    pl.BlockSpec(block_shape=(hidden_size, input_size), index_map=lambda _: (0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(b_size, 1), index_map=lambda i: (i * b_size, 0)),
)(x, weight).block_until_ready()
