# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2))


# Computation
def kernel(x_ref, out_ref):
  """
  Pallas kernel to compute the sum of a 3D tensor along axis=1.

  Args:
    x_ref: A reference to a slice of the input tensor. Based on the
           invocation, this will be a matrix of shape (1, dim1, dim2).
    out_ref: A reference to a slice of the output tensor. Based on the
             invocation, this will be a row vector of shape (1, 1, dim2).
  """
  # Each kernel instance is responsible for summing one slice of shape
  # (1, dim1, dim2) from the input tensor.
  x_slice = x_ref[...]
  # We compute the sum along axis=1 of the slice (which corresponds to
  # axis=1 of the original tensor).
  out_slice = jnp.sum(x_slice, axis=1, keepdims=True)

  # We write the resulting row vector to the location pointed to by out_ref.
  out_ref[...] = out_slice


# Define the Pallas kernel invocation
result = pl.pallas_call(
  kernel,
  # The output shape is (batch_size, 1, dim2) due to keepdims=True
  out_shape=jax.ShapeDtypeStruct((batch_size, 1, dim2), x.dtype),
  # Grid parallelizes over the batch dimension
  grid=(batch_size,),
  # Each kernel instance gets a (1, dim1, dim2) slice from x
  in_specs=[pl.BlockSpec(block_shape=(1, dim1, dim2), index_map=lambda i: (i, 0, 0))],
  # Each kernel instance writes a (1, 1, dim2) slice to the output
  out_specs=pl.BlockSpec(block_shape=(1, 1, dim2), index_map=lambda i: (i, 0, 0)),
)(x).block_until_ready()
