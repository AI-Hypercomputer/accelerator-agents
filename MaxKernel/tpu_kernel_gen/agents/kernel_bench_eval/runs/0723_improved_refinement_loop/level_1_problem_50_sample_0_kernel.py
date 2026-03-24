# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim1 = 256
dim2 = 256
reduction_dim = 1
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2))


def kernel(x_ref, out_ref):
  """
  Pallas kernel to compute the product of elements along axis=1.

  Args:
    x_ref: A reference to a slice of the input tensor `x`. For each kernel
      instance, this will be a block of shape (1, dim1, dim2), representing
      the slice x[i, :, :].
    out_ref: A reference to a slice of the output tensor. For each kernel
      instance, this will be a block of shape (1, dim2), representing
      the output slice out[i, :].
  """
  # Initialize an accumulator with ones. The accumulator will have the
  # same shape as the output slice, which is (1, dim2).
  acc = jnp.ones(out_ref.shape, dtype=x_ref.dtype)
  # Iterate over the reduction dimension (dim1).
  for i in range(x_ref.shape[1]):
    # In each iteration, multiply the accumulator by the current slice
    # of the input. x_ref[0, i, :] has shape (dim2,).
    # Broadcasting will apply the multiplication across the accumulator.
    acc *= x_ref[0, i, :]
  # Write the final accumulated value to the output.
  out_ref[:] = acc


# Computation
# The reduction is over axis=1. We can parallelize the computation
# across the batch dimension (axis 0).
# We create a 1D grid where each grid element (i) corresponds to
# the i-th batch.
#
# Each kernel instance is responsible for computing a 1D slice
# in the output tensor. To do this, it needs to read a 2D slice of the
# input tensor `x`.
#
# - grid: A 1D grid of size (batch_size,) to parallelize the batch dimension.
# - in_specs: For each grid element (i), we need the slice x[i, :, :].
#   This corresponds to a block of shape (1, dim1, dim2) starting at index (i, 0, 0).
# - out_specs: Each grid element (i) computes a 2D slice of the output.
#   This corresponds to a block of shape (1, dim2) in the output tensor at index (i, 0).
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), x.dtype),
  grid=(batch_size,),
  in_specs=[pl.BlockSpec(block_shape=(1, dim1, dim2), index_map=lambda i: (i, 0, 0))],
  out_specs=pl.BlockSpec(block_shape=(1, dim2), index_map=lambda i: (i, 0)),
)(x).block_until_ready()
