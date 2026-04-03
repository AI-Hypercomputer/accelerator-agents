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

# Define block sizes for chunking, ensuring they meet TPU constraints
b_batch = 8
b_dim2 = 128


def kernel(x_ref, out_ref):
  """
  Pallas kernel to compute the product of a tensor along a specified axis.

  This kernel receives a tile of the input tensor `x` and computes the product
  reduction along its second dimension (axis=1). The result is written to the
  corresponding output tile.

  Args:
    x_ref: A reference to a tile of the input tensor `x`. Based on the
      invocation, this tile has shape (b_batch, dim1, b_dim2).
    out_ref: A reference to a tile of the output tensor. This tile has
      shape (b_batch, b_dim2) and is where the result is stored.
  """
  # Initialize the output with ones for the product reduction.
  acc = jnp.ones(out_ref.shape, dtype=out_ref.dtype)

  # Manually implement the product reduction using a fori_loop, as jnp.prod
  # is not supported in Pallas for TPU.
  def body(k, val):
    return val * x_ref[:, k, :]

  # The loop iterates over the reduction dimension (dim1) and accumulates
  # the product.
  result_prod = jax.lax.fori_loop(0, dim1, body, acc)
  out_ref[...] = result_prod


# Computation
# The original computation reduces along axis=1. We parallelize this by
# creating a 2D grid over the batch and dim2 dimensions. Each kernel instance
# processes a tile of the input `x` to produce a corresponding tile in the output.
#
# - Grid: A (batch_size/b_batch, dim2/b_dim2) grid is created. Each
#   kernel instance (i, j) handles a unique tile.
# - in_specs: For each kernel (i, j), we need the corresponding input data.
#   The reduction is over dim1, so we take the full slice along that axis.
#   The block_shape is (b_batch, dim1, b_dim2), and the index_map
#   (i, 0, j) selects the correct tile from `x` while taking the full dim1.
# - out_specs: The output shape is (batch_size, dim2). The block_shape is
#   (b_batch, b_dim2), and the index_map (i, j) maps each kernel instance
#   directly to its output tile.
#
# These block shapes (8, 256, 128) for input and (8, 128) for output
# satisfy TPU memory layout constraints.
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), x.dtype),
  grid=(batch_size // b_batch, dim2 // b_dim2),
  in_specs=[pl.BlockSpec(block_shape=(b_batch, dim1, b_dim2), index_map=lambda i, j: (i, 0, j))],
  out_specs=pl.BlockSpec(block_shape=(b_batch, b_dim2), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()
