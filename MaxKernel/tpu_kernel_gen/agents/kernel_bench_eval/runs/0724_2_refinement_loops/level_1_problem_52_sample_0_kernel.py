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
# The computation is an argmin over axis 1. We can parallelize this operation
# across the other two axes (axis 0, the batch dimension, and axis 2).
# To satisfy TPU memory layout constraints, we create blocks that are divisible
# by 8 and 128 in their last two dimensions.
# We define a 2D grid that processes blocks of the input array.
# Grid dimensions correspond to chunks of the batch size and the last dimension.
# b_batch: block size for the batch dimension (axis 0), must be divisible by 8.
# b_dim2: block size for the last dimension (axis 2), must be divisible by 128.
b_batch = 8
b_dim2 = 128


# Computation
def kernel(x_ref, out_ref):
  """
  Pallas kernel for argmin operation.

  This kernel computes the argmin of a 3D input tensor `x` along axis 1.
  Each kernel instance processes a block of the input tensor.

  Args:
    x_ref: A reference to a block of the input tensor `x`. The shape of this
      block is (b_batch, dim1, b_dim2).
    out_ref: A reference to a block of the output tensor. The kernel will
      write the result of the argmin operation into this block. The shape of
      this block is (b_batch, b_dim2).
  """
  # Manually perform the argmin operation on the input block along the
  # reduction axis (axis=1), since jnp.argmin is not supported in Pallas.
  min_indices = jnp.zeros(out_ref.shape, dtype=jnp.int32)
  min_vals = x_ref[:, 0, :]

  def body(k, state):
    min_indices, min_vals = state
    current_vals = x_ref[:, k, :]
    is_lower = current_vals < min_vals
    new_min_vals = jnp.where(is_lower, current_vals, min_vals)
    # The following is an alternative to jnp.where(is_lower, k, min_indices)
    # that avoids broadcasting issues with the scalar loop variable `k` in Mosaic.
    new_min_indices = min_indices * (~is_lower) + k * is_lower
    return new_min_indices, new_min_vals

  min_indices, _ = jax.lax.fori_loop(1, dim1, body, (min_indices, min_vals))
  # The result is then written in-place to the output reference.
  out_ref[...] = min_indices


result = pl.pallas_call(
  kernel,
  # The output shape has the reduction axis (dim=1) removed.
  # The dtype for argmin is integer.
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), jnp.int32),
  # The grid is determined by how many blocks fit into the non-reduction axes.
  grid=(batch_size // b_batch, dim2 // b_dim2),
  in_specs=[
    # Each kernel instance gets a block of x. The block shape includes the
    # full reduction dimension (dim1) and chunks of the other dimensions.
    # The index_map (i, 0, j) maps the 2D grid to the 3D input blocks,
    # ensuring we always take the full slice along the reduction axis.
    pl.BlockSpec(block_shape=(b_batch, dim1, b_dim2), index_map=lambda i, j: (i, 0, j))
  ],
  out_specs=pl.BlockSpec(
    # The output block shape corresponds to the processed input block,
    # with the reduction dimension removed.
    block_shape=(b_batch, b_dim2),
    index_map=lambda i, j: (i, j),
  ),
)(x).block_until_ready()
