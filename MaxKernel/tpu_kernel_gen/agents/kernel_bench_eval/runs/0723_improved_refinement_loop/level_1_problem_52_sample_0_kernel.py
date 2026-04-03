# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2))


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for argmin.

  This kernel computes the argmin of a slice of the input tensor `x`
  and writes the result to the output tensor.

  Args:
    x_ref: A reference to a block of the input tensor. The shape is
      (8, 256, 128), where 256 is the reduction dimension.
    out_ref: A reference to a block of the output tensor. The shape is
      (8, 128).
  """
  # The core computation is to find the indices of the minimum values along
  # the reduction axis (axis=1) of the input block.
  # This is done by iterating through the reduction dimension and keeping
  # track of the minimum value and its index.
  min_indices = jnp.zeros_like(out_ref, dtype=jnp.int32)
  min_vals = jnp.full_like(out_ref, jnp.inf, dtype=x_ref.dtype)

  def body(i, carry):
    min_vals, min_indices = carry
    row = x_ref[:, i, :]
    is_lower = row < min_vals
    new_min_vals = jnp.where(is_lower, row, min_vals)
    new_min_indices = jnp.where(is_lower, jnp.full_like(min_indices, i), min_indices)
    return new_min_vals, new_min_indices

  _, min_indices = lax.fori_loop(0, x_ref.shape[1], body, (min_vals, min_indices))
  out_ref[...] = min_indices


# The kernel is expected to compute a block of the argmin result.
# To compute an (8, 128) block of the output, it needs an input block
# of shape (8, 256, 128), where 256 is the reduction dimension.
result = pl.pallas_call(
  kernel,
  # The output shape is the input shape with the reduction axis removed.
  # The dtype for argmin is integer.
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), jnp.int32),
  # Grid is sized to tile the output tensor with blocks of shape (8, 128).
  grid=(batch_size // 8, dim2 // 128),
  in_specs=[
    # For each output block, we need a corresponding slice of the input.
    # The slice spans the full reduction dimension (dim1).
    pl.BlockSpec((8, dim1, 128), lambda i, j: (i * 8, 0, j * 128))
  ],
  out_specs=pl.BlockSpec((8, 128), lambda i, j: (i * 8, j * 128)),
  compiler_params=dict(mosaic=dict(dimension_semantics=("parallel", "parallel"))),
)(x).block_until_ready()
