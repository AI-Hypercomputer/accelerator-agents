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

# Define block shapes that are compliant with TPU hardware
# We tile the (batch_size, dim2) output dimensions.
# A (8, 128) tiling is standard for TPUs.
out_block_shape = (8, 128)
# The input block must span the full reduction dimension (dim1)
# and match the output block in the other dimensions.
in_block_shape = (out_block_shape[0], dim1, out_block_shape[1])

# The grid is determined by how many blocks fit into the overall shape
grid = (
  batch_size // out_block_shape[0],
  dim2 // out_block_shape[1],
)


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel to compute the maximum value along axis 1.

  Args:
    x_ref: A reference to a slice of the input tensor `x`. The slice has shape
      (8, 256, 128) and contains the elements over which to compute the max.
    out_ref: A reference to a slice of the output tensor, with
      shape (8, 128), where the result should be stored.
  """
  # The input `x_ref` is a view of the data for the entire block.
  # We can compute the max over axis 1 directly.
  # Pallas will handle loading the data from HBM to vector registers.
  max_val = jnp.max(x_ref, axis=1)

  # Write the resulting block to the output.
  out_ref[...] = max_val


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), x.dtype),
  grid=grid,
  in_specs=[
    pl.BlockSpec(
      block_shape=in_block_shape,
      index_map=lambda i, j: (i * out_block_shape[0], 0, j * out_block_shape[1]),
    )
  ],
  out_specs=pl.BlockSpec(
    block_shape=out_block_shape,
    index_map=lambda i, j: (i * out_block_shape[0], j * out_block_shape[1]),
  ),
)(x).block_until_ready()
