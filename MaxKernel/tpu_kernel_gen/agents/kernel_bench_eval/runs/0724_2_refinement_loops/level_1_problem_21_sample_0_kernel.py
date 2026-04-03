# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
# For an element-wise operation, we can process the data in blocks.
# We choose a block size for the inner dimension that is compatible with TPU hardware.
block_dim = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for element-wise sigmoid.

  Args:
    x_ref: A reference to the input block.
    out_ref: A reference to the output block to be written to.
  """
  # Load the input data for the current block from SRAM.
  x = x_ref[...]
  # Compute the sigmoid element-wise.
  result = jax.nn.sigmoid(x)
  # Write the result to the corresponding output block in SRAM.
  out_ref[...] = result


# The kernel will be parallelized across the `dim` dimension.
# The grid will have `dim // block_dim` instances. Each instance will process
# a `(batch_size, block_dim)` slice of the data.
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(1, dim // block_dim),
  # The input spec slices `x` into vertical blocks. The grid index `j`
  # selects which block to process.
  in_specs=[pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i, j: (0, j))],
  # The output spec maps each grid instance to the corresponding block in the
  # output array.
  out_specs=pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i, j: (0, j)),
)(x).block_until_ready()
