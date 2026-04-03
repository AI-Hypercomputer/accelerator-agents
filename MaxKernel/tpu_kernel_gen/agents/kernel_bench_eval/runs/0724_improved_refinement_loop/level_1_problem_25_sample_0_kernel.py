# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))

# Define block sizes that satisfy TPU constraints
# For a 2D block of shape (b_m, b_n) on an array of shape (16, 16384):
# b_m must be 16 or divisible by 8.
# b_n must be 16384 or divisible by 128.
# We choose b_m=8 and b_n=1024 for a good level of parallelism.
b_m = 8
b_n = 1024


# Computation
def kernel(x_ref, out_ref):
  """
  Pallas kernel for element-wise sigmoid activation (swish).

  This kernel computes x * sigmoid(x) for each element in the input block.

  Args:
    x_ref: A reference to the input block.
    out_ref: A reference to the output block to store the result.
  """
  # Load the input block from HBM into SRAM.
  x = x_ref[...]
  # Perform the element-wise computation. The result is held in registers.
  result = x * jax.nn.sigmoid(x)
  # Write the result from registers to the output block in HBM.
  out_ref[...] = result


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[0] // b_m, x.shape[1] // b_n),
  in_specs=[pl.BlockSpec(block_shape=(b_m, b_n), index_map=lambda i, j: (i, j))],
  out_specs=pl.BlockSpec(block_shape=(b_m, b_n), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()
