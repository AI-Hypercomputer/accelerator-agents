# Imports
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
block_dim = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for element-wise sigmoid.

  Args:
    x_ref: A reference to a block of the input array.
    out_ref: A reference to a block of the output array to write results to.
  """
  # Apply the sigmoid function to the input block and write it to the output block.
  # The `[...]` notation ensures that we are operating on the entire block of
  # data that Pallas has loaded into SRAM for this kernel instance.
  out_ref[...] = jax.nn.sigmoid(x_ref[...])


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  # Create a 1D grid to iterate over the blocks along the `dim` dimension.
  grid=(dim // block_dim,),
  # For each grid index `i`, we take a slice of the full input `x`.
  # The index_map `lambda i: (0, i)` maps grid index `i` to block index (0, i),
  # effectively selecting the slice `x[:, i*block_dim:(i+1)*block_dim]`.
  in_specs=[pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i: (0, i))],
  # The output is blocked in the same way as the input.
  out_specs=pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i: (0, i)),
)(x).block_until_ready()
