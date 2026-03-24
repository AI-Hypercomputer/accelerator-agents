# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))

# The block size for the inner dimension.
# This needs to be a multiple of 128 for TPU compatibility.
block_size_d = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel to normalize a row by the sum of absolute values.

  This kernel implements the operation `x / jnp.sum(jnp.abs(x))` for each row.
  It's a two-pass algorithm:
  1. It iterates through the blocks of a row to compute the sum of absolute
     values (`sum_val`).
  2. It iterates through the blocks again, loading each block, dividing by
     `sum_val`, and storing the result in the output.
  """
  # The program ID corresponds to the row index.
  row_idx = pl.program_id(0)
  dim = x_ref.shape[1]
  num_blocks = dim // block_size_d

  # Pass 1: Compute the sum of absolute values for the entire row.
  # We initialize an accumulator `sum_val` to zero.
  sum_val = jnp.zeros((), dtype=x_ref.dtype)

  # We define the body of the loop for the first pass.
  def sum_loop_body(i, acc):
    # Calculate the offset for the current block.
    offset = i * block_size_d
    # Load a block from the input reference for the current row.
    x_block = pl.load(x_ref, (row_idx, offset), block_shape=(1, block_size_d))
    # Add the sum of absolute values of the block to the accumulator.
    return acc + jnp.sum(jnp.abs(x_block))

  # Execute the loop to compute the total sum.
  sum_val = lax.fori_loop(0, num_blocks, sum_loop_body, sum_val)

  # Pass 2: Divide each element by the sum and store it in the output.
  # We define the body of the loop for the second pass.
  def write_loop_body(i, _):
    # Calculate the offset for the current block.
    offset = i * block_size_d
    # Load the same block again from the input reference.
    x_block = pl.load(x_ref, (row_idx, offset), block_shape=(1, block_size_d))
    # Divide the block by the total sum.
    result_block = x_block / sum_val
    # Store the result block in the output reference.
    pl.store(out_ref, (row_idx, offset), result_block)

  # Execute the loop to perform the division and store the results.
  lax.fori_loop(0, num_blocks, write_loop_body, ())


result = pl.pallas_call(
  kernel,
  # The output has the same shape and dtype as the input.
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  # Create a grid where each instance processes one full row.
  grid=(batch_size,),
  # The input and output specs define how Pallas tiles the data.
  # Each kernel instance gets access to a block of size (1, block_size_d).
  # Pallas handles iterating through the inner dimension.
  in_specs=[pl.BlockSpec(lambda i: (i, 0), (1, dim))],
  out_specs=pl.BlockSpec(lambda i: (i, 0), (1, dim)),
)(x).block_until_ready()
