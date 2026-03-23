# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from jax.lax import scan

# Initialization
batch_size = 128
input_shape = (4000,)
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, *input_shape))
# Define a block size for rows that is compatible with TPU constraints (divisible by 8)
row_block_size = 8
# Define a block size for columns for processing inside the kernel.
col_block_size = 128


# Computation
def kernel(x_ref, out_ref):
  """
  Pallas kernel to compute the cumulative sum along axis 1.

  This kernel processes a block of rows from the input tensor. For each row
  in the block, it computes the cumulative sum independently and writes the
  result to the corresponding location in the output tensor.

  Args:
    x_ref: A reference to the input block of shape (row_block_size, num_cols).
    out_ref: A reference to the output block of the same shape.
  """
  # The `jnp.cumsum` primitive and manual scalar writes are not supported in
  # Pallas for TPU. We implement a blocked version of the cumulative sum.
  # For each row, we iterate through it in blocks, computing a local
  # cumulative sum for each block and carrying over the total sum from the
  # previous block.
  num_cols = x_ref.shape[1]
  num_col_blocks = (num_cols + col_block_size - 1) // col_block_size

  def scan_body(carry, x_val):
    """Scan body to compute cumulative sum on a block."""
    new_carry = carry + x_val
    return new_carry, new_carry

  for i in range(x_ref.shape[0]):
    # This accumulator will hold the sum of the previous block.
    block_accumulator = jnp.zeros((), dtype=x_ref.dtype)
    for j in range(num_col_blocks):
      # Define the slice for the current column block.
      col_slice = pl.dslice(j * col_block_size, col_block_size)

      # Load a block of data from the input. `pl.load` handles boundary
      # conditions by masking, preventing out-of-bounds errors.
      input_block = pl.load(x_ref, (i, col_slice))

      # Compute the cumulative sum within the block using `lax.scan`.
      # The final carry from the scan is the sum of the entire block.
      block_sum, output_block = scan(scan_body, jnp.zeros((), dtype=x_ref.dtype), input_block)

      # Add the sum of the previous blocks to the current block's cumsum.
      output_block_with_carry = output_block + block_accumulator

      # Write the result to the output reference. `pl.store` handles
      # boundary conditions.
      pl.store(out_ref, (i, col_slice), output_block_with_carry)

      # Update the accumulator with the sum of the current block.
      block_accumulator += block_sum


# The kernel is parallelized over blocks of rows. Each kernel instance
# handles `row_block_size` rows and computes the cumulative sum for each of them.
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[0] // row_block_size,),
  in_specs=[
    pl.BlockSpec(
      block_shape=(row_block_size, x.shape[1]),
      index_map=lambda i: (i * row_block_size, 0),
    )
  ],
  out_specs=pl.BlockSpec(
    block_shape=(row_block_size, x.shape[1]),
    index_map=lambda i: (i * row_block_size, 0),
  ),
)(x).block_until_ready()
