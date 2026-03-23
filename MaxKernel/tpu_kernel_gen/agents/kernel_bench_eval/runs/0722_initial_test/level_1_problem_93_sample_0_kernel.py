# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_shape = (4000,)
key = random.PRNGKey(0)
key_x, key_mask = random.split(key)
x = random.normal(key_x, (batch_size, *input_shape))
mask = random.randint(key_mask, x.shape, 0, 2).astype(jnp.bool_)

# Define the number of rows each program will handle. This must be a multiple of 8
# to satisfy the TPU block shape constraints.
ROWS_PER_PROGRAM = 8


# Computation
def kernel(x_ref, mask_ref, out_ref):
  """
  Pallas kernel for computing cumulative sum on masked input.

  This kernel computes `jnp.cumsum(x * mask, axis=1)`. Each program instance
  handles a block of 8 rows.

  Args:
    x_ref: A reference to a block of the input tensor 'x'.
    mask_ref: A reference to a block of the input tensor 'mask'.
    out_ref: A reference to a block of the output tensor to be written to.
  """
  # Iterate over each row in the block assigned to this program.
  for i in range(x_ref.shape[0]):
    # Perform the cumulative sum on the entire row. This is a vectorized
    # operation that avoids the "cannot store scalars to VMEM" error.
    out_ref[i, :] = jnp.cumsum(x_ref[i, :] * mask_ref[i, :])


# The `cum` tensor is removed as it's not needed when each program handles a full row.
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  # Grid is now smaller, as each program handles multiple rows.
  grid=(batch_size // ROWS_PER_PROGRAM,),
  in_specs=[
    pl.BlockSpec(block_shape=(ROWS_PER_PROGRAM, x.shape[1]), index_map=lambda i: (i * ROWS_PER_PROGRAM, 0)),
    pl.BlockSpec(block_shape=(ROWS_PER_PROGRAM, x.shape[1]), index_map=lambda i: (i * ROWS_PER_PROGRAM, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(ROWS_PER_PROGRAM, x.shape[1]), index_map=lambda i: (i * ROWS_PER_PROGRAM, 0)),
  interpret=False,
)(x, mask).block_until_ready()
