# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for row-wise softmax.

  This kernel computes the softmax function for each row of the input matrix.
  Each program instance handles one row. The computation is tiled to handle
  large row dimensions that may not fit into SRAM.

  Args:
    x_ref: A reference to the full input array in HBM.
    out_ref: A reference to the full output array in HBM.
  """
  # Each program instance processes a single row. We use `program_id` to
  # determine which row this instance is responsible for.
  row_idx = pl.program_id(axis=0)

  # Define a tile size for processing chunks of the row. This must be small
  # enough for intermediates to fit in SRAM. 1024 is a safe choice.
  TILE_SIZE = 1024

  # Pass 1: Find the maximum value in the row for numerical stability.
  # This is done in a tiled fashion to avoid loading the whole row at once.
  row_max = -jnp.inf
  for i in range(0, dim, TILE_SIZE):
    # Load a chunk of the input row from HBM to SRAM.
    chunk = pl.load(x_ref, (row_idx, pl.dslice(i, TILE_SIZE)))
    # Update the max.
    row_max = jnp.maximum(row_max, jnp.max(chunk))

  # Pass 2: Compute the denominator (sum of exponentials).
  # This also uses tiling.
  denominator = 0.0
  for i in range(0, dim, TILE_SIZE):
    chunk = pl.load(x_ref, (row_idx, pl.dslice(i, TILE_SIZE)))
    # Subtract max, exponentiate, and sum up.
    denominator += jnp.sum(jnp.exp(chunk - row_max))

  # Pass 3: Compute and store the final softmax results.
  # We iterate through the row one last time to compute and store the results.
  for i in range(0, dim, TILE_SIZE):
    chunk = pl.load(x_ref, (row_idx, pl.dslice(i, TILE_SIZE)))
    # Compute the softmax value for the chunk.
    output_chunk = jnp.exp(chunk - row_max) / denominator
    # Store the result back to the output buffer in HBM.
    pl.store(out_ref, (row_idx, pl.dslice(i, TILE_SIZE)), output_chunk)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size,),
  # We provide a reference to the *entire* arrays to the kernel.
  # The kernel then uses pl.load and pl.store with 2D indices to move tiles
  # between HBM and SRAM. This avoids staging large blocks and works with
  # the natural 2D layout of the data.
  in_specs=[
    pl.BlockSpec(x.shape, lambda i: (0, 0)),
  ],
  out_specs=pl.BlockSpec(x.shape, lambda i: (0, 0)),
)(x).block_until_ready()
