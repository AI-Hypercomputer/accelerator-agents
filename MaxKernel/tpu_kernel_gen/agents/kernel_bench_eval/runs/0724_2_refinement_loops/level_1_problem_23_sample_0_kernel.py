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
  # Each program in the grid computes softmax for a single row.
  # The program's index in the grid determines which row it processes.
  i = pl.program_id(axis=0)

  # Load the row from HBM into SRAM.
  row = x_ref[i, :]

  # Compute softmax in a numerically stable way.
  # 1. Find the maximum value in the row.
  max_val = jnp.max(row)

  # 2. Subtract the max and exponentiate.
  numerator = jnp.exp(row - max_val)

  # 3. Compute the sum of the exponentiated values (the denominator).
  denominator = jnp.sum(numerator)

  # 4. Divide to get the final result and write it to the output.
  out_ref[i, :] = numerator / denominator


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(batch_size, dim), index_map=lambda i: (0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(batch_size, dim), index_map=lambda i: (0, 0)),
)(x).block_until_ready()
