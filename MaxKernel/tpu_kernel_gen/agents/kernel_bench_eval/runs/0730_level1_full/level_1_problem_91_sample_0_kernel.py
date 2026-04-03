# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
dim = 1
batch_size = 128
input_shape = (4000,)
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, *input_shape))


def kernel(x_ref, out_ref):
  """Pallas kernel for reverse cumulative sum."""
  # Load the input data from HBM into SRAM.
  x = x_ref[...]

  # The Python for loop is unrolled by the compiler.
  for i in range(x.shape[0]):
    # Process one row at a time.
    row = x[i]

    # Use lax.scan to compute the reverse cumulative sum on the 1D row.
    def body_fn(carry, x_slice):
      new_carry = carry + x_slice
      return new_carry, new_carry

    # Scan over the 1D row.
    _, result_row = lax.scan(
      body_fn,
      init=jnp.zeros((), dtype=row.dtype),
      xs=row,
      reverse=True,
    )
    # Write the resulting row to the output buffer.
    out_ref[i, :] = result_row


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // 8,),
  in_specs=[pl.BlockSpec(block_shape=(8, input_shape[0]), index_map=lambda i: (i * 8, 0))],
  out_specs=pl.BlockSpec(block_shape=(8, input_shape[0]), index_map=lambda i: (i * 8, 0)),
)(x).block_until_ready()
