# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_shape = (4000,)
dim = 1
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, *input_shape))
block_size_batch = 8
col_block_size = 128


# Computation
def kernel(x_ref, out_ref):
  """
  Computes the exclusive cumulative sum along the second axis of the input.
  """
  accumulator = jnp.zeros(x_ref.shape[0], dtype=x_ref.dtype)

  for i in range(0, x_ref.shape[1], col_block_size):
    x_tile = x_ref[:, i : i + col_block_size]
    out_tile = jnp.empty_like(x_tile)

    def body(j, state):
      acc, out_t = state
      out_t = out_t.at[:, j].set(acc)
      # Mask updates to accumulator for the final, partial tile
      in_bounds = i + j < x_ref.shape[1]
      update = jnp.where(in_bounds, x_tile[:, j], 0.0)
      acc = acc + update
      return acc, out_t

    accumulator, out_tile = lax.fori_loop(0, col_block_size, body, (accumulator, out_tile))

    out_ref[:, i : i + col_block_size] = out_tile


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[0] // block_size_batch,),
  in_specs=[pl.BlockSpec((block_size_batch, x.shape[1]), lambda i: (i * block_size_batch, 0))],
  out_specs=pl.BlockSpec((block_size_batch, x.shape[1]), lambda i: (i * block_size_batch, 0)),
)(x).block_until_ready()
