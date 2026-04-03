# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_shape = (4000,)
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, *input_shape))
# Each kernel instance will process a block of 8 rows to satisfy TPU constraints.
batch_block_size = 8


# Computation
def kernel(x_ref, out_ref):
  """
  Computes the cumulative sum for a block of the input tensor.
  """
  # The input x_ref is a block of shape (batch_block_size, 4000).
  # We compute the cumulative sum along the second axis (the feature dimension)
  # for this block and write the result to the corresponding output block.
  # The `pallas_call` ensures that each kernel instance processes a unique
  # block of rows from the input tensor.
  # The jnp.cumsum primitive is not supported in Pallas, so we implement it
  # manually using jax.lax.scan.
  x = x_ref[...]

  def body_fn(carry, x_col):
    new_carry = carry + x_col
    return new_carry, new_carry

  # We scan over the columns of x.
  # x has shape (batch_block_size, 4000)
  # x.T has shape (4000, batch_block_size)
  # The initial carry is a zero vector of shape (batch_block_size,)
  # The output of scan will be transposed, so we transpose it back.
  _, y = jax.lax.scan(body_fn, jnp.zeros(x.shape[0], dtype=x.dtype), x.T)
  out_ref[...] = y.T


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(x.shape[0] // batch_block_size,),
  in_specs=[
    pl.BlockSpec(
      block_shape=(batch_block_size, x.shape[1]),
      index_map=lambda i: (i * batch_block_size, 0),
    )
  ],
  out_specs=pl.BlockSpec(
    block_shape=(batch_block_size, x.shape[1]),
    index_map=lambda i: (i * batch_block_size, 0),
  ),
)(x).block_until_ready()
